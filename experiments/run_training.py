"""
Complete training pipeline, taking in the raw MS images and using pytorch lightning to train a model with a classifier on them.
"""
import traceback
from functools import partial

import torch.cuda
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold
import numpy as np

from mstc.processing import Compose, HubEncoder, Map, PNGReader, HubModel, ValidationCallback
from mstc.processing.model import HUB_MODELS
from run_learning_ppp1 import homogenize_names, train_test_split_grouped
import pandas as pd
import pytorch_lightning as pl
import os
import re
import logging

os.environ["KMP_WARNINGS"] = "FALSE"
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

PATTERN = re.compile(
    r'(?P<sample_name>.+?)(\.mzXML\.gz\.image\.0\.)'
    r'(?P<modality>(itms)|(ms2\.precursor=\d{3,}\.\d{2}))'
    r'\.png'
)


def run_all_encodings_on_all_modalities(input_directory, output_directory, batch_size=4, index_csv=None, annotation_csv=None, patient_mapping=None, n_jobs=8, freeze_base_model=False, n_splits=5):
    labels = pd.read_csv(annotation_csv)
    index_csv = pd.read_csv(index_csv)
    patient_mapping = pd.read_excel(patient_mapping, engine='openpyxl', skiprows=1, index_col="Run")
    # homogenize the index
    patient_mapping.index = patient_mapping.index.map(homogenize_names)
    labels['Raw_ID'] = labels['Raw_ID'].apply(homogenize_names)
    labels.drop(["Tissue"], axis=1, inplace=True)
    labels = pd.merge(labels, index_csv, on='PPPB_ID', how='inner')

    output_directory = os.path.abspath(os.path.expanduser(output_directory))
    assert os.path.exists(output_directory)
    data_dir = os.path.abspath(os.path.expanduser(input_directory))

    sample_set = set()
    modality_set = set()
    for filepath in os.listdir(data_dir):
        groupdict = PATTERN.match(filepath).groupdict()
        sample_set.add(groupdict['sample_name'])
        modality_set.add(groupdict['modality'])

    cohort_identifier = os.path.basename(data_dir)
    glob_patterns = [
        os.path.join(data_dir, f'*{modality}*.png')
        for modality in modality_set
    ]

    modalities_reader = Map(
        PNGReader(directory=data_dir), map_reader='read modalities'
    )
    for module, (source, model_name) in HUB_MODELS.items():
        try:
            logger.info(
                f'{module} encoding starts '
                #f'({HUB_MODULES.index.get_loc(module)+1}/{len(HUB_MODULES)})'
            )
            # each encoding of all modalities consumes reader
            # so read again instead of keeping in memory with BroadcastMap
            # modalities_encoder = Map(
            #     HubEncoder(url, batch_size=batch_size,
            #                encoder_module_name=module)
            # )
            pipeline = Compose(
                [modalities_reader],

                pipeline='for encoder, map encoder over all read modalities',
                pipeline_output='single modality, single encoder'
            )

            def is_encoding_required(pattern):
                """function to filter glob_patterns with logging side effect"""
                modality = pattern.split('*')[1]
                if not os.path.exists(os.path.join(
                        output_directory,
                        cohort_identifier + '-' + module + '-' + modality + '.nc'
                )):
                    return True
                else:
                    logger.info(
                        f'skipped modality {modality}, encoding exitst.'
                    )
                    return False
            required_glob_patterns = filter(is_encoding_required, glob_patterns)  # noqa

            for modality_array in pipeline(required_glob_patterns):
                modality = PATTERN.match(
                    modality_array.sample.data[0]
                ).groupdict()['modality']
                name = cohort_identifier + '-' + module + '-' + modality
                modality_array.name = name
                results_csv_path = os.path.join(output_directory, f"{cohort_identifier}_{module}_{modality}_results.csv")
                if os.path.exists(results_csv_path):
                    logger.info(f'Skipping {module} {modality} as results already exist')
                    continue
                sample_index = [homogenize_names(sample) for sample in modality_array.indexes['sample']]
                modality_array = modality_array.assign_coords(sample=sample_index)
                # drop all the samples that are not in the labels
                sample_index = [sample for sample in sample_index if sample in labels['Raw_ID'].values]
                modality_array = modality_array.sel(sample=sample_index)
                pppb_to_patient = {pppb: patient_mapping.loc[pppb]['ID'] for pppb in patient_mapping.index}
                train_index, test_index = train_test_split_grouped(modality_array.indexes['sample'], pppb_to_patient, test_size=0.3)
                # min max scaling
                # modality_array = (modality_array - modality_array.min()) / (modality_array.max() - modality_array.min())
                logging.info(f'Freeze base model: {freeze_base_model}')
                X_train, X_test = modality_array.sel(sample=train_index), modality_array.sel(sample=test_index)
                y_train, y_test = labels.set_index('Raw_ID').loc[train_index]['Tissue'].values, labels.set_index('Raw_ID').loc[test_index]['Tissue'].values

                # Get patient IDs for grouping
                train_patient_ids = [pppb_to_patient[sample] for sample in train_index]
                # TRAINING
                dataloader_args = {'batch_size': 16}
                trainer_args = {'max_epochs': 5, 'accelerator': 'gpu'}
                # trainer_args['callbacks'] = ValidationCallback()
                trainer_args['val_check_interval'] = 1.0

                X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).permute(0, 3, 1, 2)
                label_encoder = LabelEncoder()
                y_train_encoded = label_encoder.fit_transform(y_train)
                y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long)

                repeated_group_kfold = GroupKFold(n_splits=n_splits)
                fold_results = []
                for fold, (train_idx, val_idx) in enumerate(
                        repeated_group_kfold.split(X_train_tensor, y_train_tensor, groups=train_patient_ids)):
                    print(f"Fold {fold + 1}/{n_splits}")

                    X_train_fold = X_train_tensor[train_idx]
                    y_train_fold = y_train_tensor[train_idx]
                    X_test_fold = X_train_tensor[val_idx]
                    y_test_fold = y_train_tensor[val_idx]

                    train_dataset = TensorDataset(X_train_fold, y_train_fold)
                    val_dataset = TensorDataset(X_test_fold, y_test_fold)

                    train_loader = DataLoader(train_dataset, shuffle=True, **dataloader_args)
                    val_loader = DataLoader(val_dataset, shuffle=False, **dataloader_args)

                    model = HubModel(repo=source, hub_model_name=model_name, freeze_base_model=freeze_base_model)
                    trainer = pl.Trainer(**trainer_args)

                    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

                    # Get losses
                    train_losses, val_losses = model.get_losses()

                    # Test the model
                    test_results = trainer.test(model, val_loader)

                    fold_results.append({
                        'fold': fold + 1,
                        'train_losses': train_losses,
                        'val_losses': val_losses,
                        'test_metrics': test_results[0]
                    })
                # TESTING
                X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).permute(0, 3, 1, 2)
                y_test_encoded = label_encoder.transform(y_test)
                y_test_tensor = torch.tensor(y_test_encoded, dtype=torch.long)

                test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
                val_loader = DataLoader(test_dataset, shuffle=False, **dataloader_args)

                # Train a final model on all training data
                full_train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
                full_train_loader = DataLoader(full_train_dataset, shuffle=True, **dataloader_args)
                final_model = HubModel(repo=source, hub_model_name=model_name, freeze_base_model=freeze_base_model)

                final_trainer = pl.Trainer(**trainer_args)
                final_trainer.fit(final_model, full_train_loader)

                # Test the final model
                final_test_results = final_trainer.test(final_model, val_loader)
                print(f"Final test results: {final_test_results}")
                # Compile results into a dataframe
                results_df = pd.DataFrame()
                for fold_result in fold_results:
                    fold_df = pd.DataFrame({
                        'fold': [fold_result['fold']] * len(fold_result['train_losses']),
                        'epoch': range(1, len(fold_result['train_losses']) + 1),
                        'train_loss': fold_result['train_losses'],
                        'val_loss': fold_result['val_losses']
                    })
                    for metric, value in fold_result['test_metrics'].items():
                        fold_df[metric] = value
                    results_df = pd.concat([results_df, fold_df], ignore_index=True)

                # Calculate average performance across folds
                avg_performance = results_df.groupby('epoch').mean().reset_index()
                avg_performance['fold'] = 'average'

                # Combine average performance with fold results and final test results
                final_results_df = pd.concat([results_df, avg_performance], ignore_index=True)
                for metric in final_test_results[0]:
                    final_results_df[f'final_test_{metric}'] = final_test_results[0][metric]


                # Save results to CSV
                results_csv_path = os.path.join(output_directory,
                                                f"{cohort_identifier}_{module}_{modality}_results.csv")
                final_results_df.to_csv(results_csv_path, index=False)
                print(f"Results saved to {results_csv_path}")

        except KeyboardInterrupt:
            logger.info('Interrupted by user')
            break
        except Exception:
            logger.warn(f'FAIL with module {module}')
            traceback.print_exc()

    logger.info('Processing done.')


if __name__ == "__main__":
    data_dir = "/home/henning/mass_spec_trans_coding/data/" # TODO magic path
    annotation_csv = data_dir+"annotation.csv"
    index_csv = data_dir+"index.csv"
    input_directory = data_dir+"ppp1_raw_image_512x512"
    output_directory = data_dir+"output/final"
    patient_mapping = data_dir+"inline-supplementary-material-5.xlsx"

    # Call the function with the modified parameters
    run_all_encodings_on_all_modalities(
        input_directory=input_directory,
        output_directory=output_directory,
        batch_size=4,
        index_csv=index_csv,
        annotation_csv=annotation_csv,
        patient_mapping=patient_mapping,
        freeze_base_model=True,
        n_splits=5
    )

