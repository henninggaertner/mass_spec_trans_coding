"""Encoding of directory of raw ms images.
Writing an xr.DataArray for each modality encoded with each hub module."""
import traceback
from functools import partial

import torch.cuda
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold
import numpy as np

from mstc.processing import Compose, HubEncoder, Map, PNGReader, HubModel
from mstc.processing.model import HUB_MODELS
from run_learning_ppp1 import homogenize_names, train_test_split_grouped
import pandas as pd
import pytorch_lightning as pl
import os
import re
import logging

#assert sys.version_info >= (3, 6)

# HUB_MODULES = pd.Series(OrderedDict([
#     # 1-10
#     ('inception_v3_imagenet', 'https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1'),  # noqa
#     # # ('mobilenet_v2', 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2')  # noqa
#     ('mobilenet_v2_100_224', 'https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/2'),  # noqa
#     ('inception_resnet_v2', 'https://tfhub.dev/google/imagenet/inception_resnet_v2/feature_vector/1'),  # noqa
#     ('resnet_v2_50', 'https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/1'),  # noqa
#     ('resnet_v2_152', 'https://tfhub.dev/google/imagenet/resnet_v2_152/feature_vector/1'),  # noqa
#     ('mobilenet_v2_140_224', 'https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/2'),  # noqa
#     ('pnasnet_large', 'https://tfhub.dev/google/imagenet/pnasnet_large/feature_vector/2'),  # noqa
#     ('mobilenet_v2_035_128', 'https://tfhub.dev/google/imagenet/mobilenet_v2_035_128/feature_vector/2'),  # noqa
#     ('mobilenet_v1_100_224', 'https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/feature_vector/1'),  # noqa
#     # 11-20
#     ('mobilenet_v1_050_224', 'https://tfhub.dev/google/imagenet/mobilenet_v1_050_224/feature_vector/1'),  # noqa
#     ('mobilenet_v2_075_224', 'https://tfhub.dev/google/imagenet/mobilenet_v2_075_224/feature_vector/2'),  # noqa
#     # # ('inception_v3', 'https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/2')  # noqa
#     ('resnet_v2_101', 'https://tfhub.dev/google/imagenet/resnet_v2_101/feature_vector/1'),  # noqa
#     # # ('quantops', 'https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/quantops/feature_vector/1'),  # noqa
#     ('nasnet_large', 'https://tfhub.dev/google/imagenet/nasnet_large/feature_vector/1'),  # noqa
#     ('mobilenet_v2_100_96', 'https://tfhub.dev/google/imagenet/mobilenet_v2_100_96/feature_vector/2'),  # noqa
#     ('inception_v1', 'https://tfhub.dev/google/imagenet/inception_v1/feature_vector/1'),  # noqa
#     ('mobilenet_v2_035_224', 'https://tfhub.dev/google/imagenet/mobilenet_v2_035_224/feature_vector/2'),  # noqa
#     ('mobilenet_v2_050_224', 'https://tfhub.dev/google/imagenet/mobilenet_v2_050_224/feature_vector/2'),  # noqa
#     # 21-30
#     ('mobilenet_v2_100_128', 'https://tfhub.dev/google/imagenet/mobilenet_v2_100_128/feature_vector/2'),  # noqa
#     ('nasnet_mobile', 'https://tfhub.dev/google/imagenet/nasnet_mobile/feature_vector/1'),  # noqa
#     ('inception_v3_inaturalist', 'https://tfhub.dev/google/inaturalist/inception_v3/feature_vector/1'),  # noqa
#     ('mobilenet_v1_025_128', 'https://tfhub.dev/google/imagenet/mobilenet_v1_025_128/feature_vector/1'),  # noqa
#     ('mobilenet_v2_050_128', 'https://tfhub.dev/google/imagenet/mobilenet_v2_050_128/feature_vector/2'),  # noqa
#     ('inception_v2', 'https://tfhub.dev/google/imagenet/inception_v2/feature_vector/1'),  # noqa
#     ('mobilenet_v1_025_224', 'https://tfhub.dev/google/imagenet/mobilenet_v1_025_224/feature_vector/1'),  # noqa
#     ('mobilenet_v2_075_96', 'https://tfhub.dev/google/imagenet/mobilenet_v2_075_96/feature_vector/2'),  # noqa
#     ('mobilenet_v1_100_128', 'https://tfhub.dev/google/imagenet/mobilenet_v1_100_128/feature_vector/1'),  # noqa
#     ('mobilenet_v1_050_128', 'https://tfhub.dev/google/imagenet/mobilenet_v1_050_128/feature_vector/1'),  # noqa
#     # other
#     ('amoebanet_a_n18_f448', 'https://tfhub.dev/google/imagenet/amoebanet_a_n18_f448/feature_vector/1'),  # noqa
# ]))


#tf.logging.set_verbosity('CRITICAL')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
#torch.set_float32_matmul_precision('high')

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
    # # classifiers
    # classifier_pipeline = partial(
    #     generate_cross_validation_pipeline,
    #     folds=6,
    #     repeats=2,
    #     random_state=RANDOM_STATE,
    #     number_of_jobs=n_jobs,
    #     scoring=SCORING,
    #     refit='AUC',
    # )

    # classifiers = {
    #     'LogisticRegression': classifier_pipeline(
    #         LogisticRegression(solver='lbfgs', max_iter=300),
    #         subdict(PARAMETER_GRID, ['C']),
    #     ),
    #     'SVC': classifier_pipeline(
    #         SVC(gamma='auto', probability=True),
    #         subdict(PARAMETER_GRID, ['C', 'kernel']),
    #     ),
    #     'RandomForest': classifier_pipeline(
    #         RandomForestClassifier(),
    #         subdict(PARAMETER_GRID, ['n_estimators']),
    #     ),
    #     'XGBoost': classifier_pipeline(
    #         XGBClassifier(),
    #         subdict(PARAMETER_GRID, ['n_estimators']),
    #     ),
    #     '3LP': classifier_pipeline(
    #         KerasClassifier(build_fn=create_model, epochs=100, batch_size=10, verbose=0),
    #         {},
    #     )
    #
    # }

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
                #encoded_image_size = modality_array.attrs['encoded_image_size']
                dataloader_args = {'batch_size': 16, 'num_workers': 16}
                trainer_args = {'max_epochs': 3, 'accelerator': 'gpu'}
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
                    X_val_fold = X_train_tensor[val_idx]
                    y_val_fold = y_train_tensor[val_idx]

                    train_dataset = TensorDataset(X_train_fold, y_train_fold)
                    val_dataset = TensorDataset(X_val_fold, y_val_fold)

                    train_loader = DataLoader(train_dataset, shuffle=True, **dataloader_args)
                    val_loader = DataLoader(val_dataset, shuffle=False, **dataloader_args)

                    model = HubModel(repo=source, hub_model_name=model_name, freeze_base_model=freeze_base_model)
                    trainer = pl.Trainer(**trainer_args)

                    trainer.fit(model, train_loader)

                    # Evaluate the model on the validation set
                    val_results = trainer.test(model, val_loader)
                    fold_results.append(val_results[0])  # Assuming test() returns a list with one dict

                    # Calculate average performance across folds
                avg_performance = {metric: np.mean([result[metric] for result in fold_results]) for metric in
                                   fold_results[0].keys()}
                print(f"Average performance across folds: {avg_performance}")
                # TESTING (remains the same)
                X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).permute(0, 3, 1, 2)
                y_test_encoded = label_encoder.transform(y_test)
                y_test_tensor = torch.tensor(y_test_encoded, dtype=torch.long)

                test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
                test_loader = DataLoader(test_dataset, shuffle=False, **dataloader_args)

                # Train a final model on all training data
                full_train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
                full_train_loader = DataLoader(full_train_dataset, shuffle=True, **dataloader_args)
                final_model = HubModel(repo=source, hub_model_name=model_name, freeze_base_model=freeze_base_model)
                final_trainer = pl.Trainer(**trainer_args)
                final_trainer.fit(final_model, full_train_loader)

                # Test the final model
                test_results = final_trainer.test(final_model, test_loader)
                print(f"Final test results: {test_results}")



                # END OF NEW CODE
                # for classifier, pipeline in classifiers.items():
                #     name = '-'.join([cohort_identifier, module, modality, classifier, 'results'])
                #     cv_path = os.path.join(output_directory, name + '.csv')
                #     json_path = os.path.join(output_directory, name + '.json')
                #     # check if trained results available already
                #     if os.path.exists(json_path):
                #         logger.info(f'skipping existing {name}')
                #         continue
                #     else:
                #         logger.info(f'computing {name}')
                #
                #     # train
                #     pipeline.fit(X_train, y_train)
                #     with warnings.catch_warnings():
                #         warnings.simplefilter("ignore")
                #         cv_df = pd.DataFrame(pipeline.steps[2][1].cv_results_)
                #     cv_index = int(pipeline.steps[2][1].best_index_)
                #     training_scores = cv_df.loc[cv_index, [
                #         'mean_test_AUC', 'mean_test_Accuracy', 'mean_test_F1',
                #         'mean_train_AUC', 'mean_train_Accuracy', 'mean_train_F1'
                #     ]].to_dict()
                #     # run valitation
                #     validation_scores = compute_scores(y_test, X_test, pipeline)
                #     # collect results
                #     results = {
                #         'raw_image_size':
                #             sizedict(cohort_identifier.split('_')[-1].split('x')),
                #         'encoded_image_size':
                #             encoded_image_size,
                #         #'encoded_features_size': encoded_features_size,
                #         'non_varying_features':
                #             int(sum(pipeline.steps[0][1]._get_support_mask())),
                #         'cohort_identifier': cohort_identifier,
                #         'module': module,
                #         'modality': modality,
                #         'classifier': classifier,
                #         'cv_index': cv_index,
                #         'training_scores': training_scores,
                #         'validation_scores': validation_scores,
                #     }
                #     # write to disk
                #     cv_df.to_csv(cv_path)
                #     with open(json_path, 'w') as open_file:
                #         json.dump(results, open_file)
                #     logger.info(f'{name}: {validation_scores}')

                # filename = os.path.join(output_directory, name + '.nc')
                #
                # modality_array.to_netcdf(filename)
                # logger.info(f'{name}.nc was written')
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except Exception:
            logger.warn(f'FAIL with module {module}')
            traceback.print_exc()

    logger.info('Processing done.')


if __name__ == "__main__":
    data_dir = "/home/henning/mass_spec_trans_coding/data/" # TODO magic path
    annotation_csv = data_dir+"annotation.csv"
    index_csv = data_dir+"index.csv"
    input_directory = data_dir+"ppp1_raw_image_512x512"
    output_directory = data_dir+"output_fused_encoder_classifier"
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

