"""Encodes raw MS images with tensorflow and directly trains classifiers on the encoded features.
This results in (number of modalities) * (number of different encoders) * (number of different classifiers) results,
although there is the option to only use the MS1 modality (itms). This is also the default."""
import argparse
import traceback
from functools import partial

from tensorflow.keras.models import Sequential
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from mstc.processing import Compose, HubEncoder, Map, PNGReader
from .run_classification import HUB_MODULES, PATTERN, homogenize_names, RANDOM_STATE, SCORING, subdict, PARAMETER_GRID, train_test_split_grouped, compute_scores, sizedict
import traceback
import sys
import os
import pandas as pd
from functools import partial

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold
import numpy as np
import pytorch_lightning as pl
from mstc.processing.model import MLPClassifier

from mstc.processing import Compose, HubEncoder, Map, PNGReader
from mstc.processing.model import HubModel
from mstc.learning import generate_cross_validation_pipeline
assert sys.version_info >= (3, 6)
os.environ["KMP_WARNINGS"] = "FALSE"
import glob
import json
import logging
import os
import re
import sys
import warnings
import argparse
from collections import OrderedDict
from functools import partial
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import GroupKFold
import re
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from pathlib import Path
from mstc.processing.model import MLPClassifier
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy as np
import plac
import xarray as xr
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    brier_score_loss, log_loss,
    confusion_matrix, recall_score
)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from xgboost.sklearn import XGBClassifier

from mstc.learning import generate_cross_validation_pipeline
from mstc.processing import Flatten, Stacker


#tf.logging.set_verbosity('CRITICAL')
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
os.environ["KMP_WARNINGS"] = "FALSE"



def run_all_encodings_on_all_modalities(input_directory, output_directory, batch_size=4, index_csv=None, annotation_csv=None, patient_mapping=None, n_jobs=8, all_modalities=False):
    labels = pd.read_csv(annotation_csv)
    index_csv = pd.read_csv(index_csv)
    patient_mapping = pd.read_excel(patient_mapping, engine='openpyxl', skiprows=1, index_col="Run")
    # homogenize the index
    patient_mapping.index = patient_mapping.index.map(homogenize_names)
    labels['Raw_ID'] = labels['Raw_ID'].apply(homogenize_names)
    labels.drop(["Tissue"], axis=1, inplace=True)
    labels = pd.merge(labels, index_csv, on='PPPB_ID', how='inner')

    output_directory = os.path.abspath(os.path.expanduser(output_directory))
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        logger.info(f'Created output directory {output_directory}')
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
    # classifiers
    classifier_pipeline = partial(
        generate_cross_validation_pipeline,
        folds=6,
        repeats=2,
        random_state=RANDOM_STATE,
        number_of_jobs=n_jobs,
        scoring=SCORING,
        refit='AUC',
    )

    classifiers = {
        'LogisticRegression': classifier_pipeline(
            LogisticRegression(solver='lbfgs', max_iter=300),
            subdict(PARAMETER_GRID, ['C']),
        ),
        'SVC': classifier_pipeline(
            SVC(gamma='auto', probability=True),
            subdict(PARAMETER_GRID, ['C', 'kernel']),
        ),
        'RandomForest': classifier_pipeline(
            RandomForestClassifier(),
            subdict(PARAMETER_GRID, ['n_estimators']),
        ),
        'XGBoost': classifier_pipeline(
            XGBClassifier(),
            subdict(PARAMETER_GRID, ['n_estimators']),
        ),

    }

    for module, url in HUB_MODULES.items():
        try:
            logger.info(
                f'{module} encoding starts '
                f'({HUB_MODULES.index.get_loc(module)+1}/{len(HUB_MODULES)})'
            )
            # each encoding of all modalities consumes reader
            # so read again instead of keeping in memory with BroadcastMap
            modalities_encoder = Map(
                HubEncoder(url, batch_size=batch_size,
                           encoder_module_name=module)
            )
            pipeline = Compose(
                [modalities_reader, modalities_encoder],

                pipeline='for encoder, map encoder over all read modalities',
                pipeline_output='single modality, single encoder'
            )

            def is_encoding_required(pattern):
                """function to filter glob_patterns with logging side effect"""
                modality = pattern.split('*')[1]
                if not all_modalities and modality != 'itms':
                    return False
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

                debug_array = modality_array[0]
                train_index, test_index = train_test_split_grouped(modality_array.indexes['sample'], pppb_to_patient, test_size=0.3)
                # SPLITTING
                X_train, X_test = modality_array.sel(sample=train_index), modality_array.sel(sample=test_index)
                y_train, y_test = labels.set_index('Raw_ID').loc[train_index]['Tissue'].values, labels.set_index('Raw_ID').loc[test_index]['Tissue'].values
                # TRAINING
                encoded_image_size = modality_array.attrs['encoded_image_size']

                for classifier, pipeline in classifiers.items():
                    name = '-'.join([cohort_identifier, module, modality, classifier, 'results'])
                    cv_path = os.path.join(output_directory, name + '.csv')
                    json_path = os.path.join(output_directory, name + '.json')
                    # check if trained results available already
                    if os.path.exists(json_path):
                        logger.info(f'skipping existing {name}')
                        continue
                    else:
                        logger.info(f'computing {name}')

                    # train
                    pipeline.fit(X_train, y_train)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        cv_df = pd.DataFrame(pipeline.steps[1][1].cv_results_)
                    cv_index = int(pipeline.steps[1][1].best_index_)
                    training_scores = cv_df.loc[cv_index, [
                        'mean_test_AUC', 'mean_test_Accuracy', 'mean_test_F1',
                        'mean_train_AUC', 'mean_train_Accuracy', 'mean_train_F1'
                    ]].to_dict()
                    # run valitation
                    validation_scores = compute_scores(y_test, X_test, pipeline)
                    # collect results
                    results = {
                        'raw_image_size':
                            sizedict(cohort_identifier.split('_')[-1].split('x')),
                        'encoded_image_size':
                            encoded_image_size,
                        #'encoded_features_size': encoded_features_size,
                        # 'non_varying_features':
                        #     int(sum(pipeline.steps[0][1]._get_support_mask())),
                        'cohort_identifier': cohort_identifier,
                        'module': module,
                        'modality': modality,
                        'classifier': classifier,
                        'cv_index': cv_index,
                        'training_scores': training_scores,
                        'validation_scores': validation_scores,
                    }
                    # write to disk
                    cv_df.to_csv(cv_path)
                    with open(json_path, 'w') as open_file:
                        json.dump(results, open_file, default=str)
                    logger.info(f'{name}: {validation_scores}')
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except Exception:
            logger.warn(f'FAIL with module {module} (url: {url})')
            traceback.print_exc()

    logger.info('Processing done.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run encoding on all modalities and train classifiers on the encoded features.')
    parser.add_argument('--input-directory', type=str, required=True, help='Input directory with raw MS images to encode')
    parser.add_argument('--output-directory', type=str, required=True, help='Output directory to save the encoded images and results to')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for encoding images')
    parser.add_argument('--index-csv', type=str, required=True, help='Index CSV file with PPPB_ID and sample name mapping')
    parser.add_argument('--annotation-csv', type=str, required=True, help='Annotation csv file with tissue labels')
    parser.add_argument('--patient-mapping', type=str, required=True, help='Patient mapping file (.xlsx) with PPPB_ID and patient ID mapping')
    parser.add_argument('--all-modalities', action='store_true', default=False, help='Whether to use all modalities or only MS1')
    args = parser.parse_args()

    run_all_encodings_on_all_modalities(
        input_directory=args.input_directory,
        output_directory=args.output_directory,
        batch_size=args.batch_size,
        index_csv=args.index_csv,
        annotation_csv=args.annotation_csv,
        patient_mapping=args.patient_mapping,
        all_modalities=args.all_modalities
    )