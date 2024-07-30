"""Encoding of directory of raw ms images.
Writing an xr.DataArray for each modality encoded with each hub module."""
import traceback
from functools import partial

from tensorflow.keras.models import Sequential
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from mstc.processing import Compose, HubEncoder, Map, PNGReader
from run_learning_ppp1 import *

assert sys.version_info >= (3, 6)

HUB_MODULES = pd.Series(OrderedDict([
    # 1-10
    ('inception_v3_imagenet', 'https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1'),  # noqa
    # # ('mobilenet_v2', 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2')  # noqa
    # ('mobilenet_v2_100_224', 'https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/2'),  # noqa
    # ('inception_resnet_v2', 'https://tfhub.dev/google/imagenet/inception_resnet_v2/feature_vector/1'),  # noqa
    # ('resnet_v2_50', 'https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/1'),  # noqa
    # ('resnet_v2_152', 'https://tfhub.dev/google/imagenet/resnet_v2_152/feature_vector/1'),  # noqa
    # ('mobilenet_v2_140_224', 'https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/2'),  # noqa
    # ('pnasnet_large', 'https://tfhub.dev/google/imagenet/pnasnet_large/feature_vector/2'),  # noqa
    # ('mobilenet_v2_035_128', 'https://tfhub.dev/google/imagenet/mobilenet_v2_035_128/feature_vector/2'),  # noqa
    # ('mobilenet_v1_100_224', 'https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/feature_vector/1'),  # noqa
    # # 11-20
    # ('mobilenet_v1_050_224', 'https://tfhub.dev/google/imagenet/mobilenet_v1_050_224/feature_vector/1'),  # noqa
    # ('mobilenet_v2_075_224', 'https://tfhub.dev/google/imagenet/mobilenet_v2_075_224/feature_vector/2'),  # noqa
    # # # ('inception_v3', 'https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/2')  # noqa
    # ('resnet_v2_101', 'https://tfhub.dev/google/imagenet/resnet_v2_101/feature_vector/1'),  # noqa
    # # # ('quantops', 'https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/quantops/feature_vector/1'),  # noqa
    # ('nasnet_large', 'https://tfhub.dev/google/imagenet/nasnet_large/feature_vector/1'),  # noqa
    # ('mobilenet_v2_100_96', 'https://tfhub.dev/google/imagenet/mobilenet_v2_100_96/feature_vector/2'),  # noqa
    # ('inception_v1', 'https://tfhub.dev/google/imagenet/inception_v1/feature_vector/1'),  # noqa
    # ('mobilenet_v2_035_224', 'https://tfhub.dev/google/imagenet/mobilenet_v2_035_224/feature_vector/2'),  # noqa
    # ('mobilenet_v2_050_224', 'https://tfhub.dev/google/imagenet/mobilenet_v2_050_224/feature_vector/2'),  # noqa
    # # 21-30
    # ('mobilenet_v2_100_128', 'https://tfhub.dev/google/imagenet/mobilenet_v2_100_128/feature_vector/2'),  # noqa
    # ('nasnet_mobile', 'https://tfhub.dev/google/imagenet/nasnet_mobile/feature_vector/1'),  # noqa
    # ('inception_v3_inaturalist', 'https://tfhub.dev/google/inaturalist/inception_v3/feature_vector/1'),  # noqa
    # ('mobilenet_v1_025_128', 'https://tfhub.dev/google/imagenet/mobilenet_v1_025_128/feature_vector/1'),  # noqa
    # ('mobilenet_v2_050_128', 'https://tfhub.dev/google/imagenet/mobilenet_v2_050_128/feature_vector/2'),  # noqa
    # ('inception_v2', 'https://tfhub.dev/google/imagenet/inception_v2/feature_vector/1'),  # noqa
    # ('mobilenet_v1_025_224', 'https://tfhub.dev/google/imagenet/mobilenet_v1_025_224/feature_vector/1'),  # noqa
    # ('mobilenet_v2_075_96', 'https://tfhub.dev/google/imagenet/mobilenet_v2_075_96/feature_vector/2'),  # noqa
    # ('mobilenet_v1_100_128', 'https://tfhub.dev/google/imagenet/mobilenet_v1_100_128/feature_vector/1'),  # noqa
    # ('mobilenet_v1_050_128', 'https://tfhub.dev/google/imagenet/mobilenet_v1_050_128/feature_vector/1'),  # noqa
    # # other
    # ('amoebanet_a_n18_f448', 'https://tfhub.dev/google/imagenet/amoebanet_a_n18_f448/feature_vector/1'),  # noqa
]))


#tf.logging.set_verbosity('CRITICAL')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

PATTERN = re.compile(
    r'(?P<sample_name>.+?)(\.mzXML\.gz\.image\.0\.)'
    r'(?P<modality>(itms)|(ms2\.precursor=\d{3,}\.\d{2}))'
    r'\.png'
)




def run_all_encodings_on_all_modalities(input_directory, output_directory, batch_size=4, index_csv=None, annotation_csv=None, patient_mapping=None, n_jobs=8):
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
                X_train, X_test = modality_array.sel(sample=train_index)[0], modality_array.sel(sample=test_index)[0]
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
                        cv_df = pd.DataFrame(pipeline.steps[2][1].cv_results_)
                    cv_index = int(pipeline.steps[2][1].best_index_)
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
                        'non_varying_features':
                            int(sum(pipeline.steps[0][1]._get_support_mask())),
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
                        json.dump(results, open_file)
                    logger.info(f'{name}: {validation_scores}')

                # filename = os.path.join(output_directory, name + '.nc')
                #
                # modality_array.to_netcdf(filename)
                # logger.info(f'{name}.nc was written')
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except Exception:
            logger.warn(f'FAIL with module {module} (url: {url})')
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
        patient_mapping=patient_mapping
    )

