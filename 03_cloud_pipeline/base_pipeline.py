# [1] Import Libraries
# (1) Import operating system functionalitie
import logging as pylog
logger = pylog.getLogger(__name__)

# (2) Import TensorFlow Model Analysis for model evaluation
import tensorflow_model_analysis as tfma

# (3) Import TFX components for building an ML pipeline
from tfx.components import CsvExampleGen        # For ingesting CSV data
from tfx.components import StatisticsGen        # For generating statistics on your dataset

from tfx.components import SchemaGen            # For inferring the schema of your dataset
from tfx.v1.components import ImportSchemaGen   # For importing a predefined schema

from tfx.components import ExampleValidator     # For validating data examples against schema

from tfx.components import Transform            # For feature engineering
from tfx.components import Tuner                # For hyperparameter tuning
from tfx.components import Trainer              # For training models
from tfx.components import Evaluator            # For evaluating model performance
from tfx.components import Pusher               # For deploying models to production

# (4) Import Resolver - Helps locate artifacts in the pipeline
from tfx.dsl.components.common.resolver import Resolver

# (5) Import latest_blessed_model_resolver - Find the latest validated model
from tfx.dsl.experimental import latest_blessed_model_resolver

# (6) Import protobuf message definitions for configuring components
from tfx.proto import example_gen_pb2                             # For configuring example generation         
from tfx.proto import trainer_pb2                                 # For configuring model training    
from tfx.proto import pusher_pb2                                  # For configuring model deployment

# (7) Import Channel to define data connections between components
from tfx.types import Channel

# (8) Import standard artifact types for ML models
from tfx.types.standard_artifacts import Model, ModelBlessing

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# [2] Identify Preset Parameters
TRAIN_STEPS = 81
EVAL_STEPS = 41

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# [3] Helper Functions
from google.cloud import storage
import re
def build_window(bucket="your-bucket-name",  # TODO: default bucket; pipeline_run.py passes BUCKET explicitly
                  root="data", k=3, window_dir="_window", require_complete=True, verbose=True):
    """
    Merge the latest k spans under gs://<bucket>/<root>/span-*/ into
    gs://<bucket>/<root>/<window_dir>/{train,val,test}/
    and prefix filenames with span id to avoid collisions.
    """
    client = storage.Client()
    b = client.bucket(bucket)

    # 1. Find Span Dirs
    it = client.list_blobs(bucket, prefix=f"{root}/", delimiter="/")
    for _ in it:  # exhaust to populate it.prefixes
        pass
    span_dirs = [p for p in it.prefixes if re.search(r"(?:^|/)span-0*\d+/$", p)]
    if not span_dirs:
        raise RuntimeError(f"No span-* folders under gs://{bucket}/{root}")

    def span_num(prefix: str) -> int:
        m = re.search(r"(?:^|/)span-0*(\d+)/$", prefix)
        return int(m.group(1)) if m else -1
    
    # 2. Optionally Require Complete Spans
    def has_all_splits(span_prefix: str) -> bool:
        for split in ("train/", "val/", "test/"):
            it2 = client.list_blobs(bucket, prefix=f"{span_prefix}{split}", max_results=1)
            if next(iter(it2), None) is None:
                return False
        return True

    if require_complete:
        span_dirs = [sd for sd in span_dirs if has_all_splits(sd)]
        if not span_dirs:
            raise RuntimeError("All spans are incomplete (missing train/val/test).")

    # 3. Pick Latest k by Numer Span ID
    span_dirs.sort(key=span_num)
    chosen = span_dirs[-k:]
    chosen_ids = [span_num(sd) for sd in chosen]
    if verbose and len(chosen) < k:
        print(f"Requested k={k}, only found {len(chosen)} complete span(s): {chosen_ids}")

    # 4. Clear Old Window
    window_prefix = f"{root}/{window_dir}/"
    for blob in client.list_blobs(bucket, prefix=window_prefix):
        blob.delete()

    # 5. Copy train/val/test into Window
    for sd in chosen:
        sid = span_num(sd)
        for split in ("train", "val", "test"):
            src_prefix = f"{sd}{split}/"
            for blob in client.list_blobs(bucket, prefix=src_prefix):
                base = blob.name.rsplit("/", 1)[-1]           # original filename
                dst = f"{window_prefix}{split}/span-{sid}-{base}"
                b.copy_blob(blob, b, dst)

    if verbose:
        print(f"Built window at gs://{bucket}/{window_prefix} from spans: {chosen_ids}")
    
    return f"gs://{bucket}/{window_prefix}"

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# [4] Pipeline Components' Instantiation
def init_components(
                        bucket_name,                                # bucket name
                        data_dir,                                   # data directory
                        schema_file,                                # location of the schema.pbtxt file
                        module_file,                                # location of module.py file
                        training_steps=TRAIN_STEPS,                 # number of training steps
                        eval_steps=EVAL_STEPS,                      # number of evaluation steps
                        serving_model_dir=None,                     # location of serving model
                ):

    


    # 4.1. CsvExampleGen Component
    # [1] Training Component
    latest_window_base_train = build_window(bucket=bucket_name, root=data_dir, k=3, window_dir="_window_train")
    input_cfg_train = example_gen_pb2.Input(splits=[
                                                    example_gen_pb2.Input.Split(name='train', pattern='train/*.csv'),
                                                    example_gen_pb2.Input.Split(name='val', pattern='val/*.csv'),
                                                    example_gen_pb2.Input.Split(name='test', pattern='test/*.csv')
                                                ])
    example_gen_train = CsvExampleGen(input_base=latest_window_base_train, input_config=input_cfg_train).with_id('train_example_gen')
    logger.info("[Pipeline Execution][1.1] ExampleGen Component - Training Data")

    # [2] Evaluation Component
    latest_window_base_eval = build_window(bucket=bucket_name, root=data_dir, k=1, window_dir="_window_eval")
    input_cfg_eval = example_gen_pb2.Input(splits=[
                                                    example_gen_pb2.Input.Split(name='train', pattern='train/*.csv'),
                                                    example_gen_pb2.Input.Split(name='val', pattern='val/*.csv'),
                                                    example_gen_pb2.Input.Split(name='test', pattern='test/*.csv')
                                                ])
    example_gen_eval = CsvExampleGen(input_base=latest_window_base_eval, input_config=input_cfg_eval).with_id('eval_example_gen')
    logger.info("[Pipeline Execution][1.2] ExampleGen Component - Evaluation Data")


    # 4.2. StatisticsGen Component
    statistics_gen = StatisticsGen(examples=example_gen_train.outputs['examples'])
    logger.info("[Pipeline Execution][2] StatisticsGen Component")


    # 4.3. SchemaGen Component
    schema_gen = SchemaGen(statistics=statistics_gen.outputs['statistics'], infer_feature_shape=True, exclude_splits=['val', 'test'])
    # schema_gen = ImportSchemaGen(schema_file=schema_file)
    logger.info("[Pipeline Execution][3] SchemaGen Component")

    # 4.4. ExampleValidator Component
    example_validator = ExampleValidator(
                                            statistics=statistics_gen.outputs['statistics'],
                                            schema=schema_gen.outputs['schema']
                                        )
    logger.info("[Pipeline Execution][4] ExampleValidator Component -  Anomalies will be reviewed after pipeline execution!")


    # 4.5. Transform Component
    transform = Transform(
                            examples=example_gen_train.outputs['examples'],
                            schema = schema_gen.outputs['schema'],
                            module_file=module_file
                        )
    logger.info("[Pipeline Execution][5] Transform Component")


    # 4.6. Tuner Component
    tuner = Tuner(
                    module_file=module_file,
                    examples=transform.outputs['transformed_examples'],
                    transform_graph=transform.outputs['transform_graph'],
                    train_args=trainer_pb2.TrainArgs(splits=["train"], num_steps=training_steps),
                    eval_args=trainer_pb2.EvalArgs(splits=["val"], num_steps=eval_steps)
                )
    logger.info("[Pipeline Execution][6] Tuner Component")


    # 4.7. Trainer Component
    trainer = Trainer(
                        module_file=module_file,
                        transformed_examples=transform.outputs['transformed_examples'],
                        schema=schema_gen.outputs['schema'],
                        transform_graph=transform.outputs['transform_graph'],
                        hyperparameters=tuner.outputs['best_hyperparameters'],
                        train_args=trainer_pb2.TrainArgs(splits=["train"], num_steps=training_steps),
                        eval_args=trainer_pb2.EvalArgs(splits=["val"], num_steps=eval_steps)
                    )
    logger.info("[Pipeline Execution][7] Trainer Component")


    # 4.8. Model Resolver Component
    model_resolver = Resolver(
                              strategy_class = latest_blessed_model_resolver.LatestBlessedModelResolver,
                              model = Channel(type=Model),
                              model_blessing = Channel(type=ModelBlessing)
                              ).with_id('latest_blessed_model_resolver')


    # 4.9. Evaluator Component
    eval_config = tfma.EvalConfig(
                                    model_specs=[
                                        tfma.ModelSpec(
                                            signature_name='serving_default',
                                            label_key='c_str',
                                            prediction_key="predictions"
                                        )
                                    ],
                                    slicing_specs=[],
                                    metrics_specs=[
                                                    tfma.MetricsSpec(
                                                                        metrics=[
                                                                                    tfma.MetricConfig(
                                                                                        class_name="MeanAbsoluteError",
                                                                                        threshold=tfma.MetricThreshold(
                                                                                            # Threshold for blessing: MAE < 10.0
                                                                                            value_threshold=tfma.GenericValueThreshold(upper_bound={'value': 10.0}),
                                                                                            # For subsequent runs, ensure new model is better than baseline
                                                                                            change_threshold=tfma.GenericChangeThreshold(
                                                                                                direction=tfma.MetricDirection.LOWER_IS_BETTER,
                                                                                                absolute={'value': -0.1}  # New model must improve MAE by at least 0.1
                                                                                            )
                                                                                        )
                                                                                    ),

                                                                                    tfma.MetricConfig(
                                                                                        class_name="RootMeanSquaredError",
                                                                                        threshold=tfma.MetricThreshold(
                                                                                            # Threshold for blessing: RMSE < 15.0
                                                                                            value_threshold=tfma.GenericValueThreshold(upper_bound={'value': 15.0}),
                                                                                            # For subsequent runs, ensure new model is better than baseline
                                                                                            change_threshold=tfma.GenericChangeThreshold(
                                                                                                direction=tfma.MetricDirection.LOWER_IS_BETTER,
                                                                                                absolute={'value': -0.1}  # New model must improve RMSE by at least 0.1
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                ]
                                                                    )
                                                ]
                                )
    
    evaluator_component = Evaluator(
                                        examples=example_gen_eval.outputs["examples"],
                                        model=trainer.outputs["model"],
                                        baseline_model=model_resolver.outputs["model"],
                                        eval_config=eval_config,
                                        example_splits=['test']
                                    )
    logger.info("[Pipeline Execution][9] Evaluator Component")


    # 4.10. Pusher Component
    pusher = Pusher(
                        model = trainer.outputs["model"],
                        model_blessing = evaluator_component.outputs["blessing"],
                        push_destination = pusher_pb2.PushDestination(
                            filesystem = pusher_pb2.PushDestination.Filesystem(base_directory = serving_model_dir)
                        ),
                        custom_config={'transform_graph': transform.outputs['transform_graph']} # pass the transform graph to the pusher
                    )
    logger.info("[Pipeline Execution][10] Pusher Component")


    # 4.11. Package the Pipeline Component List
    components = [
                    example_gen_train,
                    example_gen_eval,
                    statistics_gen,
                    schema_gen,
                    example_validator,
                    transform,
                    tuner,
                    trainer,
                    model_resolver,
                    evaluator_component,
                    pusher
                  ]

    return components