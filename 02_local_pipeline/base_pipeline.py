# [1] Import Libraries
# (1) Import operating system functionalitie
import os
import logging as pylog
logger = pylog.getLogger(__name__)

# (2) Import TensorFlow core libraries
import tensorflow_model_analysis as tfma

# (3) Import TFX components for building an ML pipeline
from tfx.components import CsvExampleGen        # For ingesting CSV data
from tfx.components import StatisticsGen        # For generating statistics on your dataset

from tfx.components import SchemaGen            # For inferring the schema of your dataset
from tfx.v1.components import ImportSchemaGen   # For importing a predefined schema

from tfx.components import ExampleValidator     # For validating data examples against schema

from tfx.components import Transform            # For feature engineering

from tfx.components import Tuner                # For tuning hyperparameters

from tfx.proto import trainer_pb2
from tfx.components import Trainer              # For training models

from tfx.components import Evaluator            # For evaluating model performance

from tfx.components import Pusher               # For deploying models to production

# (4) Import Resolver - Helps locate artifacts in the pipeline
from tfx.dsl.components.common.resolver import Resolver

# (5) Import latest_blessed_model_resolver - Find the latest validated model
from tfx.dsl.experimental import latest_blessed_model_resolver

# (6) Import protobuf message definitions for configuring components
from tfx.proto import example_gen_pb2           # For configuring example generation
from tfx.proto import trainer_pb2               # For configuring model training
from tfx.proto import pusher_pb2                # For configuring model deployment

# (7) Import Channel to define data connections between components
from tfx.types import Channel

# (8) Import standard artifact types for ML models
from tfx.types.standard_artifacts import Model, ModelBlessing

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# [2] Identify Preset Parameters
TRAIN_STEPS = 81
EVAL_STEPS = 41

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# [3] Pipeline Components' Instantiation
def init_components(
                        data_dir,                                   # data directory
                        schema_file,                                # location of the schema.pbtxt file
                        module_file,                                # location of module.py file
                        training_steps=TRAIN_STEPS,                 # number of training steps
                        eval_steps=EVAL_STEPS,                      # number of evaluation steps
                        _serving_model_dir=None,                    # location of serving model
                    ):


    # 3.1. CsvExampleGen Component
    input = example_gen_pb2.Input(splits=[
                                            example_gen_pb2.Input.Split(name='train', pattern='span-{SPAN}/train/*'),
                                            example_gen_pb2.Input.Split(name='eval', pattern='span-{SPAN}/eval/*'),
                                            example_gen_pb2.Input.Split(name='test', pattern='span-{SPAN}/test/*')
                                        ])
    example_gen = CsvExampleGen(input_base=data_dir, input_config=input)
    logger.info("[Pipeline Execution][1] ExampleGen Component")


    # 3.2. StatisticsGen Component
    statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])
    logger.info("[Pipeline Execution][2] StatisticsGen Component")


    # 3.3. SchemaGen Component
    if not os.path.exists(schema_file):         # Conditionally generate or import the schema
        # First Run: Generate the Schema
        logger.info("Schema does not exist! A new one will be generated during pipeline execution.")
        schema_gen = SchemaGen(statistics=statistics_gen.outputs['statistics'], infer_feature_shape=True, exclude_splits=['eval', 'test'])
    else:
        # Subsequent Runs: Import the Schema
        logger.info("Loading the schema...")
        schema_gen = ImportSchemaGen(schema_file=schema_file)
        logger.info(f"Schema loaded successfully from {schema_file}")
    logger.info("[Pipeline Execution][3] SchemaGen Component")


    # 3.4. ExampleValidator Component
    example_validator = ExampleValidator(
                                            statistics=statistics_gen.outputs['statistics'],
                                            schema=schema_gen.outputs['schema']
                                        )
    # Load the anomalies artifacts
    logger.info("[Pipeline Execution][4] ExampleValidator Component -  Anomalies will be reviewed after pipeline execution!")


    # 3.5. Transform Component
    transform = Transform(
                            examples=example_gen.outputs['examples'],
                            schema = schema_gen.outputs['schema'],
                            module_file=module_file
                        )
    logger.info("[Pipeline Execution][5] Transform Component")


    # 3.6. Tuner Component
    tuner = Tuner(
                    module_file=module_file,
                    examples=transform.outputs['transformed_examples'],
                    transform_graph=transform.outputs['transform_graph'],
                    train_args=trainer_pb2.TrainArgs(splits=["train"], num_steps=training_steps),
                    eval_args=trainer_pb2.EvalArgs(splits=["eval"], num_steps=eval_steps)
                )
    logger.info("[Pipeline Execution][6] Tuner Component")


    # 3.7. Trainer Component
    trainer = Trainer(
                        module_file=module_file,
                        transformed_examples=transform.outputs['transformed_examples'],
                        schema=schema_gen.outputs['schema'],
                        transform_graph=transform.outputs['transform_graph'],
                        hyperparameters=tuner.outputs['best_hyperparameters'],
                        train_args=trainer_pb2.TrainArgs(splits=["train"], num_steps=training_steps),
                        eval_args=trainer_pb2.EvalArgs(splits=["eval"], num_steps=eval_steps)
                    )
    logger.info("[Pipeline Execution][7] Trainer Component")
    

    # 3.8. Evaluator Component
    # 1. Get the latest blessed model
    model_resolver = Resolver(
                                strategy_class = latest_blessed_model_resolver.LatestBlessedModelResolver,
                                model = Channel(type=Model),
                                model_blessing = Channel(type=ModelBlessing)
                                ).with_id('latest_blessed_model_resolver')
    
    # 2. Define evaluation configuration for a single model that handles multiple outputs
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
                                    value_threshold=tfma.GenericValueThreshold(
                                        upper_bound={'value': 10.0}
                                    ),
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
                                    value_threshold=tfma.GenericValueThreshold(
                                        upper_bound={'value': 15.0}
                                    ),
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

    # Define Evaluator component
    evaluator = Evaluator(
                            examples=example_gen.outputs["examples"],
                            model=trainer.outputs["model"],
                            baseline_model=model_resolver.outputs["model"],
                            eval_config=eval_config,
                            example_splits=['test']
    )
    logger.info("[Pipeline Execution][8] Evaluator/Blessor Component")
    

    # 3.9. Pusher Component
    pusher = Pusher(
                    model = trainer.outputs["model"],
                    model_blessing = evaluator.outputs["blessing"],
                    push_destination = pusher_pb2.PushDestination(
                        filesystem = pusher_pb2.PushDestination.Filesystem(base_directory = _serving_model_dir)
                    ),
                    custom_config={'transform_graph': transform.outputs['transform_graph']} # pass the transform graph to the pusher
                )
    logger.info("[Pipeline Execution][9] Pusher Component")


    # 3.10. Package the Pipeline Component List
    components = [
                    example_gen,
                    statistics_gen,
                    schema_gen,
                    example_validator,
                    transform,
                    tuner,
                    trainer,
                    evaluator,
                    pusher
                ]

    return components
