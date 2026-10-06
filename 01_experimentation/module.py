# Import Libraries
# ****************
# Import standard OS operations for file handling
from typing import NamedTuple, Dict, Any, Text

# Import TensorFlow, the main deep learning framework
import tensorflow as tf

from keras_tuner import HyperParameters
from keras_tuner.tuners import BayesianOptimization
from keras_tuner.engine import base_tuner


# Import TensorFlow Transform for data preprocessing in TFX pipelines
import tensorflow_transform as tft

# Import layers for building a neural network model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization

# Import Model class to create Keras models
from tensorflow.keras.models import Model

# Import FnArgs which is used to pass arguments to TFX Trainer component
from tfx.components.trainer.fn_args_utils import FnArgs



# Define Features and Labels Keys
# *******************************
FEATURE_KEYS = ['cement', 'bfs', 'fa', 'water', 'sp', 'ca', 'fa.1', 'age']
LABEL_KEYS = ['c_str']



# 1. Data Preprocessing
# *********************
# 1.1. Function to Scale Input Features
def scale_tensor(input_tensor):
    # Normalizes the input tensor to have zero mean and unit variance using z-score standardization
    return tft.scale_to_z_score(input_tensor)

# 1.2. TFX Preprocessing Function
def preprocessing_fn(inputs):
    # This function defines how to transform input features in a TFX pipeline
    outputs = {}

    # 1. Normalize Numeric Features
    for feature in FEATURE_KEYS:
        scaled_feature = scale_tensor(inputs[feature])
        outputs[feature] = tf.reshape(scaled_feature, [-1])

    # 2. Pass Labels Without Modification
    for label in LABEL_KEYS:
        outputs[label] = inputs[label]

    # Return the dictionary of transformed features and unmodified labels
    return outputs



# 2. Model Architecture - Using Hyperparameters
# *********************************************
def build_dnn_model(hp: HyperParameters):
    """
    Builds a DNN model with tunable hyperparameters.

    Args:
        hp (HyperParameters): Hyperparameter search space.

    Returns:
        model (tf.keras.Model): Compiled model.
    """
    # Input Shape: Flat Vector
    input_shape = (len(FEATURE_KEYS),)

    # Input Layer
    inputs = Input(shape=input_shape)
    x = inputs

    # Tune the Network Structure
    for i in range(hp.Int("num_hidden_layers", 1, 4)):
        units = hp.Int(f"dense_units_{i}", min_value=32, max_value=256, step=32)
        activation = hp.Choice(f"activation_{i}", ["relu", "tanh"])
        dropout_rate = hp.Float(f"dropout_{i}", min_value=0.1, max_value=0.5, step=0.1)
        
        x = Dense(units, activation=activation)(x)
        x = BatchNormalization()(x) 
        x = Dropout(dropout_rate)(x)
    
    # Output layer
    outputs = Dense(1, activation=None)(x)

    # Define Model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile Model
    model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                    loss="mse",
                    metrics=[
                                tf.keras.metrics.MeanAbsoluteError(name="mae"),
                                tf.keras.metrics.RootMeanSquaredError(name="rmse")
                            ]
                )

    return model



# 3. Model Tuning
# ***************
TunerFnResult = NamedTuple("TunerFnResult", [("tuner", base_tuner.BaseTuner), ("fit_kwargs", Dict[Text, Any])])

# Tuner Function
def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
    """
    Tuner function to perform hyperparameter search for the DNN model.
    Uses Bayesian Optimization to find the best architecture and learning rate.
    """

    # Load the transform graph
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    # Load transformed training and validation datasets
    train_dataset = input_fn(fn_args.train_files, tf_transform_output, batch_size=32)
    eval_dataset = input_fn(fn_args.eval_files, tf_transform_output, batch_size=32)

    # Define the tuner using Bayesian Optimization
    tuner = BayesianOptimization(
                                    hypermodel=build_dnn_model,
                                    objective='val_loss',
                                    max_trials=3,
                                    num_initial_points=5,
                                    directory=fn_args.working_dir,
                                    project_name='dnn_bayes'
                            )

    # Callbacks for tuning
    stop_early = tf.keras.callbacks.EarlyStopping(
                                                    monitor='val_loss',
                                                    patience=5,
                                                    restore_best_weights=True
                                                )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                                                        monitor='val_loss',
                                                        factor=0.5,
                                                        patience=3,
                                                        min_lr=1e-5
                                                    )

    # Training arguments for tuner
    fit_kwargs = {
                    "x": train_dataset,
                    "validation_data": eval_dataset,
                    "steps_per_epoch": fn_args.train_steps,
                    "validation_steps": fn_args.eval_steps,
                    "epochs": 50,
                    "callbacks": [stop_early, reduce_lr]
    }

    return TunerFnResult(tuner=tuner, fit_kwargs=fit_kwargs)



# 4. Data Reader
# **************
# 4.1. Function that Creates a Record Reader that can Create gzip'ed Files
def gzip_reader_fn(filenames):
    return tf.data.TFRecordDataset(filenames, compression_type="GZIP", num_parallel_reads=tf.data.AUTOTUNE)


# 4.2. Input Function to Separate Features from Labels
def input_fn(file_pattern, tf_transform_output, batch_size=32):
    
    # Extract Transformed Feature Specs from tf_transform_output
    transformed_feature_spec = tf_transform_output.transformed_feature_spec().copy()

    # Prepare the Dataset
    dataset = tf.data.experimental.make_batched_features_dataset(
                                                                    file_pattern=file_pattern,
                                                                    batch_size=batch_size,
                                                                    features=transformed_feature_spec,
                                                                    reader=gzip_reader_fn
                                                                )
    
    # Define a Function to Reshape Features and Combine Labels
    def preprocess_fn(features):
        # Labels
        labels = [tf.cast(features.pop(k), tf.float32) for k in LABEL_KEYS]
        y = tf.stack(labels, axis=-1)
        y = tf.reshape(y, (-1, 1))

        # Features
        xs = [tf.cast(tf.reshape(features[f], [-1]), tf.float32) for f in FEATURE_KEYS]
        x = tf.stack(xs, axis=-1)
        x = tf.ensure_shape(x, [None, len(FEATURE_KEYS)])
        
        return x, y

    # Apply the Preprocessing Function to the Dataset
    dataset = dataset.map(preprocess_fn)
    dataset = dataset.repeat()

    return dataset


# 4.3. Function to Parse Serialized tf.Example
def get_serve_tf_examples_fn(model, tf_transform_output):
    """
    Returns a serving signature function for DNN model.
    Transforms raw tf.Examples and prepares them as input to the trained model.
    """

    # Attach the transform layer for inference
    model.tft_layer = tf_transform_output.transform_features_layer()

    # Use raw spec but remove labels (not needed at serving time)
    serving_feature_spec = tf_transform_output.raw_feature_spec().copy()
    for label_key in LABEL_KEYS:
        serving_feature_spec.pop(label_key, None)
    
    # Function to return the output used in serving signature
    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        # Parse the raw serialized TF examples
        parsed_features = tf.io.parse_example(serialized_tf_examples, serving_feature_spec)
        
        # Apply the same transformation as during training
        transformed_features = model.tft_layer(parsed_features)
        
        # Combine features into a flat vector (batch_size, num_features)
        feature_list = [transformed_features[key] for key in FEATURE_KEYS]
        combined_features = tf.stack(feature_list, axis=-1)
        combined_features = tf.ensure_shape(combined_features, [None, len(FEATURE_KEYS)])

        # Get model predictions
        predictions = model(combined_features)  # Shape: (batch_size, num_labels)
        
        # Build dictionary output
        return {"predictions": tf.squeeze(predictions, axis=-1)}
    
    return serve_tf_examples_fn


# 4.3. Function to Parse Serialized tf.Example
def get_serve_json_fn(model, tf_transform_output):
    """
    Returns a serving signature function for DNN model.
    Transforms raw JSON examples and prepares them as input to the trained model.
    """

    # Attach the transform layer for inference
    model.tft_layer = tf_transform_output.transform_features_layer()

    # Use raw spec but remove labels (not needed at serving time)
    serving_feature_spec = tf_transform_output.raw_feature_spec().copy()
    for label_key in LABEL_KEYS:
        serving_feature_spec.pop(label_key, None)
    
    # Function to return the output used in serving signature
    @tf.function(input_signature=[
        tf.TensorSpec([None], tf.float32, name="cement"),
        tf.TensorSpec([None], tf.float32, name="bfs"),
        tf.TensorSpec([None], tf.float32, name="fa"),
        tf.TensorSpec([None], tf.float32, name="water"),
        tf.TensorSpec([None], tf.float32, name="sp"),
        tf.TensorSpec([None], tf.float32, name="ca"),
        tf.TensorSpec([None], tf.float32, name="fa_1"),   # use fa_1 (no dot)
        tf.TensorSpec([None], tf.int64,  name="age"),
    ])
    def serve_json(cement, bfs, fa, water, sp, ca, fa_1, age):
        # Cast/ensure dtypes
        raw = {
            "cement": cement, "bfs": bfs, "fa": fa, "water": water,
            "sp": sp, "ca": ca, "fa.1": tf.cast(fa_1, tf.float32), "age": age,
        }
        
        # Make rank-2: [batch] -> [batch, 1] to match raw spec (FixedLenFeature([1]))
        raw = {k: tf.expand_dims(tf.cast(v, tf.float32 if k != "age" else tf.int64), -1) for k, v in raw.items()}

        # Apply the same transformation as during training
        transformed = model.tft_layer(raw)
        
        # Combine features into a flat vector (batch_size, num_features)
        features = tf.stack([transformed[k] for k in ['cement','bfs','fa','water','sp','ca','fa.1','age']], axis=-1)

        # Get model predictions
        preds = model(features)  # Shape: (batch_size, num_labels)
        
        # Build dictionary output
        return {"predictions": tf.squeeze(preds, axis=-1)}
    
    return serve_json



# 6. Model Run Function
# *********************
def run_fn(fn_args: FnArgs):
    """
    TFX run function for training and saving the DNN classification model.
    """

    # 1. Load the transform graph
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    
    # 2. Load transformed training and validation datasets
    train_dataset = input_fn(fn_args.train_files, tf_transform_output, batch_size=32)
    eval_dataset = input_fn(fn_args.eval_files, tf_transform_output, batch_size=32)
    
    # 3. Build the model from tuned hyperparameters
    hp = HyperParameters.from_config(fn_args.hyperparameters)
    model = build_dnn_model(hp=hp)
    
    # 4. Define training callbacks
    callbacks = [
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=10,
                        restore_best_weights=True
                    ),
                    tf.keras.callbacks.ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.5,
                        patience=3,
                        min_lr=1e-6
                    )
                ]
    
    # 5. Train the model
    model.fit(
                train_dataset,
                epochs=70,
                steps_per_epoch=fn_args.train_steps,
                validation_data=eval_dataset,
                validation_steps=fn_args.eval_steps,
                shuffle=True,
                callbacks=callbacks
            )

    # 6. Create serving signatures
    serve_tf_example = get_serve_tf_examples_fn(model, tf_transform_output).get_concrete_function(
                                                                                                    tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")
                                                                                                )
    serve_json = get_serve_json_fn(model, tf_transform_output).get_concrete_function()

    signatures = {
                    # IMPORTANT: Evaluator expects this one to take serialized tf.Example bytes
                    "serving_default": serve_tf_example,
                    # Your JSON-friendly entry point for online prediction
                    "serve_json": serve_json,
                }
    
    # 7. Save the trained model
    model.save(fn_args.serving_model_dir, save_format="tf", signatures=signatures)