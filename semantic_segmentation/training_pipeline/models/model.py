import datetime
import os
from typing import List
import absl
import keras_tuner
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tensorflow_transform as tft

from tensorflow_cloud import CloudTuner
from tfx.v1.components import TunerFnResult
from tfx.components.trainer.fn_args_utils import DataAccessor
from tfx.components.trainer.fn_args_utils import FnArgs
from tfx.dsl.io import fileio
from tfx_bsl.tfxio import dataset_options
import tfx.extensions.google_cloud_ai_platform.constants as vertex_const
import tfx.extensions.google_cloud_ai_platform.trainer.executor as vertex_training_const
import tfx.extensions.google_cloud_ai_platform.tuner.executor as vertex_tuner_const

_IMAGE_SHAPE = (128, 128)
_TRAIN_BATCH_SIZE = 64
_EVAL_BATCH_SIZE = 64
_EPOCHS = 2

_IMAGE_KEY = "image"
_LABEL_KEY = "label"


def INFO(text: str):
    absl.logging.info(text)


def _transformed_name(key: str) -> str:
    return key + "_xf"


def _get_signature(model):
    signatures = {
        "serving_default": _get_serve_image_fn(model).get_concrete_function(
            tf.TensorSpec(
                shape=[None, 128, 128, 3],
                dtype=tf.float32,
                name=_transformed_name(_IMAGE_KEY),
            )
        )
    }

    return signatures


def _get_serve_image_fn(model):
    @tf.function
    def serve_image_fn(image_tensor):
        return model(image_tensor)

    return serve_image_fn


# def _image_augmentation(image_features):
#     batch_size = tf.shape(image_features)[0]
#     image_features = tf.image.random_flip_left_right(image_features)
#     image_features = tf.image.resize_with_crop_or_pad(image_features, 250, 250)
#     image_features = tf.image.random_crop(image_features, (batch_size, 224, 224, 3))
#     return image_features


# def _data_augmentation(feature_dict):
#     image_features = feature_dict[_transformed_name(_IMAGE_KEY)]
#     image_features = _image_augmentation(image_features)
#     feature_dict[_transformed_name(_IMAGE_KEY)] = image_features
#     return feature_dict


def _input_fn(
    file_pattern: List[str],
    data_accessor: DataAccessor,
    tf_transform_output: tft.TFTransformOutput,
    is_train: bool = False,
    batch_size: int = 200,
) -> tf.data.Dataset:
    dataset = data_accessor.tf_dataset_factory(
        file_pattern,
        dataset_options.TensorFlowDatasetOptions(
            batch_size=batch_size, label_key=_transformed_name(_LABEL_KEY)
        ),
        tf_transform_output.transformed_metadata.schema,
    )

    # if is_train:
    #     dataset = dataset.map(lambda x, y: (_data_augmentation(x), y))

    # dataset = dataset.batch(batch_size)
    # dataset = dataset.prefetch(tf.data.AUTOTUNE)
    # dataset = dataset.map(_preprocess)
    return dataset


def _build_keras_model(hparams: keras_tuner.HyperParameters) -> tf.keras.Model:
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=[128, 128, 3], include_top=False
    )

    # Use the activations of these layers
    layer_names = [
        "block_1_expand_relu",  # 64x64
        "block_3_expand_relu",  # 32x32
        "block_6_expand_relu",  # 16x16
        "block_13_expand_relu",  # 8x8
        "block_16_project",  # 4x4
    ]
    base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)
    down_stack.trainable = False

    up_stack = [
        upsample(512, 3),  # 4x4 -> 8x8
        upsample(256, 3),  # 8x8 -> 16x16
        upsample(128, 3),  # 16x16 -> 32x32
        upsample(64, 3),  # 32x32 -> 64x64
    ]

    inputs = tf.keras.layers.Input(
        shape=[128, 128, 3], name=_transformed_name(_IMAGE_KEY)
    )

    # Downsampling through the model
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(
        filters=35, kernel_size=3, strides=2, padding="same", name="labels"
    )  # 64x64 -> 128x128

    x = last(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.compile(
        optimizer=Adam(learning_rate=hparams.get("learning_rate")),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy", "sparse_categorical_accuracy"],
    )
    return model


"""
    InstanceNormalization class and upsample function are
    borrowed from pix2pix in [TensorFlow Example repository](
    https://github.com/tensorflow/examples/tree/master/tensorflow_examples/models/pix2pix)
"""


class InstanceNormalization(tf.keras.layers.Layer):
    """Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name="scale",
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(1.0, 0.02),
            trainable=True,
        )

        self.offset = self.add_weight(
            name="offset", shape=input_shape[-1:], initializer="zeros", trainable=True
        )

    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset


def upsample(filters, size, norm_type="batchnorm", apply_dropout=False):
    """Upsamples an input.
    Conv2DTranspose => Batchnorm => Dropout => Relu
    Args:
      filters: number of filters
      size: filter size
      norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
      apply_dropout: If True, adds the dropout layer
    Returns:
      Upsample Sequential Model
    """

    initializer = tf.random_normal_initializer(0.0, 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(
            filters,
            size,
            strides=2,
            padding="same",
            kernel_initializer=initializer,
            use_bias=False,
        )
    )

    if norm_type.lower() == "batchnorm":
        result.add(tf.keras.layers.BatchNormalization())
    elif norm_type.lower() == "instancenorm":
        result.add(InstanceNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


def _get_hyperparameters() -> keras_tuner.HyperParameters:
    hp = keras_tuner.HyperParameters()
    hp.Choice("learning_rate", [1e-3, 1e-2], default=1e-3)
    return hp


# def _build_keras_model(hparams: keras_tuner.HyperParameters) -> tf.keras.Model:
#     base_model = tf.keras.applications.ResNet50(
#         input_shape=(224, 224, 3), include_top=False, weights="imagenet", pooling="max"
#     )
#     base_model.input_spec = None
#     base_model.trainable = False

#     model = tf.keras.Sequential(
#         [
#             tf.keras.layers.InputLayer(
#                 input_shape=(224, 224, 3), name=_transformed_name(_IMAGE_KEY)
#             ),
#             base_model,
#             tf.keras.layers.Dropout(0.1),
#             tf.keras.layers.Dense(10, activation="softmax"),
#         ]
#     )

#     model.compile(
#         loss="sparse_categorical_crossentropy",
#         optimizer=Adam(learning_rate=hparams.get("learning_rate")),
#         metrics=["sparse_categorical_accuracy"],
#     )
#     model.summary(print_fn=INFO)

#     return model


# def cloud_tuner_fn(fn_args: FnArgs) -> TunerFnResult:
#     TUNING_ARGS_KEY = vertex_tuner_const.TUNING_ARGS_KEY
#     TRAINING_ARGS_KEY = vertex_training_const.TRAINING_ARGS_KEY
#     VERTEX_PROJECT_KEY = "project"
#     VERTEX_REGION_KEY = "region"

#     tuner = CloudTuner(
#         _build_keras_model,
#         max_trials=6,
#         hyperparameters=_get_hyperparameters(),
#         project_id=fn_args.custom_config[TUNING_ARGS_KEY][VERTEX_PROJECT_KEY],
#         region=fn_args.custom_config[TUNING_ARGS_KEY][VERTEX_REGION_KEY],
#         objective="val_sparse_categorical_accuracy",
#         directory=fn_args.working_dir,
#     )

#     tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

#     train_dataset = _input_fn(
#         fn_args.train_files,
#         fn_args.data_accessor,
#         tf_transform_output,
#         is_train=True,
#         batch_size=_TRAIN_BATCH_SIZE,
#     )

#     eval_dataset = _input_fn(
#         fn_args.eval_files,
#         fn_args.data_accessor,
#         tf_transform_output,
#         is_train=False,
#         batch_size=_EVAL_BATCH_SIZE,
#     )

#     return TunerFnResult(
#         tuner=tuner,
#         fit_kwargs={
#             "x": train_dataset,
#             "validation_data": eval_dataset,
#             "steps_per_epoch": steps_per_epoch,
#             "validation_steps": fn_args.eval_steps,
#         },
#     )


def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
    steps_per_epoch = _TRAIN_BATCH_SIZE  # int(_TRAIN_DATA_SIZE / _TRAIN_BATCH_SIZE)

    tuner = keras_tuner.RandomSearch(
        _build_keras_model,
        max_trials=6,
        hyperparameters=_get_hyperparameters(),
        allow_new_entries=False,
        objective=keras_tuner.Objective("val_sparse_categorical_accuracy", "max"),
        directory=fn_args.working_dir,
        project_name="img_classification_tuning",
    )

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_dataset = _input_fn(
        fn_args.train_files,
        fn_args.data_accessor,
        tf_transform_output,
        is_train=True,
        batch_size=_TRAIN_BATCH_SIZE,
    )

    eval_dataset = _input_fn(
        fn_args.eval_files,
        fn_args.data_accessor,
        tf_transform_output,
        is_train=False,
        batch_size=_EVAL_BATCH_SIZE,
    )

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            "x": train_dataset,
            "validation_data": eval_dataset,
            "steps_per_epoch": steps_per_epoch,
            "validation_steps": fn_args.eval_steps,
        },
    )


def run_fn(fn_args: FnArgs):
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = _input_fn(
        fn_args.train_files,
        fn_args.data_accessor,
        tf_transform_output,
        is_train=True,
        batch_size=_TRAIN_BATCH_SIZE,
    )

    eval_dataset = _input_fn(
        fn_args.eval_files,
        fn_args.data_accessor,
        tf_transform_output,
        is_train=False,
        batch_size=_EVAL_BATCH_SIZE,
    )

    INFO("Tensorboard logging to {}".format(fn_args.model_run_dir))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=fn_args.model_run_dir, update_freq="batch"
    )

    if fn_args.hyperparameters:
        hparams = keras_tuner.HyperParameters.from_config(fn_args.hyperparameters)
    else:
        hparams = _get_hyperparameters()
    INFO(f"HyperParameters for training: ${hparams.get_config()}")

    model = _build_keras_model(hparams)
    model.fit(
        train_dataset,
        epochs=_EPOCHS,
        steps_per_epoch=_TRAIN_BATCH_SIZE,
        validation_data=eval_dataset,
        validation_steps=_EVAL_BATCH_SIZE,
        callbacks=[tensorboard_callback],
    )

    model.save(
        fn_args.serving_model_dir, save_format="tf", signatures=_get_signature(model)
    )
