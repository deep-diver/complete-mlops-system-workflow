import os
from typing import List
import absl
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tensorflow_transform as tft

from tfx.components.trainer.fn_args_utils import DataAccessor
from tfx.components.trainer.fn_args_utils import FnArgs
from tfx.components.trainer.rewriting import converters
from tfx.components.trainer.rewriting import rewriter
from tfx.components.trainer.rewriting import rewriter_factory
from tfx.dsl.io import fileio
from tfx_bsl.tfxio import dataset_options

_TRAIN_DATA_SIZE = 128
_EVAL_DATA_SIZE = 128
_TRAIN_BATCH_SIZE = 32
_EVAL_BATCH_SIZE = 32
_CLASSIFIER_LEARNING_RATE = 1e-3
_FINETUNE_LEARNING_RATE = 7e-6
_CLASSIFIER_EPOCHS = 30

_IMAGE_KEY = 'image'
_LABEL_KEY = 'label'

def INFO(text: str):
  absl.logging.info(text)

def _transformed_name(key: str) -> str:
  return key + '_xf'

def _get_signature(model): 
  signatures = {
      'serving_default':
          _get_serve_image_fn(model).get_concrete_function(
              tf.TensorSpec(
                  shape=[None, 224, 224, 3],
                  dtype=tf.float32,
                  name=_transformed_name(_IMAGE_KEY)))
  }

  return signatures  

def _get_serve_image_fn(model):
  @tf.function
  def serve_image_fn(image_tensor):
    return model(image_tensor)

  return serve_image_fn


def _image_augmentation(image_features):
  batch_size = tf.shape(image_features)[0]
  image_features = tf.image.random_flip_left_right(image_features)
  image_features = tf.image.resize_with_crop_or_pad(image_features, 250, 250)
  image_features = tf.image.random_crop(image_features,
                                        (batch_size, 224, 224, 3))
  return image_features


def _data_augmentation(feature_dict):
  image_features = feature_dict[_transformed_name(_IMAGE_KEY)]
  image_features = _image_augmentation(image_features)
  feature_dict[_transformed_name(_IMAGE_KEY)] = image_features
  return feature_dict

def _input_fn(file_pattern: List[str],
              data_accessor: DataAccessor,
              tf_transform_output: tft.TFTransformOutput,
              is_train: bool = False,
              batch_size: int = 200) -> tf.data.Dataset:
  dataset = data_accessor.tf_dataset_factory(
      file_pattern,
      dataset_options.TensorFlowDatasetOptions(
          batch_size=batch_size, 
          label_key=_transformed_name(_LABEL_KEY)
      ),
      tf_transform_output.transformed_metadata.schema)

  if is_train:
    dataset = dataset.map(lambda x, y: (_data_augmentation(x), y))

  return dataset

def _freeze_model_by_percentage(model: tf.keras.Model, percentage: float):
  if percentage < 0 or percentage > 1:
    raise ValueError('Freeze percentage should between 0.0 and 1.0')

  if not model.trainable:
    raise ValueError(
        'The model is not trainable, please set model.trainable to True')

  num_layers = len(model.layers)
  num_layers_to_freeze = int(num_layers * percentage)

  for idx, layer in enumerate(model.layers):
    if idx < num_layers_to_freeze:
      layer.trainable = False
    else:
      layer.trainable = True

def _build_keras_model() -> tf.keras.Model:
  base_model = tf.keras.applications.ResNet50(
      input_shape=(224, 224, 3),
      include_top=False,
      weights='imagenet',
      pooling='max')
  base_model.input_spec = None

  model = tf.keras.Sequential([
      tf.keras.layers.InputLayer(
          input_shape=(224, 224, 3), name=_transformed_name(_IMAGE_KEY)),
      base_model,
      tf.keras.layers.Dropout(0.1),
      tf.keras.layers.Dense(10, activation='softmax')
  ])

  return model, base_model

def _compile(model_to_fit: tf.keras.Model, 
             model_to_freeze: tf.keras.Model, 
             freeze_percentage: float,
             learning_rate: float):
  _freeze_model_by_percentage(model_to_freeze, freeze_percentage)

  model_to_fit.compile(
      loss='sparse_categorical_crossentropy',
      optimizer=Adam(lr=learning_rate),
      metrics=['sparse_categorical_accuracy'])
  model_to_fit.summary(print_fn=INFO)  

  return model_to_fit, model_to_freeze

def run_fn(fn_args: FnArgs):
  steps_per_epoch = int(_TRAIN_DATA_SIZE / _TRAIN_BATCH_SIZE)
  total_epochs = int(fn_args.train_steps / steps_per_epoch)
  if _CLASSIFIER_EPOCHS > total_epochs:
    raise ValueError('Classifier epochs is greater than the total epochs')

  tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

  train_dataset = _input_fn(
      fn_args.train_files,
      fn_args.data_accessor,
      tf_transform_output,
      is_train=True,
      batch_size=_TRAIN_BATCH_SIZE)

  eval_dataset = _input_fn(
      fn_args.eval_files,
      fn_args.data_accessor,
      tf_transform_output,
      is_train=False,
      batch_size=_EVAL_BATCH_SIZE)

  INFO('Tensorboard logging to {}'.format(fn_args.model_run_dir))
  tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=fn_args.model_run_dir, update_freq='batch')

  model, base_model = _build_keras_model()
  model, base_model = _compile(model, base_model, 1.0, 
                               _CLASSIFIER_LEARNING_RATE)

  INFO('Start training the top classifier')
  model.fit(
      train_dataset,
      epochs=_CLASSIFIER_EPOCHS,
      steps_per_epoch=steps_per_epoch,
      validation_data=eval_dataset,
      validation_steps=fn_args.eval_steps,
      callbacks=[tensorboard_callback])

  INFO('Start fine-tuning the model')
  model, base_model = _compile(model, base_model, 0.9, 
                               _FINETUNE_LEARNING_RATE)

  model.fit(
      train_dataset,
      initial_epoch=_CLASSIFIER_EPOCHS,
      epochs=total_epochs,
      steps_per_epoch=steps_per_epoch,
      validation_data=eval_dataset,
      validation_steps=fn_args.eval_steps,
      callbacks=[tensorboard_callback])

  model.save(fn_args.serving_model_dir, 
             save_format='tf', 
             signatures=_get_signature(model))