import tensorflow as tf
from tensorflow.keras.applications import resnet50
from models import features

def preprocessing_fn(inputs):
  """tf.transform's callback function for preprocessing inputs.
  Args:
    inputs: map from feature keys to raw not-yet-transformed features.
  Returns:
    Map from string feature key to transformed feature operations.
  """
  outputs = {}

  image_features = tf.map_fn(
      lambda x: tf.io.decode_png(x[0], channels=3),
      inputs[features.IMAGE_KEY],
      fn_output_signature=(tf.uint8))
  image_features = tf.image.resize(image_features, [224, 224])  
  image_features = resnet50.preprocess_input(image_features)

  outputs[features.transformed_name(features.IMAGE_KEY)] = image_features
  outputs[features.transformed_name(features.LABEL_KEY)] = inputs[features.LABEL_KEY]

  return outputs
