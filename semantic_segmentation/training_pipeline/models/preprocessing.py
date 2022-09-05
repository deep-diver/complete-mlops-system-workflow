import tensorflow as tf
from tensorflow.keras.applications import mobilenet

_IMAGE_KEY = "image"
_IMAGE_SHAPE_KEY = "image_shape"
_LABEL_KEY = "label"
_LABEL_SHAPE_KEY = "label_shape"


def _transformed_name(key: str) -> str:
    return key + "_xf"


# output should have the same keys as inputs
def preprocess(inputs):
    images = tf.reshape(inputs[_IMAGE_KEY], [1080, 1920, 3])
    labels = tf.reshape(inputs[_LABEL_KEY], [1080, 1920, 1])

    return {
        _IMAGE_KEY: images,
        _IMAGE_SHAPE_KEY: inputs[_IMAGE_SHAPE_KEY],
        _LABEL_KEY: labels,
        _LABEL_SHAPE_KEY: inputs[_LABEL_SHAPE_KEY],
    }


def preprocessing_fn(inputs):
    """tf.transform's callback function for preprocessing inputs.
    Args:
      inputs: map from feature keys to raw not-yet-transformed features.
    Returns:
      Map from string feature key to transformed feature operations.
    """
    # print(inputs)
    outputs = {}

    features = tf.map_fn(preprocess, inputs)

    features[_IMAGE_KEY] = tf.image.resize(features[_IMAGE_KEY], [128, 128])
    features[_LABEL_KEY] = tf.image.resize(features[_LABEL_KEY], [128, 128])

    image_features = mobilenet.preprocess_input(features[_IMAGE_KEY])

    outputs[_transformed_name(_IMAGE_KEY)] = image_features
    outputs[_transformed_name(_LABEL_KEY)] = features[_LABEL_KEY]

    return outputs
