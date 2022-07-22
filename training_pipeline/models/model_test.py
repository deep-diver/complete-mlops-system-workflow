import tensorflow as tf

from models import model


class ModelTest(tf.test.TestCase):
    def testBuildKerasModel(self):
        built_model = model._build_keras_model(
            ["foo", "bar"]
        )  # pylint: disable=protected-access
        self.assertEqual(len(built_model.inputs), 2)


if __name__ == "__main__":
    tf.test.main()
