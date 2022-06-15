import tensorflow as tf

from models import features


class FeaturesTest(tf.test.TestCase):

  def testLabelKey(self):
    self.assertNotIn(features.LABEL_KEY, features.FEATURE_KEYS)


if __name__ == "__main__":
  tf.test.main()
