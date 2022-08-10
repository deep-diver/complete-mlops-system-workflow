import time
import copy
import os
from typing import Any, Dict
from unittest import mock

import tensorflow as tf
from tfx.dsl.io import fileio
from tfx.extensions.google_cloud_ai_platform import constants
from tfx.extensions.google_cloud_ai_platform.pusher import executor
from tfx.types import standard_artifacts
from tfx.types import standard_component_specs
from tfx.utils import json_utils
from tfx.utils import name_utils
from tfx.utils import telemetry_utils


class ExecutorTest(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        self._source_data_dir = os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            ),
            "components",
            "testdata",
        )
        self._output_data_dir = os.path.join(
            os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", self.get_temp_dir()),
            self._testMethodName,
        )
        fileio.makedirs(self._output_data_dir)
        self._model_export = standard_artifacts.Model()
        self._model_export.uri = os.path.join(self._source_data_dir, "trainer/current")
        self._model_blessing = standard_artifacts.ModelBlessing()
        self._input_dict = {
            standard_component_specs.MODEL_KEY: [self._model_export],
            standard_component_specs.MODEL_BLESSING_KEY: [self._model_blessing],
        }

        self._model_push = standard_artifacts.PushedModel()
        self._model_push.uri = os.path.join(self._output_data_dir, "model_push")
        fileio.makedirs(self._model_push.uri)
        self._output_dict = {
            standard_component_specs.PUSHED_MODEL_KEY: [self._model_push],
        }
        # Dict format of exec_properties. custom_config needs to be serialized
        # before being passed into Do function.
        self._exec_properties = {
            "custom_config": {
                "USERNAME": "deep-diver",
                "REPONAME": "PyGithubTest",
                "ASSETNAME": "saved_model.tar.gz",
                "TAG": f"v{int(time.time())}",
            },
            "push_destination": None,
        }
        self._executor = executor.Executor()

    def _serialize_custom_config_under_test(self) -> Dict[str, Any]:
        """Converts self._exec_properties['custom_config'] to string."""
        result = copy.deepcopy(self._exec_properties)
        result["custom_config"] = json_utils.dumps(result["custom_config"])
        return result

    def assertDirectoryEmpty(self, path):
        self.assertEqual(len(fileio.listdir(path)), 0)

    def assertDirectoryNotEmpty(self, path):
        self.assertGreater(len(fileio.listdir(path)), 0)

    def assertPushed(self):
        self.assertDirectoryNotEmpty(self._model_push.uri)
        self.assertEqual(1, self._model_push.get_int_custom_property("pushed"))

    def assertNotPushed(self):
        self.assertDirectoryEmpty(self._model_push.uri)
        self.assertEqual(0, self._model_push.get_int_custom_property("pushed"))
