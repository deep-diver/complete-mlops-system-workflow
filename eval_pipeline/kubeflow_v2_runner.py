# Copyright 2020 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Define KubeflowV2DagRunner to run the pipeline."""

import os
from absl import logging

from pipeline import configs
from pipeline import pipeline
from tfx.orchestration.kubeflow.v2 import kubeflow_v2_dag_runner as runner

_OUTPUT_DIR = os.path.join("gs://", configs.GCS_BUCKET_NAME)
_PIPELINE_ROOT = os.path.join(_OUTPUT_DIR, "tfx_pipeline_output", configs.PIPELINE_NAME)

# _DATA_PATH = 'gs://{}/tfx-template/data/taxi/'.format(configs.GCS_BUCKET_NAME)


def run():
    runner_config = runner.KubeflowV2DagRunnerConfig(
        default_image=configs.PIPELINE_IMAGE
    )

    runner.KubeflowV2DagRunner(
        config=runner_config,
        output_filename=configs.PIPELINE_NAME + "_pipeline.json",
    ).run(
        pipeline.create_pipeline(
            pipeline_name=configs.PIPELINE_NAME,
            pipeline_root=_PIPELINE_ROOT,
        )
    )


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    run()
