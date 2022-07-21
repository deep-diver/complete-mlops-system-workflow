import os
from absl import logging

from tfx import v1 as tfx
from tfx import proto
from tfx.orchestration.kubeflow.v2 import kubeflow_v2_dag_runner as runner
from tfx.orchestration.data_types import RuntimeParameter
from tfx.proto import pusher_pb2
from tfx.proto import trainer_pb2

from training_pipeline import configs
from training_pipeline import pipeline

OUTPUT_DIR = os.path.join('gs://', configs.GCS_BUCKET_NAME)
PIPELINE_ROOT = os.path.join(OUTPUT_DIR, 'tfx_pipeline_output', configs.PIPELINE_NAME)
SERVING_MODEL_DIR = os.path.join(PIPELINE_ROOT, 'serving_model')
DATA_PATH = 'gs://{}/data/'.format(configs.GCS_BUCKET_NAME)

def run():
  runner_config = runner.KubeflowV2DagRunnerConfig(
      default_image=configs.PIPELINE_IMAGE)

  runner.KubeflowV2DagRunner(
      config=runner_config,
      output_filename=configs.PIPELINE_NAME+"_pipeline.json",
  ).run(
      pipeline.create_pipeline(
          """
          RuntimeParameter could be injected with TFX CLI
          : 
          --runtime-parameter output-config='{}' \
          --runtime-parameter input-config='{"splits": [{"name": "train", "pattern": "span-[12]/train/*.tfrecord"}, {"name": "val", "pattern": "span-[12]/test/*.tfrecord"}]}' 
            
          OR it could be injected programatically
          : 
            import json
            from kfp.v2.google import client

            pipelines_client = client.AIPlatformClient(
                project_id=GOOGLE_CLOUD_PROJECT, region=GOOGLE_CLOUD_REGION,
            )
            _ = pipelines_client.create_run_from_job_spec(
                PIPELINE_DEFINITION_FILE,
                enable_caching=False,
                parameter_values={
                    "input-config": json.dumps(
                        {
                            "splits": [
                                {"name": "train", "pattern": "span-[12]/train/*.tfrecord"},
                                {"name": "val", "pattern": "span-[12]/test/*.tfrecord"},
                            ]
                        }
                    ),
                    "output-config": json.dumps({}),
                },
            )          
          """
          
          input_config=RuntimeParameter(
                  name="input-config",
                  default='{"input_config": {"splits": [{"name":"train", "pattern":"span-1/train/*"}, {"name":"eval", "pattern":"span-1/test/*"}]}}',
                  ptype=str,                  
          ),
          output_config=RuntimeParameter(
                  name="output-config",
                  default="{}", 
                  ptype=str,
          ),          
          pipeline_name=configs.PIPELINE_NAME,
          pipeline_root=PIPELINE_ROOT,
          data_path=DATA_PATH,
          modules={
            "preprocessing_fn": configs.PREPROCESSING_FN,
            "training_fn": configs.TRAINING_FN,
          },
          train_args=trainer_pb2.TrainArgs(
            num_steps=configs.TRAIN_NUM_STEPS
          ),
          eval_args=trainer_pb2.EvalArgs(
            num_steps=configs.EVAL_NUM_STEPS
          ),
          serving_model_dir=SERVING_MODEL_DIR,

          ai_platform_training_args=configs.GCP_AI_PLATFORM_TRAINING_ARGS,
          ai_platform_serving_args=configs.GCP_AI_PLATFORM_SERVING_ARGS
      ))

if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  run()
