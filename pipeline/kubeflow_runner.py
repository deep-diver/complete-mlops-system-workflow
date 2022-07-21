import os
from absl import logging

from tfx import v1 as tfx
from tfx.proto import TrainArgs, EvalArgs
from tfx.orchestration.kubeflow.v2 import kubeflow_v2_dag_runner
from pipeline import configs
from pipeline import pipeline

OUTPUT_DIR = os.path.join('gs://', configs.GCS_BUCKET_NAME)
PIPELINE_ROOT = os.path.join(OUTPUT_DIR, 'tfx_pipeline_output', configs.PIPELINE_NAME)
SERVING_MODEL_DIR = os.path.join(PIPELINE_ROOT, 'serving_model')
DATA_PATH = 'gs://{}/data/'.format(configs.GCS_BUCKET_NAME)

def run():
  runner_config = kubeflow_v2_dag_runner.KubeflowV2DagRunnerConfig(
      default_image=configs.PIPELINE_IMAGE)

  kubeflow_v2_dag_runner.KubeflowV2DagRunner(
      config=runner_config
  ).run(
      pipeline.create_pipeline(
          pipeline_name=configs.PIPELINE_NAME,
          pipeline_root=PIPELINE_ROOT,
          data_path=DATA_PATH,
          preprocessing_fn=configs.PREPROCESSING_FN,
          run_fn=configs.RUN_FN,
          train_args=TrainArgs(
            num_steps=configs.TRAIN_NUM_STEPS
          ),
          train_cloud_region='us-central1',
          train_cloud_args=configs.TRAINING_JOB_SPEC,
          eval_args=EvalArgs(
            num_steps=configs.EVAL_NUM_STEPS
          ),
          eval_accuracy_threshold=configs.EVAL_ACCURACY_THRESHOLD,
          serving_model_dir=SERVING_MODEL_DIR,
      ))

if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  run()
