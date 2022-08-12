import os  # pylint: disable=unused-import
import tfx
import tfx.extensions.google_cloud_ai_platform.constants as vertex_const
import tfx.extensions.google_cloud_ai_platform.trainer.executor as vertex_training_const
import tfx.extensions.google_cloud_ai_platform.tuner.executor as vertex_tuner_const

PIPELINE_NAME = "resnet50-tfx-pipeline4"

try:
    import google.auth  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error

    try:
        _, GOOGLE_CLOUD_PROJECT = google.auth.default()
    except google.auth.exceptions.DefaultCredentialsError:
        GOOGLE_CLOUD_PROJECT = "gcp-ml-172005"
except ImportError:
    GOOGLE_CLOUD_PROJECT = "gcp-ml-172005"

GOOGLE_CLOUD_REGION = "us-central1"

GCS_BUCKET_NAME = GOOGLE_CLOUD_PROJECT + "-complete-mlops"
PIPELINE_IMAGE = f"gcr.io/{GOOGLE_CLOUD_PROJECT}/{PIPELINE_NAME}"

OUTPUT_DIR = os.path.join("gs://", GCS_BUCKET_NAME)
PIPELINE_ROOT = os.path.join(OUTPUT_DIR, "tfx_pipeline_output", PIPELINE_NAME)
DATA_PATH = f"gs://{GCS_BUCKET_NAME}/data/"


PREPROCESSING_FN = "models.preprocessing.preprocessing_fn"
TRAINING_FN = "models.model.run_fn"
TUNER_FN = "models.model.tuner_fn"
CLOUD_TUNER_FN = "models.model.cloud_tuner_fn"

TRAIN_NUM_STEPS = 160
EVAL_NUM_STEPS = 4
NUM_PARALLEL_TRIALS = 3

EVAL_ACCURACY_THRESHOLD = 0.6

GCP_AI_PLATFORM_TRAINING_ARGS = {
    vertex_const.ENABLE_VERTEX_KEY: True,
    vertex_const.VERTEX_REGION_KEY: GOOGLE_CLOUD_REGION,
    vertex_training_const.TRAINING_ARGS_KEY: {
        "project": GOOGLE_CLOUD_PROJECT,
        "worker_pool_specs": [
            {
                "machine_spec": {
                    "machine_type": "n1-standard-4",
                    "accelerator_type": "NVIDIA_TESLA_K80",
                    "accelerator_count": 1,
                },
                "replica_count": 1,
                "container_spec": {
                    "image_uri": PIPELINE_IMAGE,
                },
            }
        ],
    },
    "use_gpu": True,
}

GCP_AI_PLATFORM_TUNER_ARGS = {
    vertex_tuner_const.TUNING_ARGS_KEY: {
        "project": GOOGLE_CLOUD_PROJECT,
        "region": "us-central1",
        "scaleTier": "STANDARD_1",
        "masterType": "n1-standard-4",
        "masterConfig": {
            "imageUri": PIPELINE_IMAGE,
            "acceleratorConfig": {
                "count": 1,
                "type": "NVIDIA_TESLA_K80",
            },
        },
        # 'serviceAccount': 'vizier@gcp-ml-172005.iam.gserviceaccount.com',
    },
    vertex_tuner_const.REMOTE_TRIALS_WORKING_DIR_KEY: os.path.join(
        PIPELINE_ROOT, "trials"
    ),
}

GCP_AI_PLATFORM_SERVING_ARGS = {
    vertex_const.ENABLE_VERTEX_KEY: True,
    vertex_const.VERTEX_REGION_KEY: GOOGLE_CLOUD_REGION,
    vertex_const.VERTEX_CONTAINER_IMAGE_URI_KEY: "us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-8:latest",
    vertex_const.SERVING_ARGS_KEY: {
        "project_id": GOOGLE_CLOUD_PROJECT,
        "deployed_model_display_name": PIPELINE_NAME.replace("-", "_"),
        "endpoint_name": "prediction-" + PIPELINE_NAME.replace("-", "_"),
        "traffic_split": {"0": 100},
        "machine_type": "n1-standard-4",
        "min_replica_count": 1,
        "max_replica_count": 1,
    },
}

GH_RELEASE_ARGS = {
    "GH_RELEASE": {
        "ACCESS_TOKEN": "$GH_ACCESS_TOKEN",
        "USERNAME": "deep-diver",
        "REPONAME": "PyGithubTest",
        "BRANCH": "main",
        "ASSETNAME": "saved_model.tar.gz",
    }
}
