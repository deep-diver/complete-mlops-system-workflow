import os  # pylint: disable=unused-import

PIPELINE_NAME = "eval_pipeline"

try:
    import google.auth  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error

    try:
        _, GOOGLE_CLOUD_PROJECT = google.auth.default()
    except google.auth.exceptions.DefaultCredentialsError:
        GOOGLE_CLOUD_PROJECT = ""
except ImportError:
    GOOGLE_CLOUD_PROJECT = ""

GCS_BUCKET_NAME = GOOGLE_CLOUD_PROJECT + "-complete-mlops-eval-pipeline"
PIPELINE_IMAGE = f"gcr.io/{GOOGLE_CLOUD_PROJECT}/{PIPELINE_NAME}"

EVAL_ACCURACY_THRESHOLD = 0.6
