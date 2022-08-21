from typing import Any, Dict

import os
import tarfile
from absl import logging

from github import Github
import tensorflow as tf

from pipeline.components.pusher.FirebasePublisher import constants


def deploy_model_for_firebase_ml(
    model_version_name: str,
    serving_path: str,
    firebase_ml_args: Dict[str, Any],
):
    gcs_bucket = firebase_ml_args[constants.FIREBASE_GCS_BUCKET_KEY]

    # model_uri = f"{pushed_model.uri}/model.tflite"

    # assert model_uri.split("://")[0] == "gs"
    # assert credential_uri.split("://")[0] == "gs"

    # # create gcs client instance
    # gcs_client = gcs_storage.Client()

    # # get credential for firebase
    # credential_gcs_bucket = credential_uri.split("//")[1].split("/")[0]
    # credential_blob_path = "/".join(credential_uri.split("//")[1].split("/")[1:])

    # bucket = gcs_client.bucket(credential_gcs_bucket)
    # blob = bucket.blob(credential_blob_path)
    # blob.download_to_filename("credential.json")
    # logging.info(f"download credential.json from {credential_uri} is completed")

    # # get tflite model file
    # tflite_gcs_bucket = model_uri.split("//")[1].split("/")[0]
    # tflite_blob_path = "/".join(model_uri.split("//")[1].split("/")[1:])

    # bucket = gcs_client.bucket(tflite_gcs_bucket)
    # blob = bucket.blob(tflite_blob_path)
    # blob.download_to_filename("model.tflite")
    # logging.info(f"download model.tflite from {model_uri} is completed")

    # firebase_admin.initialize_app(
    #     credentials.Certificate("credential.json"),
    #     options={"storageBucket": firebase_dest_gcs_bucket},
    # )
    # logging.info("firebase_admin initialize app is completed")

    # model_list = ml.list_models(list_filter=f"display_name={model_display_name}")
    # # update
    # if len(model_list.models) > 0:
    #     # get the first match model
    #     model = model_list.models[0]
    #     source = ml.TFLiteGCSModelSource.from_tflite_model_file("model.tflite")
    #     model.model_format = ml.TFLiteFormat(model_source=source)

    #     updated_model = ml.update_model(model)
    #     ml.publish_model(updated_model.model_id)

    #     logging.info("model exists, so update it in FireBase ML")
    #     return {"result": "model updated"}
    # # create
    # else:
    #     # load a tflite file and upload it to Cloud Storage
    #     source = ml.TFLiteGCSModelSource.from_tflite_model_file("model.tflite")

    #     # create the model object
    #     tflite_format = ml.TFLiteFormat(model_source=source)
    #     model = ml.Model(
    #         display_name=model_display_name,
    #         tags=[model_tag],
    #         model_format=tflite_format,
    #     )

    #     # Add the model to your Firebase project and publish it
    #     new_model = ml.create_model(model)
    #     ml.publish_model(new_model.model_id)

    #     logging.info("model doesn exists, so create one in FireBase ML")
    #     return {"result": "model created"}
    return ""
