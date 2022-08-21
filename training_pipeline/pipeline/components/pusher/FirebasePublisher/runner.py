from typing import Any, Dict, Tuple
from absl import logging

import tensorflow as tf

import firebase_admin
from firebase_admin import ml
from firebase_admin import storage
from firebase_admin import credentials

from pipeline.components.pusher.FirebasePublisher import constants


def deploy_model_for_firebase_ml(
    model_version_name: str,
    model_path: str,
    firebase_ml_args: Dict[str, Any],
):
    model_name = firebase_ml_args[constants.FIREBASE_ML_MODEL_NAME_KEY]
    # firebase_gcs_bucket = firebase_ml_args[constants.FIREBASE_GCS_BUCKET_KEY]
    tags = firebase_ml_args[constants.FIREBASE_ML_MODEL_TAGS_KEY]
    tags.append(model_version_name)

    is_tfile, model_path = _download_pushed_model(model_path, "temp_model")

    firebase_admin.initialize_app()
    logging.info("firebase_admin initialize app is completed")
    # firebase_admin.initialize_app(
    #     credentials.Certificate("credential.json"),
    #     options={"storageBucket": firebase_dest_gcs_bucket},
    # )
    # logging.info("firebase_admin initialize app is completed")

    if is_tfile:
        source = ml.TFLiteGCSModelSource.from_tflite_model_file(model_path)
    else:
        source = ml.TFLiteGCSModelSource.from_saved_model(model_path)

    model_list = ml.list_models(list_filter=f"display_name={model_name}")
    # update
    if len(model_list.models) > 0:
        # get the first match model
        model = model_list.models[0]
        model.tags = tags
        model.model_format = ml.TFLiteFormat(model_source=source)

        updated_model = ml.update_model(model)
        ml.publish_model(updated_model.model_id)

        logging.info("model exists, so update it in FireBase ML")
    # create
    else:
        # create the model object
        tflite_format = ml.TFLiteFormat(model_source=source)
        model = ml.Model(
            display_name=model_name,
            tags=tags,
            model_format=tflite_format,
        )

        # Add the model to your Firebase project and publish it
        new_model = ml.create_model(model)
        ml.publish_model(new_model.model_id)

        logging.info("model doesn exists, so create one in FireBase ML")
    return "firebase ml published"


def _download_pushed_model(model_path: str, destination_path: str) -> Tuple[bool, str]:
    is_tfile = False

    logging.warning("download pushed model")
    root_dir = destination_path  # "saved_model"

    if not tf.io.gfile.exists(root_dir):
        tf.io.gfile.mkdir(root_dir)

    blobnames = tf.io.gfile.listdir(model_path)

    for blobname in blobnames:
        blob = f"{model_path}/{blobname}"

        if tf.io.gfile.isdir(blob):
            sub_dir = f"{root_dir}/{blobname}"
            tf.io.gfile.mkdir(sub_dir)

            sub_blobnames = tf.io.gfile.listdir(blob)
            for sub_blobname in sub_blobnames:
                sub_blob = f"{blob}{sub_blobname}"

                logging.warning(f"{sub_dir}/{sub_blobname}")
                tf.io.gfile.copy(sub_blob, f"{sub_dir}{sub_blobname}")
        else:
            logging.warning(f"{root_dir}/{blobname}")
            tf.io.gfile.copy(blob, f"{root_dir}/{blobname}")

            if "tflite" in blobname:
                is_tfile = True
                model_path = f"{root_dir}/{blobname}"

    return (is_tfile, model_path)
