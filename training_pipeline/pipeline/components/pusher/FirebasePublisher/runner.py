from typing import Any, Dict, List, Tuple
from absl import logging

import os
import glob
import tempfile
import tensorflow as tf
from tfx.dsl.io import fileio
from tfx.utils import io_utils

import firebase_admin
from firebase_admin import ml
from firebase_admin import credentials
from firebase_admin.ml import TFLiteModelSource, ListModelsPage


_SIZE_LIMIT_MB = 80

def _prepare_fb_download_model(credential_path: str, storage_bucket: str,
                               model_path: str, options: Dict[str, Any]) -> str:
    tmp_dir = tempfile.gettempdir()
    credential = None

    if credential_path is not None:
        tmp_credential_path = os.path.join(tmp_dir, "credentials.json")
        io_utils.copy_file(credential_path, tmp_credential_path)
        credential = credentials.Certificate(tmp_credential_path)
        logging.info(
            "credentials are copied into a temporary directory in local filesystem"
        )

    options["storageBucket"] = storage_bucket
    
    firebase_admin.initialize_app(credential=credential, options=options)
    logging.info("firebase app initialization is completed")
    
    tmp_model_path = os.path.join(tmp_dir, "model")
    io_utils.copy_dir(model_path, tmp_model_path)
    
    return tmp_model_path

def _get_model_path_and_type(tmp_model_path) -> Tuple[bool, str]:
    tflite_files = glob.glob(f"{tmp_model_path}/**/*.tflite")
    is_tflite = len(tflite_files) > 0
    model_path = tflite_files[0] if is_tflite else tmp_model_path
    
    return is_tflite, model_path

def _upload_model(is_tflite: bool, model_path: str) -> TFLiteModelSource:
    if is_tflite:
        source = ml.TFLiteGCSModelSource.from_tflite_model_file(model_path)
    else:
        source = ml.TFLiteGCSModelSource.from_saved_model(model_path)
        
    return source

def _check_model_size(source: TFLiteModelSource):
    gcs_path_for_uploaded_file = source.as_dict().get('gcsTfliteUri')
    with tf.io.gfile.GFile(gcs_path_for_uploaded_file) as f:
        file_size_in_mb = f.size() / (1 << 20)
        
    if file_size_in_mb > _SIZE_LIMIT_MB:
        fileio.remove(gcs_path_for_uploaded_file)
        raise RuntimeError(
            f"the file size exceeds the limit of {_SIZE_LIMIT_MB}. Uploaded file is removed."
        )

def _update_model(model_list: ListModelsPage,
                  source: TFLiteModelSource,
                  tags: List[str], model_version: str):
    tags.append(model_version)
    
    # get the first match model
    model = model_list.models[0]

    model.tags = tags
    model.model_format = ml.TFLiteFormat(model_source=source)

    updated_model = ml.update_model(model)
    ml.publish_model(updated_model.model_id)

    logging.info("model exists, so it is updated")

def _create_model(display_name: str,
                  source: TFLiteModelSource, 
                  tags: List[str], model_version: str):
    tags.append(model_version)
    
    tflite_format = ml.TFLiteFormat(model_source=source)
    model = ml.Model(
        display_name=display_name,
        tags=tags,
        model_format=tflite_format,
    )

    # Add the model to your Firebase project and publish it
    new_model = ml.create_model(model)
    ml.publish_model(new_model.model_id)

    logging.info("model didn't exist, so it is created")    

def deploy_model_for_firebase_ml(
    display_name: str,
    storage_bucket: str,
    tags: List[Any],
    options: Dict[str, Any],
    model_path: str,
    model_version: str,
    credential_path: str,
) -> str:
    model_list = ml.list_models(list_filter=f"display_name={display_name}")

    tmp_model_path = _prepare_fb_download_model(credential_path, storage_bucket,
                                                model_path, options)
    
    is_tflite, model_path = _get_model_path_and_type(tmp_model_path)
    source = _upload_model(is_tflite, model_path)

    _check_model_size(source)
    
    if len(model_list.models) > 0:
        _update_model(model_list, source, tags, model_version)
    else:
        _create_model(display_name, source, tags, model_version)

    return source.as_dict().get('gcsTfliteUri')