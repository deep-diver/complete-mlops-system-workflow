from typing import Any, Dict

import os
import tarfile
import time
from absl import logging

import tensorflow as tf

from huggingface_hub import HfApi
from requests.exceptions import HTTPError

from pipeline.components.pusher.HFModelPusher import constants


def release_model_for_hf_model(
    model_path: str,
    model_version_name: str,
    gh_release_args: Dict[str, Any],
) -> str:
    access_token = gh_release_args[constants.ACCESS_TOKEN_KEY]

    username = gh_release_args[constants.USERNAME_KEY]
    reponame = gh_release_args[constants.REPONAME_KEY]
    repo_id = f"{username}/{reponame}"

    repo_type = "model"

    hf_api = HfApi()
    hf_api.set_access_token(access_token)

    logging.warning(f"model_path: {model_path}")

    logging.warning("download pushed model")
    model_name = f"v{int(time.time())}"
    root_dir = model_name
    os.mkdir(root_dir)

    blobnames = tf.io.gfile.listdir(model_path)

    for blobname in blobnames:
        blob = f"{model_path}/{blobname}"

        if tf.io.gfile.isdir(blob):
            sub_dir = f"{root_dir}/{blobname}"
            os.mkdir(sub_dir)

            sub_blobnames = tf.io.gfile.listdir(blob)
            for sub_blobname in sub_blobnames:
                sub_blob = f"{blob}{sub_blobname}"

                logging.warning(f"{sub_dir}/{sub_blobname}")
                tf.io.gfile.copy(sub_blob, f"{sub_dir}{sub_blobname}")
        else:
            logging.warning(f"{root_dir}/{blobname}")
            tf.io.gfile.copy(blob, f"{root_dir}/{blobname}")

    model_path = root_dir

    hf_hub_path = ""
    try:
        hf_api.create_repo(
            token=access_token, repo_id=f"{repo_id}-model", repo_type=repo_type
        )
    except HTTPError as e:
        logging.warning(e)
        logging.warning(f"{repo_id}-model repository may already exist")
    finally:
        try:
            hf_hub_path = hf_api.upload_folder(
                repo_id=f"{repo_id}-model",
                folder_path=model_path,
                token=access_token,
                create_pr=True,
                repo_type=repo_type,
                commit_message=model_name,
            )
            logging.warning(f"file is uploaded at {repo_id}-model")
        except HTTPError:
            logging.warning(e)
            raise HTTPError

    return hf_hub_path
