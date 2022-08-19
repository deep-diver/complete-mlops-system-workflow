from typing import Any, Dict

import os
import tarfile
import time
from absl import logging

import tensorflow as tf

from huggingface_hub import Repository
from huggingface_hub import HfApi
from requests.exceptions import HTTPError

from pipeline.components.pusher.HFModelPusher import constants


def release_model_for_hf_model(
    model_path: str,
    model_version_name: str,
    hf_release_args: Dict[str, Any],
) -> str:
    access_token = hf_release_args[constants.ACCESS_TOKEN_KEY]

    username = hf_release_args[constants.USERNAME_KEY]
    reponame = hf_release_args[constants.REPONAME_KEY]

    repo_type = "model"

    repo_id = f"{username}/{reponame}-{repo_type}"
    repo_url_prefix = "https://huggingface.co"
    repo_url = f"{repo_url_prefix}/{repo_id}"

    logging.warning(f"model_path: {model_path}")

    logging.warning("download pushed model")
    root_dir = model_version_name
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

    try:
        HfApi().create_repo(
            token=access_token, repo_id=f"{repo_id}-model", repo_type=repo_type
        )
    except HTTPError:
        logging.warning(f"{repo_id}-model repository may already exist")
        pass

    repository = Repository(
        local_dir=repo_id, clone_from=repo_url, use_auth_token=access_token
    )

    repository.git_checkout(revision={model_version_name}, create_branch_ok=True)

    dummy_filepath = f"{username}/{reponame}/{model_version_name}"
    open(dummy_filepath, mode="w", encoding="utf-8").close()

    repository.git_add(pattern=".")
    repository.git_commit(commit_message="add dummy data to create branch")
    repository.git_push(upstream=f"origin {model_version_name}")

    try:
        HfApi().upload_folder(
            repo_id=f"{repo_id}",
            folder_path=model_path,
            path_in_repo=".",
            token=access_token,
            create_pr=True,
            repo_type=repo_type,
            commit_message=model_version_name,
            revision=model_version_name,
        )
        logging.warning(f"file is uploaded at {repo_id}")
    except HTTPError as error:
        logging.warning(error)

    return (f"{repo_id}", repo_url)
