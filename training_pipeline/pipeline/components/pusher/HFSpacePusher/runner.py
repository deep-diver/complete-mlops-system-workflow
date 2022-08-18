from typing import Any, Dict

import os
import tarfile
import time
from os import listdir
from absl import logging

import tensorflow as tf

from huggingface_hub import HfApi
from requests.exceptions import HTTPError

from pipeline.components.pusher.HFModelPusher import constants


def release_model_for_hf_space(
    model_repo_id: str,
    model_version_name: str,
    hf_release_args: Dict[str, Any],
) -> str:
    access_token = hf_release_args[constants.ACCESS_TOKEN_KEY]

    username = hf_release_args[constants.USERNAME_KEY]
    reponame = hf_release_args[constants.REPONAME_KEY]
    repo_id = f"{username}/{reponame}"

    app_path = hf_release_args[constants.APP_PATH]

    repo_type = "space"

    hf_api = HfApi()
    hf_api.set_access_token(access_token)

    logging.warning(listdir("."))

    # hf_hub_path = ""
    # try:
    #     hf_api.create_repo(
    #         token=access_token, repo_id=f"{repo_id}-space", repo_type=repo_type
    #     )
    # except HTTPError:
    #     logging.warning(f"{repo_id}-model repository may already exist")
    #     pass

    # try:
    #     hf_hub_path = hf_api.upload_folder(
    #         repo_id=f"{repo_id}-model",
    #         folder_path=app_path,
    #         path_in_repo=".",
    #         token=access_token,
    #         create_pr=True,
    #         repo_type=repo_type,
    #         commit_message=model_version_name,
    #     )
    #     logging.warning(f"file is uploaded at {repo_id}-space")
    # except HTTPError as error:
    #     logging.warning(error)

    # return (f"{repo_id}-space", hf_hub_path)
    return ("dummy", "dummy")
