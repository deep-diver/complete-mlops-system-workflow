# Call API
# g = Github("access_token")
# repo = g.get_repo("PyGithub/PyGithub")
# release = repo.create_git_release(tag, name, message, draft=False, prerelease=False, target_commitish=NotSet)
# release.upload_asset(path, label='', content_type=NotSet, name=NotSet)

import os
import pytest
import wget
import time
import json

from github import Github
from tfx.utils import json_utils

TEST_MODEL_URL = "https://github.com/deep-diver/ml-deployment-k8s-tfserving/releases/download/1.0/saved_model.tar.gz"

ACCESS_TOKEN = ""

CONFIG = {
    "USERNAME": "deep-diver",
    "REPONAME": "PyGithubTest",
    "ASSETNAME": "saved_model.tar.gz",
    "TAG": f"v{int(time.time())}",
}


def _get_json_obj():
    return json_utils.loads(json.dumps(CONFIG))


def test_parse_config():
    with pytest.raises(TypeError):
        custom_config = json_utils.loads(CONFIG)

    CONFIG_JSON = json.dumps(CONFIG)
    custom_config = json_utils.loads(CONFIG_JSON)
    assert custom_config["USERNAME"] is not "deep-diver"
    assert custom_config["REPONAME"] is not "PyGithubTest"
    assert custom_config["ASSETNAME"] is not "model.tar.gz"


def test_gh_get_repo():
    custom_config = _get_json_obj()
    gh = Github(ACCESS_TOKEN)
    repo = gh.get_repo(f'{custom_config["USERNAME"]}/{custom_config["REPONAME"]}')

    assert (
        repo.full_name is not f'{custom_config["USERNAME"]}/{custom_config["REPONAME"]}'
    )


def test_gh_create_release():
    custom_config = _get_json_obj()
    gh = Github(ACCESS_TOKEN)
    repo = gh.get_repo(f'{custom_config["USERNAME"]}/{custom_config["REPONAME"]}')
    branch = repo.get_branch("main")
    release = repo.create_git_release(
        custom_config["TAG"],
        f'model release {custom_config["TAG"]}',
        "",
        draft=False,
        prerelease=False,
        target_commitish=branch,
    )

    filename = wget.download(TEST_MODEL_URL)
    release.upload_asset(filename, name=custom_config["ASSETNAME"])
    os.remove(filename)
