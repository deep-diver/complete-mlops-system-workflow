from typing import Any, Dict

import tarfile
from github import Github

from pipeline.components.pusher.GHReleasePusher import constants


def release_model_for_github(
    model_path: str,
    model_version_name: str,
    gh_release_args: Dict[str, Any],
) -> str:
    access_token = gh_release_args[constants.ACCESS_TOKEN_KEY]

    username = gh_release_args[constants.USERNAME_KEY]
    reponame = gh_release_args[constants.REPONAME_KEY]
    repo_uri = f"{username}/{reponame}"

    branch_name = gh_release_args[constants.BRANCH_KEY]

    model_archive = gh_release_args[constants.ASSETNAME_KEY]

    gh = Github(access_token)
    repo = gh.get_repo(repo_uri)
    branch = repo.get_branch(branch_name)

    release = repo.create_git_release(
        model_version_name,
        f"model release {model_version_name}",
        "",
        draft=False,
        prerelease=False,
        target_commitish=branch,
    )

    with tarfile.open(model_archive, "w:gz") as tar:
        tar.add(model_path)

    release.upload_asset(model_archive, name=model_archive)
    return f"https://github.com/{username}/{reponame}/releases/tag/{model_version_name}"