import argparse
import os
import subprocess
from glob import glob

import git
import requests
from dotenv import load_dotenv

load_dotenv()
github_token = os.environ.get("GITHUB_TOKEN")

if not github_token:
    raise ValueError("GITHUB_TOKEN is not set")

headers = {
    "Authorization": f"token {github_token}",
    "Accept": "application/vnd.github.v3+json",
}


def get_user_and_email():
    url = "https://api.github.com/user"
    res = requests.get(url, headers=headers)
    data = res.json()
    user_name = data["login"]
    email = data["email"]
    return user_name, email


def bump_version(version):
    version_path = glob("**/VERSION", recursive=True)[0]
    with open(version_path, "w") as f:
        f.write(version)

    # set current user email and name
    username, email = get_user_and_email()
    subprocess.run(["git", "config", "user.email", email])
    subprocess.run(["git", "config", "user.name", username])

    # commit the change
    subprocess.run(["git", "add", version_path])
    subprocess.run(["git", "commit", "-m", f"Bump version to {version}"])
    subprocess.run(["git", "push"])
    print(f"Bumped version to {version}")


def get_version():
    version_file = glob("**/VERSION", recursive=True)[0]
    with open(version_file, "r") as f:
        return f.read().strip()


def get_next_version(version):
    major, minor, patch = version.split(".")
    return f"{major}.{minor}.{int(patch) + 1}"


def get_diff_between_versions(old_version, new_version):
    return subprocess.run(
        ["git", "log", f"{old_version}..{new_version}", "--oneline"],
        capture_output=True,
    ).stdout.decode()


def get_current_user_name_with_github_token():
    url = "https://api.github.com/user"

    res = requests.get(url, headers=headers)
    res.raise_for_status()
    return res.json()["login"]


def get_current_repo_name():
    current_path = os.getcwd()

    repo = git.Repo(current_path)
    return repo.remotes.origin.url.split("/")[-1].replace(".git", "")


def create_release(next_version=None):
    from_version = get_version()
    next_version = next_version or get_next_version(from_version)
    current_user_name = get_current_user_name_with_github_token()
    current_repo = get_current_repo_name()
    bump_version(next_version)

    # create tag
    tag_url = (
        f"https://api.github.com/repos/{current_user_name}/{current_repo}/git/refs"
    )
    current_commit_id = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True
    ).stdout.decode()
    tag_data = {"ref": f"refs/tags/{next_version}", "sha": current_commit_id.strip()}

    res = requests.post(tag_url, headers=headers, json=tag_data)
    res.raise_for_status()

    change_log = get_diff_between_versions(from_version, next_version)
    release_url = (
        f"https://api.github.com/repos/{current_user_name}/{current_repo}/releases"
    )
    release_data = {"tag_name": next_version, "name": next_version, "body": change_log}

    res = requests.post(release_url, headers=headers, json=release_data)
    res.raise_for_status()
    print(f"Release created for version {next_version}")
    print(f"Change log: {change_log}")
    print(f'Release URL: {res.json()["html_url"]}')
    print(f'Release ID: {res.json()["id"]}')
    return next_version


if __name__ == "__main__":

    # Define the parser
    parser = argparse.ArgumentParser()

    # Define an argument "-v" for version
    parser.add_argument("-v", "--version", help="version number like 1.0.0")

    # Parse the arguments
    args = parser.parse_args()
    version = args.version
    if version:
        create_release(version)
    else:
        current_version = get_version()
        next_version = get_next_version(current_version)
        input(
            f"Version is not provided. Press Enter to create a release for '{current_version} --> {next_version}'"
        )
        create_release()
