from datetime import datetime
import json
import os
from typing import Optional, Any
import yaml

import git
from common.logger import logger
from pydantic import BaseModel
from enum import Enum

from common.constants import PRICE_PER_KILOTOKEN



def get_git_root():
    try:
        # Start the search from the directory of the current file
        repo = git.Repo(
            os.path.dirname(os.path.abspath(__file__)), search_parent_directories=True
        )
        return repo.git.rev_parse("--show-toplevel")
    except git.InvalidGitRepositoryError:
        logger.error("Error: This directory is not part of a Git repository.")
        return None


def get_path(
    relative_path: str, makedirs: bool = True, root_directory: Optional[str] = None
) -> str:
    if root_directory is None:
        # Get the root directory of the Git repository
        root_directory = get_git_root()
    if root_directory is None:
        raise ValueError("root_directory must be specified if not in a Git repository.")
    path = os.path.join(root_directory, relative_path)
    if makedirs:
        # Check if the path ends with an extension (assumed to be a file in this case)
        if os.path.splitext(relative_path)[1]:
            dir_to_make = os.path.dirname(path)
        else:
            dir_to_make = path
        os.makedirs(dir_to_make, exist_ok=True)
    return path


def get_key(name: str) -> Optional[str]:
    git_root = get_git_root()
    if git_root is None:
        return None

    # Build the full path to keys.json using the Git root directory
    keys_path = os.path.join(git_root, "creds", "keys.json")

    try:
        with open(keys_path, "r") as file:
            keys = json.load(file)
            return keys.get(name, None)
    except FileNotFoundError:
        logger.error(f"Error: {keys_path} not found.")
        return None
    except json.JSONDecodeError:
        logger.error(f"Error: {keys_path} is not in valid JSON format.")
        return None

