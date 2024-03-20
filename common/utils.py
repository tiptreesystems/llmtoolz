from datetime import datetime
import json
import os
from typing import Optional, Any
import yaml

import git
from common.logger import logger
from pydantic import BaseModel
from enum import Enum

from common.constants import PRICE_PER_KILOTOKEN, VENDOR_QUERY_PRICE


class ResearchType(Enum):
    PUBLIC_RESEARCH = "public_research"
    PROPRIETARY_DATA_ACCESS = "proprietary_data_access"
    EXPERT_RESEARCH = "expert_research"


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


class AltheaConfig:
    def __init__(self, config: dict):
        self.config = config

    def get_attribute_for_service(
        self, *, name: str, attribute: str, service_type: Optional[str] = None
    ) -> Any:
        service_type = "comm" if service_type is None else service_type
        return self.config[f"{service_type}_services"][name][attribute]

    def get_host_for_service(
        self, name: str, service_type: Optional[str] = None
    ) -> str:
        return self.get_attribute_for_service(
            name=name, attribute="host", service_type=service_type
        )

    def get_port_for_service(
        self, name: str, service_type: Optional[str] = None
    ) -> int:
        return self.get_attribute_for_service(
            name=name, attribute="port", service_type=service_type
        )

    def get_endpoint_for_service(
        self, name: str, service_type: Optional[str] = None, prefix: str = "http://"
    ) -> str:
        host = self.get_host_for_service(name, service_type)
        port = self.get_port_for_service(name, service_type)
        return f"{prefix}{host}:{port}"

    def get_token_for_service(
        self, name: str, service_type: Optional[str] = None
    ) -> str:
        return self.get_attribute_for_service(
            name=name, attribute="token", service_type=service_type
        )

    def get_beat_interval(self, default: int = 60, cycle_name: str = "short") -> int:
        cycle_spec = self.config["beats"].get(f"{cycle_name}_cycle", {})
        return cycle_spec.get("polling_interval", default)

    def get_beat_thread_filter_config(self) -> dict:
        return self.config["beats"].get("thread_filter_config", {})

    @property
    def app_config(self):
        return self.config["app_config"]

    def get_app_config(
        self, key: str, default: Any = None, raise_when_not_found: bool = True
    ):
        if key not in self.config["app_config"] and raise_when_not_found:
            raise KeyError(f"Key {key} not found in {self.config['app_config'].keys()}")
        return self.config["app_config"].get(key, default)

    @classmethod
    def from_path(cls, path: str) -> "AltheaConfig":
        with open(get_path(path, makedirs=False), "r") as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        return cls(config)

    @classmethod
    def from_env(cls, variable="ALTHEA_CONFIG_PATH") -> "AltheaConfig":
        path = os.getenv(variable)
        if path is None:
            raise ValueError(f"Environment variable {variable} not set.")
        return cls.from_path(path)


class UsageRecord(BaseModel):
    id: Optional[int] = None
    answer_id: str
    name: str
    user_id: str
    vendor_id: Optional[str]
    research_type: str
    usage_description: str
    usage_amount: float
    log_path: str
    processed: bool
    time: float

    def __post_init__(self):
        assert ResearchType(self.research_type) in ResearchType
        assert self.usage_amount >= 0
        assert self.time >= 0

    @property
    def timestamp(self) -> datetime:
        return datetime.fromtimestamp(self.time)

    @property
    def price(self) -> float:
        return self.usage_amount * self.get_price_per_unit()

    def get_price_per_unit(self) -> float:
        if self.research_type == ResearchType.PUBLIC_RESEARCH.value:
            return PRICE_PER_KILOTOKEN / 1000
        elif self.research_type == ResearchType.PROPRIETARY_DATA_ACCESS.value:
            return VENDOR_QUERY_PRICE[self.name]
        elif self.research_type == ResearchType.EXPERT_RESEARCH.value:
            return VENDOR_QUERY_PRICE[self.vendor_id]
        else:
            raise ValueError(f"Unknown usage type {self.usage_type}")

    @classmethod
    def from_model_dump(cls, model_dump: dict) -> "UsageRecord":
        if isinstance(model_dump["research_type"], ResearchType):
            model_dump["research_type"] = model_dump["research_type"].value
        return cls(**model_dump)
