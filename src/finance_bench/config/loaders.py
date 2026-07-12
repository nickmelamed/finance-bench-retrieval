from pathlib import Path
from typing import Type, TypeVar

import yaml
from pydantic import BaseModel


ROOT = Path(__file__).resolve().parents[3]

CONFIG_DIR = ROOT / "configs"

T = TypeVar("T", bound=BaseModel)


def resolve_config_path(
    relative_path: str,
) -> Path:

    path = CONFIG_DIR / relative_path

    if not path.exists():
        raise FileNotFoundError(
            f"Missing config file: {path}"
        )

    return path


def load_yaml(
    relative_path: str,
) -> dict:

    path = resolve_config_path(relative_path)

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_yaml_config(
    relative_path: str,
    schema: Type[T],
) -> T:

    raw = load_yaml(relative_path)

    return schema.model_validate(raw)