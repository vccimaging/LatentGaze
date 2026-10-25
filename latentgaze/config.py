"""YAML configs with dotted command-line overrides (``train.lr=1e-4``; ``+key=value`` adds a key)."""

import argparse
import copy
import difflib
import os
import re
from typing import Any, Dict, List

import yaml


class Config(dict):
    """Dict with attribute access, recursively."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key, value):
        self[key] = value

    @staticmethod
    def wrap(obj):
        if isinstance(obj, dict):
            return Config({k: Config.wrap(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [Config.wrap(v) for v in obj]
        return obj

    def to_dict(self) -> Dict[str, Any]:
        def unwrap(o):
            if isinstance(o, dict):
                return {k: unwrap(v) for k, v in o.items()}
            if isinstance(o, list):
                return [unwrap(v) for v in o]
            return o
        return unwrap(self)


def _merge(base: dict, other: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in other.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str, overrides: List[str] = ()) -> Config:
    """Load a YAML file (optionally with a ``_base_`` key) and apply ``a.b=value`` overrides."""
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("_base_", None)
    if base is not None:
        base_cfg = load_config(os.path.join(os.path.dirname(path), base)).to_dict()
        cfg = _merge(base_cfg, cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override '{item}' must look like key.sub=value")
        key, value = item.split("=", 1)
        add = key.startswith("+")
        parts = key.lstrip("+").split(".")
        node = cfg
        for depth, p in enumerate(parts):
            last = depth == len(parts) - 1
            if p not in node and not add:
                close = difflib.get_close_matches(p, list(node), n=1)
                hint = f"; did you mean '{close[0]}'?" if close else ""
                raise KeyError(f"Override '{key}': unknown key '{p}'{hint} (prefix with + to add a new key)")
            if not last:
                node = node.setdefault(p, {})
        node[parts[-1]] = parse_value(value)
    return Config.wrap(cfg)


_SCIENTIFIC = re.compile(r"[+-]?(\d+\.?\d*|\.\d+)[eE][+-]?\d+")


def parse_value(text: str):
    """YAML value of an override; also reads 1e-4 as a number (YAML 1.1 needs 1.0e-4)."""
    value = yaml.safe_load(text)
    if isinstance(value, str) and _SCIENTIFIC.fullmatch(value.strip()):
        return float(value)
    return value


def parse_args(description: str) -> Config:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="YAML config file")
    parser.add_argument("overrides", nargs="*",
                        help="dotted overrides of existing keys, e.g. train.lr=1e-4; +key=value adds a key")
    args = parser.parse_args()
    return load_config(args.config, args.overrides)


def save_config(cfg: Config, path: str) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)
