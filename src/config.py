"""Configuration loading. Single source for paths, simulation design and privacy settings."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"


def _load(name: str) -> dict[str, Any]:
    with open(CONFIG_DIR / name, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_settings() -> dict[str, Any]:
    return _load("settings.yaml")


def load_simulation() -> dict[str, Any]:
    return _load("simulation.yaml")


def iter_zones(sim: dict[str, Any]):
    """Yield (site_id, zone_id, zone_attrs) for every configured zone."""
    for site_id, site in sim["sites"].items():
        for zone_id, attrs in site["zones"].items():
            yield site_id, zone_id, attrs


def expected_observation_count(sim: dict[str, Any] | None = None) -> int:
    """Number of date x site x zone x shift observations implied by the simulation config."""
    sim = sim or load_simulation()
    n_zones = sum(1 for _ in iter_zones(sim))
    return sim["n_days"] * n_zones * len(sim["shifts"])


def path_for(layer: str) -> Path:
    """Absolute path for a data layer: 'raw', 'processed' or 'ground_truth'."""
    return PROJECT_ROOT / load_settings()["paths"][layer]
