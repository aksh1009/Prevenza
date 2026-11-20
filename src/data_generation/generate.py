"""Synthetic data generator CLI.

    python -m src.data_generation.generate                # seed from config/simulation.yaml
    python -m src.data_generation.generate --seed 7       # override seed
    python -m src.data_generation.generate --no-write     # generate + validate only

Outputs: data/raw/*.csv (5 sources), data/ground_truth/*.csv, docs/simulation_summary.md,
docs/data_quality_report.md. Ground truth is written separately and never joined to raw sources.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src import config, schemas

from .environment import simulate_environment
from .grid import build_grid, shift_start_hours
from .health import simulate_workforce_and_health
from .observation import KEY, _key_sort, build_sources
from .scenarios import build_events, build_labels, event_windows, forcing


def generate(sim: dict | None = None, seed: int | None = None) -> dict[str, pd.DataFrame]:
    """Generate all datasets in memory. Deterministic for a given (config, seed)."""
    sim = sim or config.load_simulation()
    seed = sim["random_seed"] if seed is None else seed
    # Independent child streams per component so changing one stage doesn't reshuffle others.
    rng_env, rng_health, rng_obs = (np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3))

    grid = build_grid(sim)
    windows = event_windows(sim, shift_start_hours())
    force = forcing(grid, windows)
    env = simulate_environment(grid, sim, force, rng_env)
    health = simulate_workforce_and_health(grid, sim, env, force, rng_health)
    sources, dq_log = build_sources(grid, sim, windows, env, health, rng_obs)

    events = build_events(windows)
    labels = build_labels(grid, events).sort_values(KEY, key=_key_sort).reset_index(drop=True)
    gt = {
        "heat_risk_events": schemas.coerce(events, schemas.HEAT_RISK_EVENTS),
        "observation_labels": schemas.coerce(labels, schemas.OBSERVATION_LABELS),
        "data_quality_log": schemas.coerce(dq_log, schemas.DATA_QUALITY_LOG),
    }
    return {**sources, **gt}


def validate_all(data: dict[str, pd.DataFrame]) -> dict[str, list[str]]:
    issues = {name: schemas.validate(df, schemas.ALL_CONTRACTS[name]) for name, df in data.items()}
    return {k: v for k, v in issues.items() if v}


def write(data: dict[str, pd.DataFrame]) -> list[Path]:
    paths = []
    for name, df in data.items():
        c = schemas.ALL_CONTRACTS[name]
        p = config.path_for(c.layer) / c.file_name
        p.parent.mkdir(parents=True, exist_ok=True)
        out = df.copy()
        for f in c.fields:                       # ISO dates; NA written as empty cell (never 0)
            if f.dtype == "date":
                out[f.name] = out[f.name].dt.strftime("%Y-%m-%d")
            elif f.dtype == "datetime":
                out[f.name] = out[f.name].dt.strftime("%Y-%m-%d %H:%M")
        out.to_csv(p, index=False, na_rep="")
        paths.append(p)
    return paths


def read_dataset(name: str) -> pd.DataFrame:
    """Load a written dataset back with contract dtypes (shared loader in src/preprocessing/load.py)."""
    from src.preprocessing.load import load_dataset
    return load_dataset(name, validate=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate PREVENZA synthetic heat-risk datasets.")
    ap.add_argument("--seed", type=int, default=None, help="override random_seed in config/simulation.yaml")
    ap.add_argument("--no-write", action="store_true", help="validate only; do not write files")
    args = ap.parse_args(argv)

    sim = config.load_simulation()
    seed = sim["random_seed"] if args.seed is None else args.seed
    data = generate(sim, seed)
    problems = validate_all(data)
    if problems:
        for k, v in problems.items():
            print(f"[INVALID] {k}: {v}")
        raise SystemExit(1)
    print(f"seed={seed}; all datasets conform to contracts")
    for name, df in data.items():
        print(f"  {name:22s} {len(df):5d} rows")
    if not args.no_write:
        from .reports import write_reports
        for p in write(data):
            print(f"  wrote {p.relative_to(config.PROJECT_ROOT)}")
        for p in write_reports(data, sim, seed):
            print(f"  wrote {p.relative_to(config.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
