"""Load contract-typed datasets from disk."""
from __future__ import annotations

import pandas as pd

from src import config, schemas


class ContractError(ValueError):
    pass


def load_dataset(name: str, validate: bool = True) -> pd.DataFrame:
    """Read a dataset CSV with contract dtypes (NA stays NA). Raises ContractError if invalid."""
    c = schemas.ALL_CONTRACTS[name]
    df = pd.read_csv(config.path_for(c.layer) / c.file_name, dtype=str, keep_default_na=False, na_values=[""])
    df = schemas.coerce(df, c)
    if validate:
        issues = schemas.validate(df, c)
        if issues:
            raise ContractError(f"{name}: {issues}")
    return df


def load_sources(validate: bool = True) -> dict[str, pd.DataFrame]:
    return {name: load_dataset(name, validate) for name in schemas.SOURCE_DATASETS}
