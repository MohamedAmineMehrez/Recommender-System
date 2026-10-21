"""Raw data loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def load_raw_articles(config: dict[str, Any]) -> pd.DataFrame:
    path = Path(config["data"]["articles_path"])
    return pd.read_csv(path)


def load_raw_interactions(config: dict[str, Any]) -> pd.DataFrame:
    path = Path(config["data"]["interactions_path"])
    return pd.read_csv(path)
