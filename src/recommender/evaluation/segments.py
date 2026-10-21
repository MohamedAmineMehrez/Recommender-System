"""Cold/warm item and user segmentation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.features import Catalog


def item_segment_table(catalog: Catalog, item_counts: pd.Series) -> pd.DataFrame:
    counts = catalog.articles[["contentId"]].copy()
    counts["interaction_weight"] = counts["contentId"].map(item_counts).fillna(0.0)
    counts["segment"] = np.where(counts["interaction_weight"] > 0, "warm", "cold")
    return counts


def cold_item_ids(catalog: Catalog, item_counts: pd.Series) -> set[int]:
    warm = set(item_counts.index.tolist())
    return {int(item) for item in catalog.item_ids if int(item) not in warm}
