"""Canonical cleaning for the raw articles and interaction tables."""

from __future__ import annotations

import re
import unicodedata
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd

EVENT_WEIGHTS: dict[str, float] = {
    "VIEW": 1.0,
    "LIKE": 2.0,
    "BOOKMARK": 2.5,
    "FOLLOW": 3.0,
    "COMMENT CREATED": 4.0,
}


def _normalize_text(value: Any) -> tuple[str, int]:
    """Strip replacement characters and normalize whitespace.

    The raw CSV contains U+FFFD characters where UTF-8 bytes were previously
    decoded as Latin-1. The original bytes cannot be recovered reliably, so we
    remove the markers and log how many were found.
    """
    if not isinstance(value, str):
        return "", 0
    replacement_count = value.count("\ufffd")
    value = value.replace("\ufffd", "")
    value = unicodedata.normalize("NFKC", value)
    value = value.replace("\x00", " ")
    value = re.sub(r"\s+", " ", value).strip()
    return value, replacement_count


def _source_domain(url: Any) -> str:
    if not isinstance(url, str) or not url:
        return "unknown"
    try:
        host = urlparse(url).netloc.lower()
    except ValueError:
        return "unknown"
    if not host:
        return "unknown"
    if host.startswith("www."):
        host = host[4:]
    return host


def clean_articles(raw: pd.DataFrame) -> tuple[pd.DataFrame, set[int], dict[str, Any]]:
    stats: dict[str, Any] = {"raw_rows": int(len(raw))}
    removed_mask = raw["eventType"].eq("CONTENT REMOVED")
    removed_ids = set(
        pd.to_numeric(raw.loc[removed_mask, "contentId"], errors="coerce")
        .dropna()
        .astype("int64")
        .tolist()
    )
    stats["removed_rows"] = int(removed_mask.sum())
    stats["removed_ids"] = int(len(removed_ids))

    articles = raw.loc[~removed_mask].copy()
    articles["contentId"] = pd.to_numeric(articles["contentId"], errors="coerce")
    articles["timestamp"] = pd.to_numeric(articles["timestamp"], errors="coerce")
    before = len(articles)
    articles = articles.dropna(subset=["contentId", "timestamp"]).copy()
    stats["invalid_rows"] = int(before - len(articles))
    articles["contentId"] = articles["contentId"].astype("int64")
    articles["timestamp"] = articles["timestamp"].astype("int64")

    articles = articles.sort_values(["contentId", "timestamp"], kind="stable")
    before = len(articles)
    articles = articles.drop_duplicates("contentId", keep="first").copy()
    stats["duplicate_content_ids"] = int(before - len(articles))

    replacement_chars = 0
    clean_titles: list[str] = []
    clean_texts: list[str] = []
    for title, text in zip(articles.get("title", ""), articles.get("text", "")):
        title_clean, title_replacements = _normalize_text(title)
        text_clean, text_replacements = _normalize_text(text)
        replacement_chars += title_replacements + text_replacements
        clean_titles.append(title_clean)
        clean_texts.append(text_clean)
    articles["title"] = clean_titles
    articles["text"] = clean_texts
    articles["text_all"] = (
        articles["title"].fillna("") + " " + articles["text"].fillna("")
    ).str.strip()
    stats["replacement_chars_removed"] = int(replacement_chars)
    stats["articles_with_replacement_chars"] = int(
        sum(("\ufffd" in str(t)) or ("\ufffd" in str(x)) for t, x in zip(raw.get("title", ""), raw.get("text", "")))
    )

    articles["share_timestamp"] = pd.to_datetime(articles["timestamp"], unit="s", utc=True)
    articles["share_ts"] = articles["timestamp"].astype("int64")
    articles["lang"] = (
        articles.get("lang", pd.Series(index=articles.index, dtype=object))
        .fillna("unknown")
        .astype(str)
        .str.lower()
        .str.strip()
        .replace({"": "unknown"})
    )
    articles["contentType"] = articles.get("contentType", "unknown").fillna("unknown").astype(str)
    articles["url"] = articles.get("url", "").fillna("").astype(str)
    articles["source_domain"] = articles["url"].map(_source_domain)
    for column in ("authorPersonId", "authorSessionId", "authorUserAgent", "authorRegion", "authorCountry"):
        if column not in articles.columns:
            articles[column] = "unknown"
        articles[column] = articles[column].fillna("unknown").astype(str)
    articles["author_id"] = articles["authorPersonId"].replace({"unknown": "unknown"})

    articles = articles.reset_index(drop=True)
    stats["clean_articles"] = int(len(articles))
    stats["languages"] = articles["lang"].value_counts().to_dict()
    return articles, removed_ids, stats


def clean_interactions(
    raw: pd.DataFrame,
    articles: pd.DataFrame,
    removed_ids: set[int],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    stats: dict[str, Any] = {"raw_rows": int(len(raw))}
    interactions = raw.copy()
    for column in ("personId", "contentId", "timestamp"):
        interactions[column] = pd.to_numeric(interactions[column], errors="coerce")
    before = len(interactions)
    interactions = interactions.dropna(subset=["personId", "contentId", "timestamp"]).copy()
    stats["invalid_rows"] = int(before - len(interactions))
    interactions["personId"] = interactions["personId"].astype("int64")
    interactions["contentId"] = interactions["contentId"].astype("int64")
    interactions["timestamp"] = interactions["timestamp"].astype("int64")

    before = len(interactions)
    interactions = interactions.loc[~interactions["contentId"].isin(removed_ids)].copy()
    stats["removed_article_interactions"] = int(before - len(interactions))

    known_items = set(articles["contentId"].tolist())
    before = len(interactions)
    interactions = interactions.loc[interactions["contentId"].isin(known_items)].copy()
    stats["unknown_article_interactions"] = int(before - len(interactions))

    interactions["weight"] = interactions["eventType"].map(EVENT_WEIGHTS)
    before = len(interactions)
    interactions = interactions.dropna(subset=["weight"]).copy()
    stats["unknown_event_type_interactions"] = int(before - len(interactions))

    share_ts = articles.set_index("contentId")["share_ts"]
    interactions["share_ts"] = interactions["contentId"].map(share_ts)
    before = len(interactions)
    interactions = interactions.loc[interactions["timestamp"] >= interactions["share_ts"]].copy()
    stats["before_share_interactions"] = int(before - len(interactions))
    interactions = interactions.drop(columns=["share_ts"])

    interactions["datetime"] = pd.to_datetime(interactions["timestamp"], unit="s", utc=True)
    interactions = interactions.sort_values(["timestamp", "personId", "contentId"], kind="stable")
    interactions = interactions.reset_index(drop=True)
    stats["clean_interactions"] = int(len(interactions))
    stats["users"] = int(interactions["personId"].nunique())
    stats["items"] = int(interactions["contentId"].nunique())
    stats["events"] = interactions["eventType"].value_counts().to_dict()
    return interactions, stats


def build_first_events(interactions: pd.DataFrame) -> pd.DataFrame:
    """Return the first interaction for every (personId, contentId) pair."""
    first = (
        interactions.sort_values(["timestamp", "personId", "contentId"], kind="stable")
        .drop_duplicates(["personId", "contentId"], keep="first")
        .copy()
        .reset_index(drop=True)
    )
    first = first.rename(columns={"timestamp": "first_ts", "datetime": "first_timestamp"})
    first["eventStrength"] = first["weight"]
    return first
