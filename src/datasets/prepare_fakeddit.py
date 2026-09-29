from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

REQUIRED_COLUMNS = {
    "id",
    "clean_title",
    "image_url",
    "hasImage",
    "2_way_label",
}


def load_split(path: Path) -> pd.DataFrame:
    """Load a Fakeddit TSV split."""
    df = pd.read_csv(path, sep="\t")

    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns in {path.name}: {sorted(missing_columns)}"
        )

    return df


def filter_multimodal_samples(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only records suitable for multimodal processing.

    Image-byte validation is intentionally handled separately after
    metadata filtering.
    """
    filtered = df.copy()

    filtered["clean_title"] = filtered["clean_title"].astype("string")
    filtered["image_url"] = filtered["image_url"].astype("string")

    mask = (
        filtered["clean_title"].notna()
        & filtered["clean_title"].str.strip().ne("")
        & filtered["image_url"].notna()
        & filtered["image_url"].str.strip().ne("")
        & filtered["hasImage"].eq(True)
        & filtered["2_way_label"].isin([0, 1])
        & filtered["id"].notna()
    )

    filtered = filtered.loc[mask].copy()

    # Remove duplicate IDs defensively.
    filtered = filtered.drop_duplicates(subset=["id"], keep="first")

    return filtered


def sample_stratified(
    df: pd.DataFrame,
    n_samples: int,
    seed: int,
) -> pd.DataFrame:
    """Create a reproducible stratified sample."""
    if len(df) < n_samples:
        raise ValueError(
            f"Requested {n_samples} samples, but only {len(df)} are available."
        )

    sampled, _ = train_test_split(
        df,
        train_size=n_samples,
        stratify=df["2_way_label"],
        random_state=seed,
    )

    return sampled.reset_index(drop=True)


def prepare_metadata(
    paths: dict[str, Path],
    sizes: dict[str, int],
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """
    Load and filter the official Fakeddit splits, then create
    reproducible stratified candidate subsets.
    """
    prepared = {}

    for split_name, path in paths.items():
        df = load_split(path)
        df = filter_multimodal_samples(df)

        prepared[split_name] = sample_stratified(
            df,
            n_samples=sizes[split_name],
            seed=seed,
        )

    return prepared
