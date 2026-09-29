from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "base.yaml"


def load_successful_split(
    processed_dir: Path,
    split: str,
) -> pd.DataFrame:
    """Load candidate metadata and retain only successful downloads."""

    candidate_path = processed_dir / f"{split}_candidates.csv"

    result_path = processed_dir / f"{split}_image_results.csv"

    candidates = pd.read_csv(candidate_path)
    results = pd.read_csv(result_path)

    results = results[results["success"] == True].copy()

    df = candidates.merge(
        results[
            [
                "id",
                "image_path",
                "sha256",
                "phash",
                "width",
                "height",
            ]
        ],
        on="id",
        how="inner",
        validate="one_to_one",
    )

    df["split"] = split

    return df


def remove_duplicate_images(
    data: pd.DataFrame,
    split_priority: list[str],
) -> tuple[pd.DataFrame, dict]:
    """
    Remove exact duplicate image content using SHA-256.

    Evaluation splits receive priority over training when
    the exact same image occurs across different splits.
    """

    priority = {split: index for index, split in enumerate(split_priority)}

    data = data.copy()

    data["_priority"] = data["split"].map(priority)

    duplicate_sha_count = int(data["sha256"].duplicated(keep=False).sum())

    # Keep evaluation data preferentially, then remove duplicate
    # image content from lower-priority splits.
    data = (
        data.sort_values(["sha256", "_priority"])
        .drop_duplicates(
            subset=["sha256"],
            keep="first",
        )
        .drop(columns="_priority")
        .reset_index(drop=True)
    )

    return data, {
        "duplicate_sha_rows_detected": duplicate_sha_count,
    }


def remove_conflicting_labels(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Remove image hashes associated with conflicting labels.

    This avoids training/evaluation ambiguity for identical images.
    """

    label_counts = data.groupby("sha256")["2_way_label"].nunique()

    conflicting_hashes = set(label_counts[label_counts > 1].index)

    conflicting_rows = int(data["sha256"].isin(conflicting_hashes).sum())

    data = data[~data["sha256"].isin(conflicting_hashes)].copy()

    return data, {
        "conflicting_image_hashes": len(conflicting_hashes),
        "conflicting_rows_removed": conflicting_rows,
    }


def sample_final_split(
    df: pd.DataFrame,
    size: int,
    seed: int,
) -> pd.DataFrame:
    """Create the final reproducible stratified split."""

    if len(df) < size:
        raise ValueError(
            f"Not enough clean samples: requested {size:,}, available {len(df):,}"
        )

    sampled, _ = train_test_split(
        df,
        train_size=size,
        stratify=df["2_way_label"],
        random_state=seed,
    )

    return sampled.reset_index(drop=True)


def main() -> None:
    config = load_config(CONFIG_PATH)

    processed_dir = PROJECT_ROOT / config["paths"]["processed_dir"]

    final_sizes = config["dataset"]["final_size"]
    seed = config["project"]["seed"]

    split_priority = config["dataset"]["split_priority"]

    # ---------------------------------------------------------
    # 1. Load successfully downloaded data
    # ---------------------------------------------------------

    split_frames = {}

    for split in ["train", "validate", "test"]:
        split_frames[split] = load_successful_split(
            processed_dir,
            split,
        )

        print(f"{split:10s}: {len(split_frames[split]):,} successful images")

    # ---------------------------------------------------------
    # 2. Combine for cross-split duplicate analysis
    # ---------------------------------------------------------

    combined = pd.concat(
        split_frames.values(),
        ignore_index=True,
    )

    print(f"\nCombined successful samples: {len(combined):,}")

    # ---------------------------------------------------------
    # 3. Check duplicate image URLs
    # ---------------------------------------------------------

    duplicate_urls = int(combined["image_url"].duplicated(keep=False).sum())

    print(f"Rows involved in duplicate URLs: {duplicate_urls:,}")

    # ---------------------------------------------------------
    # 4. Remove exact duplicate image content
    # ---------------------------------------------------------

    combined, duplicate_stats = remove_duplicate_images(
        combined,
        split_priority=split_priority,
    )

    print(
        "Rows involved in duplicate SHA-256 images: "
        f"{duplicate_stats['duplicate_sha_rows_detected']:,}"
    )

    # ---------------------------------------------------------
    # 5. Remove identical images with conflicting labels
    # ---------------------------------------------------------

    combined, conflict_stats = remove_conflicting_labels(combined)

    print(f"Conflicting image hashes: {conflict_stats['conflicting_image_hashes']:,}")

    print(
        "Rows removed because of conflicting labels: "
        f"{conflict_stats['conflicting_rows_removed']:,}"
    )

    # ---------------------------------------------------------
    # 6. Rebuild split-specific cleaned pools
    # ---------------------------------------------------------

    cleaned_splits = {
        split: combined[combined["split"] == split].copy()
        for split in ["train", "validate", "test"]
    }

    # ---------------------------------------------------------
    # 7. Sample exact final dataset sizes
    # ---------------------------------------------------------

    final_frames = {}

    for split, size in final_sizes.items():
        final_df = sample_final_split(
            cleaned_splits[split],
            size=size,
            seed=seed,
        )

        final_frames[split] = final_df

        print(f"{split:10s}: {len(final_df):,} final samples")

    # ---------------------------------------------------------
    # 8. Build final schema
    # ---------------------------------------------------------

    output_dir = processed_dir

    for split, df in final_frames.items():
        final_df = df[
            [
                "id",
                "clean_title",
                "image_path",
                "2_way_label",
                "image_url",
                "sha256",
                "phash",
                "width",
                "height",
            ]
        ].copy()

        # Store portable paths relative to data/processed/.
        final_df["image_path"] = final_df["image_path"].map(
            lambda path: Path(path).relative_to(processed_dir).as_posix()
        )

        final_df = final_df.rename(
            columns={
                "clean_title": "text",
                "2_way_label": "label",
            }
        )

        final_df.to_csv(
            output_dir / f"{split}.csv",
            index=False,
        )

    # ---------------------------------------------------------
    # 9. Save audit report
    # ---------------------------------------------------------

    audit = {
        "candidate_success_counts": {
            split: len(frame) for split, frame in split_frames.items()
        },
        "final_counts": {split: len(frame) for split, frame in final_frames.items()},
        "duplicate_url_rows": duplicate_urls,
        **duplicate_stats,
        **conflict_stats,
        "seed": seed,
    }

    with (output_dir / "dataset_audit.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            audit,
            file,
            indent=2,
        )

    print("\nFinal dataset construction complete.")


if __name__ == "__main__":
    main()
