from pathlib import Path

import pandas as pd

from src.datasets.image_downloader import download_images
from src.utils.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "base.yaml"


def main() -> None:
    config = load_config(CONFIG_PATH)

    candidates_path = (
        PROJECT_ROOT / config["paths"]["processed_dir"] / "train_candidates.csv"
    )

    image_dir = PROJECT_ROOT / "data" / "processed" / "images" / "pilot"

    df = pd.read_csv(candidates_path)

    # Pilot only — deliberately small.
    df = df.head(100).copy()

    settings = config["image_download"]

    results = download_images(
        df=df,
        image_dir=image_dir,
        timeout=settings["timeout_seconds"],
        max_retries=settings["max_retries"],
        workers=settings["workers"],
        jpeg_quality=settings["jpeg_quality"],
    )

    print("\nImage download pilot results")
    print("=" * 50)

    print(f"Requested : {len(results):,}")
    print(f"Successful: {results['success'].sum():,}")
    print(f"Failed    : {(~results['success']).sum():,}")

    if (~results["success"]).any():
        print("\nSample failures:")
        print(
            results.loc[
                ~results["success"],
                ["id", "error"],
            ]
            .head(10)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
