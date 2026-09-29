from pathlib import Path

import pandas as pd

from src.datasets.image_downloader import download_images
from src.utils.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "base.yaml"


def main() -> None:
    config = load_config(CONFIG_PATH)

    processed_dir = PROJECT_ROOT / config["paths"]["processed_dir"]

    image_config = config["image_download"]

    for split in (
        "train",
        "validate",
        "test",
    ):
        print("\n" + "=" * 70)
        print(f"Downloading {split} images")
        print("-" * 70)

        candidate_path = processed_dir / f"{split}_candidates.csv"

        df = pd.read_csv(candidate_path)

        image_dir = processed_dir / "images" / split

        results = download_images(
            df=df,
            image_dir=image_dir,
            timeout=image_config["timeout_seconds"],
            max_retries=image_config["max_retries"],
            workers=image_config["workers"],
            jpeg_quality=image_config["jpeg_quality"],
        )

        output_path = processed_dir / f"{split}_image_results.csv"

        results.to_csv(
            output_path,
            index=False,
        )

        successful = results["success"].sum()
        failed = len(results) - successful

        print(f"Requested : {len(results):,}")
        print(f"Successful: {successful:,}")
        print(f"Failed    : {failed:,}")

        print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
