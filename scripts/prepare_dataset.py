from pathlib import Path

from src.datasets.prepare_fakeddit import prepare_metadata
from src.utils.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "base.yaml"


def main() -> None:
    config = load_config(CONFIG_PATH)

    raw_dir = PROJECT_ROOT / config["paths"]["raw_dir"]
    processed_dir = PROJECT_ROOT / config["paths"]["processed_dir"]

    dataset_config = config["dataset"]

    processed_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    base_dir = raw_dir / "multimodal_only_samples"

    paths = {
        "train": base_dir / "multimodal_train.tsv",
        "validate": base_dir / "multimodal_validate.tsv",
        "test": base_dir / "multimodal_test_public.tsv",
    }

    sizes = dataset_config["candidate_size"]

    prepared = prepare_metadata(
        paths=paths,
        sizes=sizes,
        seed=config["project"]["seed"],
    )

    for split_name, df in prepared.items():
        output_path = processed_dir / f"{split_name}_candidates.csv"

        df[
            [
                "id",
                "clean_title",
                "image_url",
                "2_way_label",
            ]
        ].to_csv(
            output_path,
            index=False,
        )

        print(f"{split_name:10s} -> {len(df):,} candidate samples -> {output_path}")


if __name__ == "__main__":
    main()
