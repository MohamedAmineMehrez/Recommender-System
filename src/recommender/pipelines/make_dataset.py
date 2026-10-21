"""Build the canonical processed dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config import load_config
from ..data.dataset import build_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean raw data and build chronological splits")
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset = build_dataset(config)
    output_dir = Path(config["data"]["processed_dir"])
    dataset.save(output_dir)
    print(json.dumps(dataset.stats, indent=2, default=str))
    print(f"Saved processed dataset to {output_dir}")


if __name__ == "__main__":
    main()
