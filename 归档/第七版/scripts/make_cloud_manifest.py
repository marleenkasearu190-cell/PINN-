"""Rewrite a LakePINN manifest to use cloud Linux data paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DATA_FILES = {
    "era5": "era5_for_model.csv",
    "lst": "lst_night_for_model.csv",
    "profile_obs": "profile_for_model.csv",
    "metadata": "metadata.json",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Source manifest JSON.")
    parser.add_argument("--output", required=True, help="Output cloud manifest JSON.")
    parser.add_argument(
        "--data-root",
        default="/root/LakePINN/data/_standard_inputs",
        help="Cloud directory containing lake_id subdirectories.",
    )
    parser.add_argument(
        "--experiment-suffix",
        default="_cloud",
        help="Suffix appended to the manifest experiment name.",
    )
    args = parser.parse_args()

    source = Path(args.input)
    manifest = json.loads(source.read_text(encoding="utf-8"))
    experiment = str(manifest.get("experiment", source.stem))
    if args.experiment_suffix and not experiment.endswith(args.experiment_suffix):
        manifest["experiment"] = f"{experiment}{args.experiment_suffix}"

    data_root = args.data_root.rstrip("/")
    for lake in manifest.get("lakes", []):
        lake_id = lake["lake_id"]
        for key, filename in DATA_FILES.items():
            lake[key] = f"{data_root}/{lake_id}/{filename}"

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
