"""Export a clean LakePINN v10 source package.

The package intentionally contains only source code, tests, and versioned
experiment configuration. Results, checkpoints, caches, and remote artifacts
are excluded even if they are present in the working tree.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lake_pinn.source_package import export_source_package


def main() -> None:
    default_name = f"lake_pinn_v10_source_clean_{datetime.now():%Y%m%d_%H%M%S}.zip"
    parser = argparse.ArgumentParser(description="Export a clean LakePINN v10 source zip.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT.parent / default_name,
        help="Output zip path. Defaults to a timestamped file next to the v10 folder.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be included without writing a zip.",
    )
    args = parser.parse_args()

    files = export_source_package(ROOT, args.output.resolve(), dry_run=args.dry_run)
    if args.dry_run:
        for relative in files:
            print(relative.as_posix())
        print(f"DRY RUN: {len(files)} files would be included.")
    else:
        print(f"Wrote {args.output.resolve()} with {len(files)} files.")


if __name__ == "__main__":
    main()
