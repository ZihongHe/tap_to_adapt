"""Create a consent-aware, pseudonymized Tap-to-Adapt dataset release.

Run with --help for required input paths, selection options, and output files.
Supply review-local artifacts and endpoints when exercising this utility.
Runtime files and credentials are not part of the source distribution.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.dataset_release import export_research_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a consent-aware, pseudonymized Tap-to-Adapt dataset release."
    )
    parser.add_argument("--run-slug", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        help="Release directory. Defaults to data/dataset_exports/<run>-<timestamp>.",
    )
    parser.add_argument("--study-db", type=Path, default=ROOT / "data" / "study.sqlite3")
    parser.add_argument("--consent-db", type=Path, default=ROOT / "data" / "consent.sqlite3")
    parser.add_argument("--media-dir", type=Path, default=ROOT / "data" / "media")
    parser.add_argument("--audio-review-csv", type=Path)
    parser.add_argument("--include-controlled-audio", action="store_true")
    args = parser.parse_args()
    pepper = os.getenv("DATASET_EXPORT_PEPPER", "")
    if not pepper:
        parser.error("DATASET_EXPORT_PEPPER must be set and must not be the study code/IP pepper")
    output = (
        args.output.resolve()
        if args.output
        else (
            ROOT
            / "data"
            / "dataset_exports"
            / f"{args.run_slug}-{time.strftime('%Y%m%d-%H%M%S')}"
        ).resolve()
    )
    result = export_research_dataset(
        study_db=args.study_db.resolve(),
        consent_db=args.consent_db.resolve(),
        media_dir=args.media_dir.resolve(),
        output_dir=output,
        run_slug=args.run_slug,
        export_pepper=pepper,
        include_controlled_audio=args.include_controlled_audio,
        audio_review_csv=(
            args.audio_review_csv.resolve() if args.audio_review_csv else None
        ),
    )
    print(
        f"release={result.output_dir} public_participants={result.public_participants} "
        f"controlled_participants={result.controlled_participants} "
        f"controlled_audio_files={result.controlled_audio_files}"
    )


if __name__ == "__main__":
    main()
