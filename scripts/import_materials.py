"""Import the approved topic DOCX into study JSON.

Run with --help for required input paths, selection options, and output files.
Supply review-local artifacts and endpoints when exercising this utility.
Runtime files and credentials are not part of the source distribution.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from docx import Document


HEADING_RE = re.compile(r"^##\s*(.*?)\s*[（(](.*?)[）)]\s*$")
ITEM_RE = re.compile(r"^(\d+)\.\s*(.+)$")


def import_topics(source: Path) -> dict:
    categories: list[dict] = []
    current: dict | None = None

    for paragraph in Document(source).paragraphs:
        text = paragraph.text.strip()
        if not text or text == "---":
            continue
        heading = HEADING_RE.match(text)
        if heading:
            current = {
                "id": re.sub(r"[^a-z0-9]+", "_", heading.group(1).lower()).strip("_"),
                "label_en": heading.group(1).strip(),
                "label_zh": heading.group(2).strip(),
                "prompts": [],
            }
            categories.append(current)
            continue
        item = ITEM_RE.match(text)
        if item and current is not None:
            current["prompts"].append(
                {"id": f"{current['id']}_{int(item.group(1)):02d}", "text": item.group(2)}
            )

    if len(categories) != 10 or any(len(category["prompts"]) != 15 for category in categories):
        counts = [len(category["prompts"]) for category in categories]
        raise ValueError(f"Expected 10 categories x 15 prompts, got {len(categories)}: {counts}")

    return {
        "version": "topics-2026-07-13-v1",
        "source": source.name,
        "presentation": "same_pool_all_days_free_choice",
        "categories": categories,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Import the approved topic DOCX into study JSON.")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    payload = import_topics(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
