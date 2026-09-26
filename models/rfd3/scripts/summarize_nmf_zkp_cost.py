#!/usr/bin/env python
"""Summarize training cost/resource measurements from an NMF sweep."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

_SUMMARY_SUFFIX = ".run_summary.json"
_STEP_RE = re.compile(r"\.step(\d+)$")


def tag_and_step(path: Path) -> tuple[str, int]:
    """`<tag>.run_summary.json` / `<tag>.step<N>.run_summary.json` -> (tag, step).

    train_lora._get_run_summary_path names the file after hydra's `name` and adds
    a `.step<N>` suffix once a global step exists, so the tag cannot be obtained
    by stripping the suffix alone.
    """
    stem = path.name
    if stem.endswith(_SUMMARY_SUFFIX):
        stem = stem[: -len(_SUMMARY_SUFFIX)]
    match = _STEP_RE.search(stem)
    if match:
        return stem[: match.start()], int(match.group(1))
    return stem, -1


def discover_tags(sweep_dir: Path) -> list[str]:
    """Discover current single-layer jobs from summaries or training directories."""
    tags = set()
    for path in sweep_dir.glob(f"*{_SUMMARY_SUFFIX}"):
        tags.add(tag_and_step(path)[0])
    train_root = sweep_dir / "train"
    if train_root.exists():
        tags.update(path.name for path in train_root.iterdir() if path.is_dir())
    return sorted(tags, key=lambda tag: (tag != "baseline", tag))


def find_summary(sweep_dir: Path, tag: str) -> dict:
    """Return the summary belonging to exactly `tag`, preferring the latest step.

    A plain `sorted(glob(f"{tag}*.run_summary.json"))[-1]` also matched tags that
    merely *begin* with `tag` and then kept the alphabetically last one, so in a
    `LAYER_SET=all` sweep `process_z` reported the cost of `process_z_init`
    ('.' sorts before '_'). Match the tag exactly instead.
    """
    matches = [
        path
        for path in sweep_dir.glob(f"*{_SUMMARY_SUFFIX}")
        if tag_and_step(path)[0] == tag
    ]
    if not matches:
        return {}
    best = max(matches, key=lambda path: (tag_and_step(path)[1], path.name))
    try:
        return json.loads(best.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_dir", type=Path)
    args = ap.parse_args()
    rows = []
    for tag in discover_tags(args.sweep_dir):
        summary = find_summary(args.sweep_dir, tag)
        cost = summary.get("cost", {})
        wall = cost.get("wall_time_seconds")
        wall_file = args.sweep_dir / f"{tag}.wall_time_seconds"
        if wall is None and wall_file.exists():
            wall = float(wall_file.read_text().strip())
        rows.append({
            "setting": tag,
            "status": summary.get("status", "missing"),
            "wall_time_seconds": wall,
            "wall_time_hours": None if wall is None else wall / 3600.0,
            "gpu_count": cost.get("gpu_count"),
            "gpu_peak_allocated_bytes": cost.get("gpu_peak_allocated_bytes"),
            "gpu_peak_allocated_gib": None if cost.get("gpu_peak_allocated_bytes") is None else cost["gpu_peak_allocated_bytes"] / 2**30,
            "gpu_peak_reserved_bytes": cost.get("gpu_peak_reserved_bytes"),
            "gpu_peak_reserved_gib": None if cost.get("gpu_peak_reserved_bytes") is None else cost["gpu_peak_reserved_bytes"] / 2**30,
            "cpu_max_rss_bytes": cost.get("cpu_max_rss_bytes"),
            "cpu_max_rss_gib": None if cost.get("cpu_max_rss_bytes") is None else cost["cpu_max_rss_bytes"] / 2**30,
            "output_dir_size_bytes": cost.get("output_dir_size_bytes"),
            "output_dir_size_gib": None if cost.get("output_dir_size_bytes") is None else cost["output_dir_size_bytes"] / 2**30,
            "trainable_params": summary.get("trainable_params"),
            "total_params": summary.get("total_params"),
            "trainable_fraction": (summary["trainable_params"] / summary["total_params"]
                                   if summary.get("trainable_params") is not None and summary.get("total_params") else None),
        })
    df = pd.DataFrame(rows)
    df.to_csv(args.sweep_dir / "paper_training_cost.csv", index=False)
    (args.sweep_dir / "paper_training_cost.md").write_text(
        "# Training cost and resource summary\n\n" + df.to_markdown(index=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.sweep_dir / 'paper_training_cost.csv'}")
    print(f"Wrote {args.sweep_dir / 'paper_training_cost.md'}")


if __name__ == "__main__":
    main()
