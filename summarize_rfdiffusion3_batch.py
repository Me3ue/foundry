#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import re
from collections import Counter, defaultdict

ROOT = Path(".").resolve()
BATCH_LOG_DIR = ROOT / "logs" / "rfdiffusion3_batch"
OUT_ROOT = ROOT / "logs" / "inference_outs"
TEMPLATE_DIR = ROOT / "rfdiffusion3_inputs" / "unified_templates"
REPORT_PATH = BATCH_LOG_DIR / "batch_report.json"
MD_REPORT_PATH = BATCH_LOG_DIR / "batch_report.md"

VARIANT_ORDER = ["baseline", "fixed", "partial", "symmetry"]
VARIANT_LABELS = {
    "baseline": "Baseline",
    "fixed": "Fixed core",
    "partial": "Partial diffusion",
    "symmetry": "Symmetry",
}

SUCCESS_RE = re.compile(r"^(?P<pdb>[0-9A-Za-z]{4})/(?P<variant>[a-z]+) ok$")
FAIL_RE = re.compile(r"^(?P<pdb>[0-9A-Za-z]{4})/(?P<variant>[a-z]+) (?P<reason>.+)$")

ERROR_PATTERNS = [
    (re.compile(r"Input provided but unused in composition specification", re.I), "unused_input"),
    (re.compile(r"Residue\s+[A-Z]?[0-9]+\s+not found in atom array", re.I), "missing_residue"),
    (re.compile(r"missing input field", re.I), "missing_input_field"),
    (re.compile(r"missing pdb", re.I), "missing_pdb"),
    (re.compile(r"missing checkpoint", re.I), "missing_checkpoint"),
    (re.compile(r"no_symmetry_config", re.I), "no_symmetry_config"),
    (re.compile(r"ValidationError", re.I), "validation_error"),
]


@dataclass
class VariantResult:
    pdb_id: str
    variant: str
    status: str
    output_dir: str | None = None
    log_file: str | None = None
    reason: str | None = None
    reason_class: str | None = None


@dataclass
class ProteinSummary:
    pdb_id: str
    best_variant: str | None
    best_status: str | None
    all_variants: dict
    notes: list[str]


def load_template_meta() -> dict:
    meta = {}
    if not TEMPLATE_DIR.exists():
        return meta
    for path in TEMPLATE_DIR.glob("*.json"):
        if path.name == "summary.json":
            continue
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
            pdb_id = next(iter(obj))
            meta[pdb_id] = obj[pdb_id]
        except Exception:
            continue
    return meta


def classify_reason(reason: str, log_file: str | None = None) -> str:
    text = reason or ""
    if log_file:
        p = Path(log_file)
        if p.exists():
            try:
                text = p.read_text(encoding="utf-8", errors="ignore") + "\n" + text
            except Exception:
                pass
    for pattern, label in ERROR_PATTERNS:
        if pattern.search(text):
            return label
    if "exit_1" in text:
        return "exit_1"
    if "skipped_" in text:
        return "skipped"
    return "other"


def parse_logs() -> dict[tuple[str, str], VariantResult]:
    results: dict[tuple[str, str], VariantResult] = {}
    success_file = BATCH_LOG_DIR / "success.txt"
    fail_file = BATCH_LOG_DIR / "fail.txt"

    if success_file.exists():
        for line in success_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            m = SUCCESS_RE.match(line)
            if not m:
                continue
            pdb_id = m.group("pdb")
            variant = m.group("variant")
            results[(pdb_id, variant)] = VariantResult(
                pdb_id=pdb_id,
                variant=variant,
                status="ok",
                output_dir=str(OUT_ROOT / pdb_id / variant),
                log_file=str(BATCH_LOG_DIR / f"{pdb_id}_{variant}.log"),
                reason_class="ok",
            )

    if fail_file.exists():
        for line in fail_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            m = FAIL_RE.match(line)
            if not m:
                continue
            pdb_id = m.group("pdb")
            variant = m.group("variant")
            reason = m.group("reason")
            key = (pdb_id, variant)
            if key not in results:
                log_file = str(BATCH_LOG_DIR / f"{pdb_id}_{variant}.log")
                results[key] = VariantResult(
                    pdb_id=pdb_id,
                    variant=variant,
                    status="failed",
                    output_dir=str(OUT_ROOT / pdb_id / variant),
                    log_file=log_file,
                    reason=reason,
                    reason_class=classify_reason(reason, log_file=log_file),
                )

    return results


def score_variant(result: VariantResult) -> int:
    if result.status != "ok":
        return -10_000
    base = 0
    if result.variant == "fixed":
        base += 50
    elif result.variant == "partial":
        base += 40
    elif result.variant == "symmetry":
        base += 30
    elif result.variant == "baseline":
        base += 20

    out_dir = Path(result.output_dir) if result.output_dir else None
    if out_dir and out_dir.exists():
        try:
            file_count = sum(1 for p in out_dir.rglob("*") if p.is_file())
        except Exception:
            file_count = 0
        base += min(file_count, 25)
    return base


def choose_best_variant(variants: list[VariantResult]) -> VariantResult | None:
    if not variants:
        return None
    return sorted(variants, key=score_variant, reverse=True)[0]


def summarize() -> tuple[list[ProteinSummary], dict]:
    template_meta = load_template_meta()
    results = parse_logs()

    by_pdb: dict[str, list[VariantResult]] = defaultdict(list)
    for (pdb_id, _variant), result in results.items():
        by_pdb[pdb_id].append(result)

    failure_counter = Counter()
    failures_by_reason = defaultdict(list)
    for result in results.values():
        if result.status != "ok":
            cls = result.reason_class or classify_reason(result.reason or "", result.log_file)
            failure_counter[cls] += 1
            failures_by_reason[cls].append(f"{result.pdb_id}/{result.variant}")

    summaries: list[ProteinSummary] = []
    for pdb_id in sorted(template_meta.keys() | by_pdb.keys()):
        variants = by_pdb.get(pdb_id, [])
        best = choose_best_variant(variants)
        notes = []
        if pdb_id in template_meta:
            meta = template_meta[pdb_id]
            notes.append(f"chain={meta.get('selected_chain')}")
            notes.append(f"partial_t={meta.get('partial_t')}")
            if meta.get("symmetry"):
                notes.append("symmetry=yes")
            else:
                notes.append("symmetry=no")
        summaries.append(
            ProteinSummary(
                pdb_id=pdb_id,
                best_variant=best.variant if best else None,
                best_status=best.status if best else None,
                all_variants={
                    v.variant: {
                        "status": v.status,
                        "reason": v.reason,
                        "reason_class": v.reason_class,
                        "output_dir": v.output_dir,
                    }
                    for v in sorted(variants, key=lambda x: VARIANT_ORDER.index(x.variant) if x.variant in VARIANT_ORDER else 999)
                },
                notes=notes,
            )
        )

    report = {
        "total_targets": len(summaries),
        "failure_summary": {
            "by_class": dict(failure_counter),
            "examples": {k: v[:10] for k, v in failures_by_reason.items()},
        },
        "summaries": [asdict(s) for s in summaries],
    }
    return summaries, report


def render_markdown(summaries: list[ProteinSummary], report: dict) -> str:
    lines = []
    lines.append("# RFdiffusion3 batch report")
    lines.append("")
    lines.append(f"- Report dir: `{BATCH_LOG_DIR}`")
    lines.append(f"- Output dir: `{OUT_ROOT}`")
    lines.append("")
    lines.append("## Failure reason classification")
    lines.append("")
    failure_summary = report.get("failure_summary", {})
    by_class = failure_summary.get("by_class", {})
    if by_class:
        lines.append("| Class | Count | Examples |")
        lines.append("|---|---:|---|")
        examples = failure_summary.get("examples", {})
        for cls, count in sorted(by_class.items(), key=lambda kv: (-kv[1], kv[0])):
            ex = ", ".join(examples.get(cls, [])[:5]) if examples.get(cls) else ""
            lines.append(f"| {cls} | {count} | {ex} |")
    else:
        lines.append("No failures recorded.")

    lines.append("")
    lines.append("## Best-condition summary")
    lines.append("")
    lines.append("| PDB | Best condition | Status | Notes |")
    lines.append("|---|---|---|---|")
    for s in summaries:
        best = s.best_variant or "-"
        status = s.best_status or "missing"
        notes = "; ".join(s.notes) if s.notes else ""
        lines.append(f"| {s.pdb_id} | {VARIANT_LABELS.get(best, best)} | {status} | {notes} |")

    lines.append("")
    lines.append("## Per-protein details")
    for s in summaries:
        lines.append(f"### {s.pdb_id}")
        lines.append("")
        if s.best_variant:
            lines.append(f"Best condition: **{VARIANT_LABELS.get(s.best_variant, s.best_variant)}** ({s.best_status})")
        else:
            lines.append("Best condition: **none**")
        if s.notes:
            lines.append(f"Notes: {'; '.join(s.notes)}")
        lines.append("")
        lines.append("| Variant | Status | Output | Reason | Reason class |")
        lines.append("|---|---|---|---|---|")
        for variant in VARIANT_ORDER:
            info = s.all_variants.get(variant)
            if info is None:
                lines.append(f"| {VARIANT_LABELS[variant]} | missing | - | - | - |")
            else:
                lines.append(
                    f"| {VARIANT_LABELS[variant]} | {info.get('status', '-') } | {info.get('output_dir', '-') } | {info.get('reason', '-') or '-'} | {info.get('reason_class', '-') or '-'} |"
                )
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    summaries, report = summarize()
    BATCH_LOG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    MD_REPORT_PATH.write_text(render_markdown(summaries, report), encoding="utf-8")

    print(f"wrote {REPORT_PATH}")
    print(f"wrote {MD_REPORT_PATH}")

    print("\nFailure reason summary:")
    for cls, count in sorted(report.get("failure_summary", {}).get("by_class", {}).items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"- {cls}: {count}")

    print("\nBest condition per protein:")
    for s in summaries:
        print(f"- {s.pdb_id}: {s.best_variant or 'none'} ({s.best_status or 'missing'})")


if __name__ == "__main__":
    main()
