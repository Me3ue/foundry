#!/usr/bin/env python3
"""把 17_exp8_speed.sh 产出的 speed_scaling.csv 画成论文 Fig. 1d 风格的曲线。

刻意不依赖 matplotlib —— 直接输出 SVG，任何环境都能跑。

    python 91_plot_speed.py
    python 91_plot_speed.py --csv out/metrics/speed_scaling.csv --out out/reports/fig1d.svg
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib import common  # noqa: E402

W, H = 720, 460
PAD_L, PAD_R, PAD_T, PAD_B = 78, 150, 44, 62


def load(csv_path: Path) -> dict[str, dict[int, tuple[float, float]]]:
    data: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    with csv_path.open() as fh:
        for row in csv.DictReader(fh):
            try:
                data[row["backend"]][int(row["length"])].append(float(row["wallclock_s"]))
            except (KeyError, ValueError):
                continue
    out = {}
    for backend, by_len in data.items():
        out[backend] = {
            L: (statistics.mean(v), statistics.pstdev(v) if len(v) > 1 else 0.0)
            for L, v in by_len.items()
        }
    return out


def plot(data: dict, out_path: Path) -> None:
    import math

    colors = {"rfd3": "#0a7ea4", "rfd2": "#c2571a", "rfd1": "#6f42c1"}
    all_L = sorted({L for d in data.values() for L in d})
    all_v = [v for d in data.values() for v, _ in d.values()]
    if not all_L or not all_v:
        raise SystemExit("CSV 里没有可用数据")
    x_min, x_max = min(all_L), max(all_L)
    y_max = max(all_v) * 1.15

    def sx(L: float) -> float:
        span = (x_max - x_min) or 1
        return PAD_L + (L - x_min) / span * (W - PAD_L - PAD_R)

    def sy(v: float) -> float:
        return H - PAD_B - (v / y_max) * (H - PAD_T - PAD_B)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        f'width="{W}" height="{H}" font-family="Helvetica,Arial,sans-serif">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        f'<text x="{PAD_L}" y="26" font-size="16" font-weight="600" fill="#1a1a1a">'
        f'RFD3 inference time vs. protein length (paper Fig. 1d)</text>',
    ]
    # 坐标轴
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        v = y_max * frac
        y = sy(v)
        parts.append(f'<line x1="{PAD_L}" y1="{y:.1f}" x2="{W-PAD_R}" y2="{y:.1f}" '
                     f'stroke="#e6e6e6" stroke-width="1"/>')
        parts.append(f'<text x="{PAD_L-10}" y="{y+4:.1f}" font-size="11" fill="#666" '
                     f'text-anchor="end">{v:.0f}</text>')
    parts.append(f'<line x1="{PAD_L}" y1="{H-PAD_B}" x2="{W-PAD_R}" y2="{H-PAD_B}" '
                 f'stroke="#333" stroke-width="1.2"/>')
    parts.append(f'<line x1="{PAD_L}" y1="{PAD_T}" x2="{PAD_L}" y2="{H-PAD_B}" '
                 f'stroke="#333" stroke-width="1.2"/>')
    for L in all_L:
        parts.append(f'<text x="{sx(L):.1f}" y="{H-PAD_B+18}" font-size="11" fill="#666" '
                     f'text-anchor="middle">{L}</text>')
    parts.append(f'<text x="{(PAD_L+W-PAD_R)/2:.0f}" y="{H-18}" font-size="12" '
                 f'fill="#333" text-anchor="middle">protein length (aa)</text>')
    parts.append(f'<text x="18" y="{(PAD_T+H-PAD_B)/2:.0f}" font-size="12" fill="#333" '
                 f'transform="rotate(-90 18 {(PAD_T+H-PAD_B)/2:.0f})" '
                 f'text-anchor="middle">wall-clock per design (s)</text>')

    for i, (backend, by_len) in enumerate(sorted(data.items())):
        color = colors.get(backend, f"hsl({i*70},60%,45%)")
        pts = sorted(by_len.items())
        path = " ".join(f"{'M' if k == 0 else 'L'}{sx(L):.1f},{sy(v):.1f}"
                        for k, (L, (v, _)) in enumerate(pts))
        parts.append(f'<path d="{path}" fill="none" stroke="{color}" '
                     f'stroke-width="2.4" stroke-linejoin="round"/>')
        for L, (v, sd) in pts:
            parts.append(f'<circle cx="{sx(L):.1f}" cy="{sy(v):.1f}" r="3.4" '
                         f'fill="{color}"/>')
            if sd:
                parts.append(f'<line x1="{sx(L):.1f}" y1="{sy(v-sd):.1f}" '
                             f'x2="{sx(L):.1f}" y2="{sy(v+sd):.1f}" '
                             f'stroke="{color}" stroke-width="1.2" opacity="0.6"/>')
        # 图例
        ly = PAD_T + 12 + i * 22
        parts.append(f'<line x1="{W-PAD_R+14}" y1="{ly}" x2="{W-PAD_R+42}" y2="{ly}" '
                     f'stroke="{color}" stroke-width="2.6"/>')
        parts.append(f'<text x="{W-PAD_R+48}" y="{ly+4}" font-size="12" '
                     f'fill="#1a1a1a">{backend.upper()}</text>')
    parts.append("</svg>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description="画 Fig. 1d 速度曲线")
    ap.add_argument("--csv", default=str(common.METRICS_DIR / "speed_scaling.csv"))
    ap.add_argument("--out", default=str(common.REPORTS_DIR / "fig1d_speed.svg"))
    args = ap.parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"找不到 {csv_path}，请先运行 ./17_exp8_speed.sh", file=sys.stderr)
        return 1
    plot(load(csv_path), Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
