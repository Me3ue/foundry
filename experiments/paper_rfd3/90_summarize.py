#!/usr/bin/env python3
"""一键汇总入口（薄封装，真正逻辑在 lib/summarize.py）。

    python 90_summarize.py
    python 90_summarize.py --experiment exp2_ppi
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.summarize import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
