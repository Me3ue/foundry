#!/usr/bin/env python3
"""一键汇总入口（薄封装，真正逻辑在 lib/summarize.py）。

    python 90_summarize.py
    python 90_summarize.py --experiment exp2_ppi
"""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

try:
    from lib.summarize import main  # noqa: E402
except ModuleNotFoundError as _exc:  # 通常是同步时漏了 lib/ 子目录
    raise SystemExit(
        f"无法导入 lib 包（{_exc}）\n"
        f"  脚本目录 : {_HERE}\n"
        f"  期望文件 : {_HERE / 'lib' / 'common.py'}\n"
        f"  lib/ 应含: __init__.py common.py metrics.py summarize.py gpu_pool.py\n"
        f"  原因：同步脚本时只传了单个 .py 文件，没带上 lib/ 子目录。\n"
        f"  修法（在本地仓库执行，源路径结尾的斜杠不能省 —— 那是「传目录内容」的意思）：\n"
        f"    rsync -av --exclude out/ --exclude __pycache__/ \\\n"
        f"      experiments/paper_rfd3/ \\\n"
        f"      zhangzijian@hit:/backup01/zzj/protein/foundry/experiments/paper_rfd3/"
    ) from None

if __name__ == "__main__":
    raise SystemExit(main())
