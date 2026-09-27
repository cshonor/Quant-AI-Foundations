"""跑所有小节脚本并汇总出站点。

    python run_all.py                # 跑全部
    python run_all.py 1.1 1.3        # 只跑指定编号（前缀匹配即可）
    MATHLAB_OUT=site python run_all.py

每个小节脚本的契约很简单：模块级有一个 `SEC: mathviz.Section`，
以及一个 `def run(sec: mathviz.Section) -> None`。
"""
from __future__ import annotations

import importlib.util
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from mathviz import Section  # noqa: E402
from mathviz.site import render  # noqa: E402

OUT_DIR = Path(ROOT / "site")
SKIP_DIRS = {"site", "mathviz", "__pycache__", ".git", ".venv", "venv", ".idea"}


def discover() -> list[Path]:
    files = []
    for p in sorted(ROOT.rglob("*.py")):
        if any(part in SKIP_DIRS or part.startswith(".") for part in p.parts):
            continue
        if p.name.startswith("_") or p.name in {"run_all.py"}:
            continue
        # 只收 N.N-*.py 这种编号文件
        head = p.name.split("-", 1)[0]
        if len(head.split(".")) == 2 and head.replace(".", "").isdigit():
            files.append(p)
    return files


def load(path: Path):
    spec = importlib.util.spec_from_file_location(f" sec_{path.stem.replace('-', '_')}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv: list[str]) -> int:
    OUT_DIR.mkdir(exist_ok=True)
    pick = argv[1:]
    paths = [p for p in discover() if not pick or any(p.name.startswith(t) for t in pick)]
    if not paths:
        print("没找到匹配的小节脚本")
        return 1

    sections: list[Section] = []
    failed_files: list[tuple[Path, str]] = []
    for p in paths:
        print(f"── 运行 {p.relative_to(ROOT)}", flush=True)
        try:
            mod = load(p)
            sec: Section = mod.SEC
            sec.out_dir = OUT_DIR
            if hasattr(mod, "run"):
                mod.run(sec)
            sections.append(sec)
            print(sec.summary())
        except Exception:
            failed_files.append((p, traceback.format_exc()))
            print(traceback.format_exc())

    sections.sort(key=lambda s: [int(x) for x in s.number.split(".")])
    idx = render(sections, OUT_DIR)
    n_ok = sum(1 for s in sections if s.passed)
    print(f"\n✅ 站点已生成: {idx}")
    print(f"   小节 {len(sections)} / 全通过 {n_ok} / 脚本报错 {len(failed_files)}")
    for p, tb in failed_files:
        print(f"   [脚本异常] {p.name}: {tb.strip().splitlines()[-1]}")
    return 0 if not failed_files else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
