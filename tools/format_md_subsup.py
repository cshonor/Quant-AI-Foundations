# -*- coding: utf-8 -*-
"""
Conservative pass: add HTML <sub>/<sup> for common math shorthands in Markdown.
- Skips fenced ``` ... ``` blocks entirely.
- Inside inline `...` spans, skips (preserves code, paths, commands).
- Skips files that already look migrated (many <sub> tags).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET_DIRS = [
    ROOT / "MidLow_Freq" / "PSTAT_Basic",
    ROOT / "MidLow_Freq" / "PSTAT_Quant",
    ROOT / "MidLow_Freq" / "TS",
    ROOT / "MidLow_Freq" / "RFA",
    ROOT / "MidLow_Freq" / "ECON-CSPD",
    ROOT / "MidLow_Freq" / "Alg",
]
SKIP_SUB_COUNT = 12  # files with >= this many <sub> are skipped (e.g. ch4 already done)

TRIPLE_FENCE = re.compile(r"^```")


def split_inline_code(line: str) -> list[str]:
    parts = re.split(r"(`[^`]*`)", line)
    return parts


def fix_segment(seg: str) -> str:
    if not seg or seg.startswith("`"):
        return seg
    s = seg
    # --- superscripts (do before subscripts that might share chars) ---
    s = re.sub(r"\bsigma\^2\b", "σ<sup>2</sup>", s)
    s = re.sub(r"\bphi\^2\b", "φ<sup>2</sup>", s)
    s = re.sub(r"\be\^\(([^)]+)\)", r"e<sup>\1</sup>", s)  # e^(...) once
    s = re.sub(r"\be\^\{([^}]+)\}", r"e<sup>\1</sup>", s)
    # --- braced subscripts: x_{t-1}, I_{t-1} ---
    s = re.sub(r"\b([A-Za-z])_\{([^}]+)\}", r"\1<sub>\2</sub>", s)
    # --- phi_1, phi_2, ... ---
    s = re.sub(r"\bphi_(\d+)\b", r"φ<sub>\1</sub>", s)
    s = re.sub(r"\bphi_([a-z])\b", r"φ<sub>\1</sub>", s)
    # --- single-letter time subscript: x_t, w_t, y_t, z_t (lowercase letter + _t) ---
    s = re.sub(r"\b([a-z])_t\b", r"\1<sub>t</sub>", s)
    # --- common RV / statistics: X_n, F_n, ... (single capital + short subscript) ---
    def cap_sub(m: re.Match[str]) -> str:
        a, b = m.group(1), m.group(2)
        if len(b) > 8:
            return m.group(0)
        return f"{a}<sub>{b}</sub>"

    s = re.sub(r"\b([XYFZSWMRLKNIDF])_([a-zA-Z0-9+-]+)\b", cap_sub, s)
    # --- lim_{n->∞} variants ---
    s = re.sub(
        r"lim_\{n\s*->\s*∞\}",
        "**lim**<sub>n→∞</sub>",
        s,
    )
    s = re.sub(
        r"lim_\{n\s*->\s*infty\}",
        "**lim**<sub>n→∞</sub>",
        s,
        flags=re.I,
    )
    # --- n_A style ---
    s = re.sub(r"\bn_A\b", "n<sub>A</sub>", s)
    return s


def fix_line(line: str) -> str:
    parts = split_inline_code(line)
    out = []
    for p in parts:
        out.append(fix_segment(p))
    return "".join(out)


def process_file(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if text.count("<sub>") >= SKIP_SUB_COUNT:
        return False
    lines = text.splitlines(keepends=True)
    out_lines = []
    in_fence = False
    changed = False
    for line in lines:
        if TRIPLE_FENCE.match(line.strip()):
            in_fence = not in_fence
            out_lines.append(line)
            continue
        if in_fence:
            out_lines.append(line)
            continue
        new_line = fix_line(line)
        if new_line != line:
            changed = True
        out_lines.append(new_line)
    new_text = "".join(out_lines)
    if changed and new_text != text:
        path.write_text(new_text, encoding="utf-8", newline="")
        return True
    return False


def main() -> int:
    changed_files = []
    for d in TARGET_DIRS:
        if not d.is_dir():
            continue
        for path in sorted(d.rglob("*.md")):
            try:
                if process_file(path):
                    changed_files.append(path.relative_to(ROOT))
            except OSError as e:
                print(f"skip {path}: {e}", file=sys.stderr)
    for p in changed_files:
        print(f"updated: {p}")
    print(f"total updated: {len(changed_files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
