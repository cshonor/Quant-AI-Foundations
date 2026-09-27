"""把跑完的小节汇总成 site/index.html。"""
from __future__ import annotations

import html
import json
from pathlib import Path

CSS = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body { margin:0; background:#f6f7f9; color:#1f2328;
  font: 15px/1.65 -apple-system, BlinkMacSystemFont, "PingFang SC", "Segoe UI", sans-serif; }
header { padding:28px 32px 18px; background:#fff; border-bottom:1px solid #e5e7eb; }
header h1 { margin:0 0 6px; font-size:20px; font-weight:600; }
header p { margin:0; color:#6b7280; font-size:13px; }
main { max-width: 960px; margin:0 auto; padding: 20px 16px 60px; }
.card { background:#fff; border:1px solid #e5e7eb; border-radius:12px;
  padding:18px 20px; margin:16px 0; }
.card h2 { margin:0 0 4px; font-size:17px; font-weight:600; }
.card .num { display:inline-block; min-width:44px; color:#185FA5; font-variant-numeric: tabular-nums; }
.meta { margin:0 0 10px; color:#9ca3af; font-size:12px; }
blockquote { margin:10px 0; padding:8px 12px; border-left:3px solid #185FA5;
  background:#f1f6fc; color:#0c447c; font-size:14px; }
.prop { margin:8px 0 12px; font-size:14px; }
table.checks { width:100%; border-collapse:collapse; font-size:13px; margin:6px 0 12px; }
table.checks th { text-align:left; color:#6b7280; font-weight:500;
  border-bottom:1px solid #e5e7eb; padding:4px 8px; }
table.checks td { border-bottom:1px solid #f1f2f4; padding:5px 8px; vertical-align:top; }
td.m { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size:12px; width:52px; }
td.m.pass { color:#3B6D11; } td.m.fail { color:#A32D2D; } td.m.info { color:#6b7280; }
td.d { color:#6b7280; }
figure { margin:12px 0; }
figure img { width:100%; border:1px solid #e5e7eb; border-radius:8px; background:#fff; }
figcaption { color:#6b7280; font-size:12px; margin-top:4px; }
p.note { margin:8px 0; font-size:14px; padding:6px 10px; background:#fafafa;
  border-radius:8px; border:1px solid #f1f2f4; }
p.note b { color:#185FA5; font-weight:500; }
.badge { font-size:11px; padding:2px 8px; border-radius:999px; vertical-align:2px; margin-left:6px; }
.badge.ok { background:#EAF3DE; color:#3B6D11; }
.badge.bad { background:#FCEBEB; color:#A32D2D; }
.summary { display:flex; gap:14px; flex-wrap:wrap; color:#374151; font-size:13px; }
"""


def render(sections: list, out_dir: Path, title: str = "math-lab · 逐小节实验结果") -> Path:
    e = html.escape
    out_dir = Path(out_dir)
    (out_dir / "assets").mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps([s.to_dict() for s in sections], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    n_all = len(sections)
    n_ok = sum(1 for s in sections if s.passed)
    n_chk = sum(len(s.checks) for s in sections)
    body = "\n".join(s.card_html() for s in sections)
    now = sections[0].to_dict()["ran_at"] if sections else ""
    doc = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{e(title)}</title><style>{CSS}</style></head><body>
<header><h1>{e(title)}</h1>
<p class="summary"><span>小节 {n_all}</span><span>全部通过 {n_ok}</span>
<span>失败 {n_all - n_ok}</span><span>检查项 {n_chk}</span><span>生成于 {e(now)}</span></p>
<p>每个小节先写「要判决的命题」，再用数值实验判它对错；图只是顺手留下的证据。</p>
</header>
<main>{body}</main></body></html>"""
    idx = out_dir / "index.html"
    idx.write_text(doc, encoding="utf-8")
    return idx
