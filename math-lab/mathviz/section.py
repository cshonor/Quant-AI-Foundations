"""一小节的结构化容器。

用法（每个小节脚本都长这个样子）：

    SEC = Section(
        number="1.1",
        chapter="第 1 章 · 向量",
        title="内积、范数与柯西-施瓦茨不等式",
        claim="书上原文/结论：|u·v| ≤ ‖u‖·‖v‖，等号当且仅当 u, v 线性相关。",
        proposition="可证伪命题：随机向量 →命题要能判对错，不能只是"展示一下"。",
    )

    def run(sec: Section) -> None:
        ...
        sec.check("10 万组随机向量都不违背 C-S", ok, f"最大相对超出 {m:.2e}")
        sec.figure("cs-distribution", fig)
        sec.observe("...)

    if __name__ == "__main__":
        run(SEC)
        print(SEC.summary())
"""
from __future__ import annotations

import html
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_OUT = Path(os.environ.get("MATHLAB_OUT", "site"))


@dataclass
class Check:
    name: str
    ok: bool | None          # True / False / None(= 只报告不判定)
    detail: str = ""
    tolerance: str = ""

    @property
    def mark(self) -> str:
        return "PASS" if self.ok else ("FAIL" if self.ok is False else "INFO")


@dataclass
class Section:
    number: str
    title: str
    chapter: str = ""
    book: str = ""
    claim: str = ""          # 书上怎么说
    proposition: str = ""    # 本节要判决的那个命题
    source: str = ""         # 页码/小节出处
    out_dir: Path = DEFAULT_OUT

    checks: list[Check] = field(default_factory=list)
    figures: list[dict[str, str]] = field(default_factory=list)
    notes: list[dict[str, str]] = field(default_factory=list)

    # ------------------------------------------------------------ 收集
    @staticmethod
    def _norm_ok(ok: Any) -> bool | None:
        """把判定值归一成 Python 的 True/False/None。

        numpy 比较返回的是 np.bool_，它不是 Python 的 False 单例，
        `ok is False` 会漏判，进而把 FAIL 显示成 INFO —— 这里先把 *.item() 解开。
        """
        if ok is None:
            return None
        if hasattr(ok, "item"):          # numpy 标量 / 0 维数组
            try:
                ok = ok.item()
            except Exception:
                pass
        return bool(ok)

    def check(self, name: str, ok: bool | None, detail: str = "", tolerance: str = "") -> Check:
        c = Check(name=name, ok=self._norm_ok(ok), detail=detail, tolerance=tolerance)
        self.checks.append(c)
        return c

    def info(self, name: str, detail: str = "") -> Check:
        return self.check(name, None, detail)

    def observe(self, text: str, kind: str = "观察") -> None:
        """「实际看到什么」与「书上说什么」的对照。"""
        self.notes.append({"kind": kind, "text": text})

    def pitfall(self, text: str) -> None:
        self.observe(text, kind="坑")

    def figure(self, name: str, fig: Any = None, caption: str = "") -> str:
        """存图。返回相对路径，供 HTML 引用。"""
        import matplotlib.pyplot as plt

        fig = fig if fig is not None else plt.gcf()
        assets = Path(self.out_dir) / "assets"
        assets.mkdir(parents=True, exist_ok=True)
        rel = f"assets/{self.number}-{name}.png"
        fig.savefig(Path(self.out_dir) / rel, dpi=130, bbox_inches="tight")
        plt.close(fig)
        self.figures.append({"path": rel, "caption": caption or name})
        return rel

    # ------------------------------------------------------------ 输出
    @property
    def passed(self) -> bool:
        judged = [c for c in self.checks if c.ok is not None]
        return bool(judged) and all(c.ok for c in judged)

    @property
    def n_pass(self) -> int:
        return sum(1 for c in self.checks if c.ok is True)

    @property
    def n_fail(self) -> int:
        return sum(1 for c in self.checks if c.ok is False)

    def summary(self) -> str:
        lines = [
            f"[{self.number}] {self.title}",
            f"  书上结论 : {self.claim}",
            f"  待验证   : {self.proposition}",
        ]
        for c in self.checks:
            lines.append(f"  [{c.mark:4}] {c.name}" + (f"  ({c.detail})" if c.detail else ""))
        for n in self.notes:
            lines.append(f"  - {n['kind']}: {n['text']}")
        for f in self.figures:
            lines.append(f"  - 图: {f['path']}")
        lines.append(f"  => {'全部通过' if self.passed else '有失败项，回去看书'}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "number": self.number,
            "title": self.title,
            "chapter": self.chapter,
            "book": self.book,
            "claim": self.claim,
            "proposition": self.proposition,
            "source": self.source,
            "checks": [
                {"name": c.name, "mark": c.mark, "detail": c.detail, "tolerance": c.tolerance}
                for c in self.checks
            ],
            "figures": self.figures,
            "notes": self.notes,
            "passed": self.passed,
            "n_pass": self.n_pass,
            "n_fail": self.n_fail,
            "ran_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    def card_html(self) -> str:
        e = html.escape
        badge = ("<span class='badge ok'>全部通过</span>" if self.passed
                 else "<span class='badge bad'>有失败项</span>")
        rows = "".join(
            f"<tr><td class='m {c.mark.lower()}'>{c.mark}</td>"
            f"<td>{e(c.name)}</td><td class='d'>{e(c.detail)}</td></tr>"
            for c in self.checks
        ) or "<tr><td colspan=3 class='d'>（本节没有检查项 —— A 类小节不允许）</td></tr>"
        figs = "".join(
            f"<figure><img src='{f['path']}' alt='{e(f['caption'])}'>"
            f"<figcaption>{e(f['caption'])}</figcaption></figure>"
            for f in self.figures
        )
        notes = "".join(
            f"<p class='note'><b>{e(n['kind'])}</b> · {e(n['text'])}</p>" for n in self.notes
        )
        return f"""<section class='card' id='{e(self.number)}'>
<h2><span class='num'>{e(self.number)}</span>{e(self.title)} {badge}</h2>
<p class='meta'>{e(self.chapter)}{(' · ' + e(self.source)) if self.source else ''}</p>
<blockquote><b>书上结论</b>：{e(self.claim)}</blockquote>
<p class='prop'><b>要判决的命题</b>：{e(self.proposition)}</p>
<table class='checks'><thead><tr><th></th><th>检查项</th><th>实测</th></tr></thead>
<tbody>{rows}</tbody></table>
{figs}{notes}</section>"""
