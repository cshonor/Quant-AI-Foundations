# 仓库书写约定：上下标与公式（易读版）

适用于本仓库内 **PSTAT_Basic、PSTAT_Quant、TS、RFA、ECON-CSPD、Alg** 等文件夹中的 Markdown 笔记（含中文文件名）。

## 为什么用 HTML 上下标

用 `X_n`、`sigma^2`、`x_{t-1}` 等**同级字符**拼公式时，在部分字体下**下标、上标与正文挤在同一基线**，容易与「普通下划线含义」混淆。

**推荐**：在 Markdown 正文中用 HTML：

- **下标**：`<sub>…</sub>`，例如 `X<sub>n</sub>` → **X<sub>n</sub>**
- **上标**：`<sup>…</sup>`，例如 `σ<sup>2</sup>` → **σ<sup>2</sup>**（也可用 `sigma<sup>2</sup>`）

在 VS Code、GitHub、常见笔记预览中，**上下标与正文层级区分清楚**。  
**这不是 LaTeX**：不写 `$...$` 块公式；`<sub>` / `<sup>` 为 HTML，预览普遍支持。

## 基本示例

| 含义 | 推荐写法 |
| :--- | :--- |
| 序列下标 | `X<sub>n</sub>`、`x<sub>t</sub>`、`w<sub>t</sub>` |
| 多字符下标 | `x<sub>t-1</sub>`、`I<sub>t-1</sub>`（勿与 `x_t-1` 混淆） |
| 方差 | `σ<sup>2</sup>` 或 `sigma<sup>2</sup>` |
| 极限 | `**lim**<sub>n→∞</sub>` |
| 复指数 | `e<sup>i t X</sup>` |
| 求和 | 优先 `X<sub>1</sub>+⋯+X<sub>n</sub>`，少写易混的 `∑` 上下界堆叠 |

## 嵌套与可读性

- **尽量避免**在 `<sup>` 里再套一层 `<sup>`；复杂指数拆句写。  
- **反引号代码** `` `like_this` `` 内的内容不会被自动脚本改写（避免破坏路径、命令）。

## 改版进度

- **PSTAT_Basic** 第 4 章极限定理：已按本约定手写改版。
- **TS / RFA / PSTAT 其余 / Alg**：可用仓库脚本 `tools/format_md_subsup.py` 做**保守替换**（见脚本说明）；之后仍建议人工扫一眼。**Alg** 中匹配到常见模式的 `.md` 已跑过一轮，其余文件随编辑增补。

## 一键批量（可选）

在项目根目录执行（需已安装 Python 3）：

```bash
python tools/format_md_subsup.py
```

默认处理 `PSTAT_Basic/`、`PSTAT_Quant/`、`TS/`、`RFA/`、`ECON-CSPD/`、`Alg/` 下全部 `.md`（跳过已含大量 `<sub>` 的文件以免重复加工）。
