# 「MA 交叉 + ADX 开关 + ATR 止损」回测代码 —— 逐条体检

> 环境：macOS arm64 / Python 3.13.12（隔离 venv）/ **vectorbt 1.1.1** / numpy 2.5.3 / pandas 3.0.6 / TA-Lib 0.8.1
> 数据：BTCUSDT 现货日线，Binance 公开镜像（`data-api.binance.vision`），2023-09-27 → 2026-09-26 共 1096 根
> Yahoo 出口在本机被限流/反爬（429），所以换成 Binance；两者报价口径略有差异，结论已按 BTCUSDT 重算。
> ⚠️ 代码仅用于学习与演示，不构成投资建议，严禁直接实盘。

---

## 0. 一句话结论

**这段代码在 vectorbt 1.1.1 上跑不起来**（`vbt.ADX` 不存在）。把它补成能跑之后，里面还有 **1 个致命 bug（ATR 止损一次都没触发）** 和 **1 处逻辑与注释自相矛盾的地方（平仓信号也被 ADX 过滤了，和文档里「过滤只拦截新开仓」的说法相反）**。这两条叠加，实测把 3 年总收益从 +13.5% 打成 **-42.5%**、最大回撤从 51.3% 放大到 **79.3%**。

| 贴子里的说法 | 实测判定 | 依据 |
|---|---|---|
| `vbt.ADX.run(...)` | ❌ 不成立 | `hasattr(vbt, "ADX") == False`，报 `AttributeError` |
| `sl_stop=atr.atr * 1.5` 是 1.5 倍 ATR 止损 | ❌ **完全没生效** | `sl_stop` 单位是百分比；同一信号「有止损」与「无止损」两种版本的资金流 `np.allclose == True` |
| 「ADX 过滤只拦截新开仓，老仓照常被反向平仓」 | ❌ **与代码相反** | `exits=entries_short` 里的 `entries_short` 本身就是 `raw_short & (ADX≥20)`，实测有 **8 次**持仓中的反手信号被吞 |
| 「交易次数会变少」 | ✅ 成立 | 22 笔 → 12 笔 |
| 「会错过一部分趋势启动」 | ✅ 成立 | 22 个交叉信号里 8 个（36.4%）被 ADX<20 吞掉 |
| 「夏普一般会变好，总收益可能下降」 | ❌ 本样本里没出现 | Sharpe 0.309（基线）→ 0.148（原样）/ -0.009（修正 intent）；收益同时大幅下降（见第 3 节）。样本仅十几笔，这条本来就不该当作稳定预期 |

---

## 1. Bug 清单（每条都给了源码证据 + 实测证据）

### B1（致命）`sl_stop` 是**百分比**，不是美元金额 → ATR 止损从未触发

源码证据 `vectorbt/portfolio/base.py:2230`：

```
sl_stop (array_like of float): Stop loss.
    A percentage below/above the acquisition price for long/short position.
    Note that 0.01 = 1%.
```

贴子里 BTC 日线的 `atr.atr * 1.5` 是几百到几千的**价格单位**，喂进去等于告诉 vectorbt「回撤 200000% 才止损」，于是止损价被算成一个巨大的负数，**永远不会成交，而且不报错、不告警**。

实测（构造数据：收盘恒为 100，第 2 根 `low=95`）：

```
sl_stop=5.0  -> 未触发（若按金额理解应该在 95 触发）   → 证明是百分比口径
sl_stop=0.03 -> 第 2 根触发
```

真实数据上的验证更直接：把 V1 的资金流和「同一套信号、完全不加止损」的版本逐根比对，

```
V1(ATR*1.5 当金额) 资金流 == 同信号无止损版本 ? True  => 止损完全没生效
```

**修法**：`sl_stop = atr_mult * atr / close`（除以价格换成百分比）。

### B2（致命）没传 `high` / `low` → 盘中插针打不到止损

源码证据 `vectorbt/portfolio/nb.py:2094-2095`：

```python
if np.isnan(_low):
    _low = min(_open, _close)
if np.isnan(_high):
    _high = max(_open, _close)
```

`high` / `low` 的默认值是 `np.nan`（`base.py:2805`）。贴子里花了力气算 `high` / `low` 但**没有传进 `from_signals`**，于是回测引擎用「开盘价和收盘价里较小的那个」当最低价——**真实的最低价直接被丢弃**。

实测（收盘恒为 100，第 2 根盘中 `low=90`，5% 止损）：

```
不传 high/low: 不触发（因为 low 被当成 min(open,close)=100）
传入 high/low: 第 2 根触发
```

**修法**：`Portfolio.from_signals(..., high=high, low=low)`。

### B3（逻辑 bug）`exits` 也过了 ADX 过滤 → 和文档里第 2 条自相矛盾

贴子里写：

```python
entries_long  = raw_long  & (adx.adx >= 20)
entries_short = raw_short & (adx.adx >= 20)
...
exits=entries_short          # ← 这里用的是【已过滤】的信号
short_exits=entries_long     # ← 同上
```

`exits` 用的是 `entries_short`，它已经被 `& (adx>=20)` 过滤过了。所以 ADX<20 那天的死叉，**既不会反手做空，也不会把多头平掉**——老仓被锁死。这与「重点：过滤只拦截新开仓，不会强行平掉已经在手的仓位」这句话描述的恰恰相反（那句话描述的代码应该是 `exits=raw_short`）。

实测（显式状态机逐 bar 回放，持仓中却被 ADX 拦掉的反手信号共 **8 次**）：

```
2024-02-11  持仓=-1  金叉->应反手做多  => 被丢弃, 仓位继续持有
2025-01-09  持仓=+1  死叉->应反手做空  => 被丢弃
2025-01-20  持仓=+1  金叉->应反手做多  => 被丢弃
2025-08-30  持仓=+1  死叉->应反手做空  => 被丢弃
2025-09-29  持仓=+1  金叉->应反手做多  => 被丢弃
2026-03-29  持仓=-1  金叉->应反手做多  => 被丢弃
2026-04-06  持仓=-1  死叉->应反手做空  => 被丢弃
2026-04-10  持仓=-1  金叉->应反手做多  => 被丢弃
```

后果很直观：**最长持仓从 127 天撑到 183 天**，平均持仓从 44 天涨到 81 天（因为它接替位接不上，只能干等）。

**修法**：`exits=raw_short` / `short_exits=raw_long`（平仓不过滤），或者如果你**真的**想要「ADX 掉下去就锁仓」的效果，那请把注释改成对的——这是个策略选择，但别让它藏在变量名里。

### B4 `vbt.ADX` 在 vectorbt 里根本不存在

`hasattr(vbt, "ADX") == False`，跑到那一行就是 `AttributeError`。vectorbt 自带的指标只有 `MA / MSTD / BBANDS / RSI / STOCH / MACD / ATR / OBV`（见 `vectorbt/indicators/__init__.py:10`）。

**修法**：装 `TA-Lib` 后桥接，注意参数名是 TA-Lib 的 **`timeperiod`**，不是 vectorbt 的 `window`（写错会报 `TypeError: ADX() got an unexpected keyword argument 'window'`）：

```python
ADX = vbt.IndicatorFactory.from_talib("ADX")
adx = ADX.run(high, low, close, timeperiod=14).real
```

### B5 同根 bar 同时出现入场/出场时，默认行为是「两个都丢掉」

`vectorbt/_settings.py:410-412` 的默认值是两个 `ignore`：

```
upon_long_conflict="ignore", upon_short_conflict="ignore", upon_dir_conflict="ignore"
```

`ConflictMode.Ignore` 的注释是「Ignore both signals」。这套信号虽然交叉不会同根撞车，但加了 ADX 之外的一堆条件（比如以后加 volume 过滤、加减仓）就会踩到。显式声明一下不吃亏：

```python
upon_long_conflict="exit", upon_short_conflict="exit"
```

### B6 `direction` 别依赖默认行为

默认 `direction = portfolio_cfg["signal_direction"] = "longonly"`（`_settings.py:414`）。在 1.1.1 里，只要传了 `short_entries`/`short_exits` 就会自动转双向并打一行 `UserWarning: direction has no effect if short_entries and short_exits are set`，但这属于版本相关的隐式行为。**显式写 `direction="both"`。**

### B7 同根收盘价算信号、又按同一根收盘价成交

信号用 Bar N 的收盘价算出来，订单也按 Bar N 的收盘价成交——现实中做不到。改成下一根开盘成交后，同一套逻辑的总收益从 **-11.3% 变成 +25.9%**（第 3 节 V4 vs V5）。

这个 37 个百分点的差**不是 alpha**，它只说明这套东西对撮合假设极度敏感：

| 成交假设 | 总收益% | Sharpe |
|---|---|---|
| 同根收盘（V4） | -11.3 | -0.009 |
| 下根开盘（V5） | +25.9 | 0.407 |

### B8 `vbt.ATR` 是**简单移动平均**口径，不是 Wilder 口径

`vbt.ATR` 内部用 `vbt.MA.run(tr, window=n)`，而 Wilder 的 ATR 用的是自己的平滑。实测两者平均相对偏差 **6.57%**。移植/复现时要写清楚跟哪个口径对齐，否则两边对不上账又查不出原因。

### B9（最该说的一条）样本量根本不够做结论

3 年日线、20/60 均线，**总共只有 22 个交叉信号**，ADX≥20 之后只剩 14 笔交易。在这个量级上：

* 一次成交时点的改动就能带来几十个百分点的差异（B7）；
* 参数网格上收益跳来跳去，没有任何单调结构（第 4 节）；
* 买入持有同期 **+220.1%**（回撤 53.0%、Sharpe 1.06），**所有变体都被按在地上摩擦**。

所以这份代码现在的正确用途是：**验证「逻辑是不是照你想的那样执行」**，而不是「这个策略好不好」。

---

## 2. 复现所需的最小改动（等价于 `strategy_fixed.py`）

```diff
- adx = vbt.ADX.run(high, low, close, window=ADX_WINDOW)
+ ADX_IND = vbt.IndicatorFactory.from_talib("ADX")     # vbt 没有 ADX
+ adx = ADX_IND.run(high, low, close, timeperiod=ADX_WINDOW).real

- entries_long  = raw_long  & (adx.adx >= ADX_THRESHOLD)
- entries_short = raw_short & (adx.adx >= ADX_THRESHOLD)
+ entries_long  = raw_long  & trend
+ entries_short = raw_short & trend
+ long_exit     = raw_short        # ← 平仓不再被 ADX 过滤（B3）
+ short_exit    = raw_long

  pf = vbt.Portfolio.from_signals(
      close,
      entries=entries_long, exits=long_exit,
      short_entries=entries_short, short_exits=short_exit,
-     sl_stop=atr.atr * ATR_SL_MULTIPLIER,
+     sl_stop=ATR_SL_MULTIPLIER * atr.atr / close,       # ← 百分比口径（B1）
+     high=high, low=low,                                # ← 止损要认盘中极值（B2）
+     direction="both",                                  # ← 别靠默认（B6）
+     upon_long_conflict="exit", upon_short_conflict="exit",   # ← 默认 ignore（B5）
      init_cash=INIT_CAPITAL, fees=0.001,
  )
```

> 可选：把信号整体 `.shift(1)` 并传 `price=open`，改成下根开盘成交（B7）。

---

## 3. 实测对比（BTCUSDT 日线，2023-09-27 → 2026-09-26，1096 根，10 万本金，费率 10bp）

| 变体 | 总收益% | CAGR% | 最大回撤% | Sharpe | Sortino | 交易数 | 胜率% | 平均持仓(天) | 最长持仓(天) | 敞口% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V0 基线：纯交叉反手 / 无 ADX / 无止损 | 13.5 | 4.3 | 51.3 | 0.309 | 0.459 | 22 | 38.1 | 44.2 | 127 | 88.8 |
| **V1 原样**（止损失效 + 平仓被过滤 + 未传 high/low） | **-42.5** | -16.9 | **79.3** | 0.148 | 0.218 | 12 | 45.5 | 81.0 | **183** | 88.8 |
| V2 只修止损单位 | -9.6 | -3.3 | 41.8 | 0.040 | 0.056 | 12 | 33.3 | 53.8 | 183 | 58.9 |
| V3 + 传 high/low（平仓仍过滤） | -26.0 | -9.6 | 43.3 | -0.200 | -0.278 | 14 | 28.6 | 41.2 | 183 | 52.6 |
| V4 作者意图（平仓不过滤） | -11.3 | -3.9 | 43.3 | -0.009 | -0.013 | 14 | 35.7 | 34.5 | 127 | 44.1 |
| V4b V4 去掉止损 | 23.8 | 7.4 | 53.1 | 0.377 | 0.564 | 14 | 46.2 | 52.7 | 127 | 67.4 |
| V5 V4 + 下根开盘成交 | 25.9 | 8.0 | 40.7 | 0.407 | 0.610 | 13 | 50.0 | 42.7 | 127 | 50.7 |
| **买入持有基准** | **220.1** | — | 53.0 | **1.06** | — | 1 | — | — | — | 100 |

读表要点：

1. **V1 vs V0** 就是两个致命 bug 的代价：-42.5% vs +13.5%，回撤 79.3% vs 51.3%。
2. **V4 vs V4b**：把 ATR 止损加上反而少了 35 个百分点。这不代表「止损没用」，而是 1.5×ATR 在 BTC 日线上太紧（V4 的 14 笔里有 **9 笔**是被止损打掉的）。
3. **所有主动策略都被买入持有吊打**。这段样本是单边上涨行情，一个多空反手系统天然吃亏。
4. V4 里 9/14 笔由止损离场 → 说明这套参数下「止损」才是主导变量，均线/ADX 反而是配角。

---

## 4. 参数敏感性：别指望「调 ADX 阈值」能救

同一份数据、同一套修正后的逻辑（`sweep.py`），铺开 ADX 阈值 × 均线组合：

**总收益 %**

| 均线 | ADX≥0 | ADX≥15 | ADX≥20 | ADX≥25 | ADX≥30 | ADX≥35 |
|---|---:|---:|---:|---:|---:|---:|
| 10/30 | -30.2 | -73.3 | -56.9 | -47.0 | -22.4 | -11.0 |
| 20/60 | -18.5 | -14.3 | -11.3 | 8.6 | 10.2 | 0.0 |
| 30/90 | -13.2 | -13.2 | -5.0 | -9.8 | -24.5 | -7.6 |
| 50/150 | 6.7 | 6.7 | -11.4 | -11.4 | -28.2 | -15.5 |

**Sharpe**

| 均线 | ADX≥0 | ADX≥15 | ADX≥20 | ADX≥25 | ADX≥30 | ADX≥35 |
|---|---:|---:|---:|---:|---:|---:|
| 10/30 | -0.14 | -1.39 | -1.15 | -1.57 | -0.93 | -0.51 |
| 20/60 | -0.04 | 0.01 | -0.01 | 0.23 | 0.28 | inf* |
| 30/90 | -0.06 | -0.06 | 0.05 | -0.07 | -0.55 | -1.02 |
| 50/150 | 0.21 | 0.21 | -0.14 | -0.14 | -0.88 | -0.76 |

**交易笔数**

| 均线 | ADX≥0 | ADX≥15 | ADX≥20 | ADX≥25 | ADX≥30 | ADX≥35 |
|---|---:|---:|---:|---:|---:|---:|
| 10/30 | 41 | 38 | 23 | 11 | 5 | 3 |
| 20/60 | 21 | 20 | 14 | 8 | 3 | 0 |
| 30/90 | 13 | 13 | 11 | 7 | 4 | 1 |
| 50/150 | 8 | 8 | 6 | 6 | 5 | 3 |

\* 20/60 + ADX≥35 那一格是 0 笔交易，Sharpe 出来 `inf` —— 顺手提醒：**网格里出现 0 笔交易时，指标全是无意义的，回测脚本要主动过滤掉这些格子。**

结论：网格上收益**不单调**、换均线组合就翻号，典型的小样本噪声。任何「ADX=22 最好」式结论都是过拟合。

---

## 5. 移植到 Go / C++ 的伪代码

指标口径已用 `adx_reference.py` 与 TA-Lib 逐根对齐（**ADX 最大绝对误差 3.6e-14，ATR 1.8e-12**），可以直接照着写。

```
# ---------- 状态（每个 symbol 一份，滚动更新，别重算全表）----------
state: prev_high, prev_low, prev_close
       sum_tr, sum_plus_dm, sum_minus_dm      # 注意：TA-Lib 用「和」，不是均值
       dx_ring[N]                             # 环形缓冲，存最近 N 个 DX
       adx
       ma_fast_buf[F], ma_slow_buf[S]
       position: FLAT | LONG | SHORT
       entry_price, stop_price

# ---------- 每根新 bar (o,h,l,c) 到达 ----------
TR  = max(h - l, abs(h - prev_close), abs(l - prev_close))
up  = h - prev_high
dn  = prev_low - l
plus_dm  = (up > dn && up > 0) ? up : 0
minus_dm = (dn > up && dn > 0) ? dn : 0

# Wilder 平滑（TA-Lib 口径，保持"和"量纲）
#   预热期: sum += x                      前 N-1 根
#   之后  : sum  = sum - sum/N + x        每根一次
if warmup < N-1:
    sum_tr += TR; sum_plus_dm += plus_dm; sum_minus_dm += minus_dm; warmup++
else:
    sum_tr      = sum_tr      - sum_tr/N      + TR
    sum_plus_dm = sum_plus_dm - sum_plus_dm/N + plus_dm
    sum_minus   = sum_minus   - sum_minus/N   + minus_dm

    if sum_tr > 0:
        plus_di  = 100 * sum_plus_dm / sum_tr
        minus_di = 100 * sum_minus   / sum_tr
        if (plus_di + minus_di) != 0:
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            push dx_ring (dx)                      # 存够 N 个才有第一个 ADX
            if dx_ring full and adx 未初始化:
                adx = mean(dx_ring)                # = sumDX / N
            else if adx 已初始化:
                adx = (adx * (N-1) + dx) / N      # 乘 N-1 再除 N，别写 adx - adx/N + dx

# ATR（价格单位，注意和上面"和"量纲差一个 N）
atr = sum_tr / N                                   # 前提：sum_tr 是 Wilder 平滑后的值
#   若你从零起算：第一个 ATR = mean(TR[1..N])，之后 atr = (atr*(N-1) + TR) / N
#   ⚠️ 别用 SMA(TR, N)：那和 Wilder 口径平均差 6.6%

# ---------- 信号（用已收盘的 bar N，订单在 bar N+1 开盘成交）----------
cross_up = ma_fast > ma_slow  && prev_ma_fast <= prev_ma_slow
cross_dn = ma_fast < ma_slow  && prev_ma_fast >= prev_ma_slow
trend_on = adx >= ADX_THRESHOLD

long_entry  = cross_up && trend_on        # ADX 只拦【开仓】
short_entry = cross_dn && trend_on
long_exit   = cross_dn                    # ← 平仓不拦（原贴这里写成 cross_dn && trend_on，是 bug）
short_exit  = cross_up

# ---------- 撮合与风控（bar N+1 开盘执行 bar N 的信号）----------
if position == LONG and (long_exit or open_price <= stop_price): close_all()
if position == SHORT and (short_exit or open_price >= stop_price): close_all()
if position == FLAT and long_entry:  buy_all();  stop_price = entry_price * (1 - K*atr/entry_price)
if position == FLAT and short_entry: sell_all(); stop_price = entry_price * (1 + K*atr/entry_price)
# 同一根 bar 上 close 与 open 的优先级要写死：
#   先平仓再开新仓（= upon_long_conflict = "exit"），别让两个信号互相抵消(vectorbt 默认是都丢掉)

prev_high, prev_low, prev_close = h, l, c
```

两个容易踩的点写进注释了：

1. **ADX 递推是 `(adx*(N-1) + dx)/N`**。只有在你把状态存成「未归一化的和」时才可以写成 `adx - adx/N + dx`；混着写会在预热期之后留下一个缓慢衰减的偏差（我在对齐 TA-Lib 时就踩了这个，偏差最大 0.21 个 ADX 点，要 200 多根才消掉）。
2. **DX 的前 N 个值要攒起来取平均才有第一个 ADX**，位置是第 `2N-1` 根（N=14 时第 27 根）。

---

## 6. 文件说明

| 文件 | 作用 |
|---|---|
| `fetch_data.py` | 抓 BTCUSDT 日线（翻页，避开 Yahoo 限流） |
| `adx_reference.py` | 纯 numpy 手写 Wilder ADX/ATR，与 TA-Lib 对齐到 1e-12；顺便量了 `vbt.ATR`（SMA 口径）与 Wilder 口径的偏差 |
| `audit_run.py` | V0–V5 七个变体的对比，含「止损到底触发没」「被吞掉的反手信号」的显式核查 |
| `strategy_fixed.py` | 修复后的可执行版本，参数集中在 `CONFIG` |
| `sweep.py` | ADX 阈值 × 均线组合敏感性网格 |
| `equity_plot.py` → `equity.html` | 四条净值曲线对比图 |
| `results.csv` / `results.md` | 各变体指标 |

运行（第一次需要先建 venv）：

```bash
python -m venv .venv && .venv/bin/pip install vectorbt TA-Lib plotly
python fetch_data.py
python adx_reference.py     # 指标对齐自检
python audit_run.py         # 变体对比
python sweep.py             # 参数敏感性
python strategy_fixed.py
python equity_plot.py
```

> ⚠️ 本机实测：跑这些脚本需要关沙箱或设 `PYTHONDONTWRITEBYTECODE=1`，否则 byte-code 临时目录清理会被拦住导致进程中断。

---

## 7. 还能继续往下做的点

1. **先修口径再谈收益**：B1/B2/B3 三条没修之前，任何参数调优都是在调 bug。
2. **样本量**：20/60 日线 3 年只有 22 个交叉。要么换成 4h/1h，要么做多品种 / walked-forward，否则统计上没意义。
3. **把 ADX 从「开关」改成「仓位缩放」**：你自己在迭代点 2 里提了，20–25 弱趋势减半仓，比 0/1 开关稳健。
4. **止损要做成本/滑点敏感性**：本样本里 1.5×ATR 止损吃掉了 35 个点收益，参数敏感度必须先量化。
5. **别忘了对基准**：这段行情买入持有 +220%，策略赚不赚钱要跟它比，不能只看绝对收益。
