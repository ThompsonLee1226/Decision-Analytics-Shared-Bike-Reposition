# Handoff: 06:30 GP 库存的 OD-only 估计

**状态：** 设计文档，尚未实现  
**日期：** 2026-11-09  
**范围：** `xgb_bikeflow/data/hw6_grid_ownership.csv` 中的 90 个 HW6 gathering point（GP）；核心区域 = 该文件的全部 10,473 个 L8 格点（含 4,843 个 `UNASSIGNED`）  
**硬约束：** 原始数据只有订单的起点/终点/时间，没有可靠的、跨天持续的车辆编号 `bike_id`

> 本文替代之前“用 bike_id 找当天首程起点”的思路。那个方法在本数据条件下不可用，原因见 §9.3。  
> 本文只使用 OD 计数、GP 归属、时间窗和总量约束。

---

## 0. TL;DR

只用 OD 计数，**无法唯一识别 06:30 每个 GP 的库存“水平”**，只能识别库存随时间的“变化”。要得到可用的 `I_i(06:30)`，必须补上一个水平约束（总车数 `N`、某个时刻的库存、或稳态/先验假设）。

推荐的三件套：

1. **净流下界 `m_i`**：只用 OD 就能算出的最小可行库存；
2. **晚间到达先验 `w_i`**：只用 OD 估计 06:30 的空间分布；
3. **投影**：在 `I_i >= m_i`、`I_i <= q_i`、`sum(I_i) = N` 下，取最接近 `N * w_i` 的解。

`N` 未知时：先用 `N_min = sum(m_i)`，再用早高峰仿真校准 `N`，并报告 `N` 的可行区间。

不要直接使用 `π = πP` 求稳态再乘 `N`；它给出的不是库存。详见 §8。

---

## 1. 数据与口径

### 1.1 输入文件

| 文件 | 用途 |
|---|---|
| `../xgb_bikeflow/data/hw6_grid_ownership.csv` | L8 格点 → GP 归属；**全部 10,473 行都属于核心区域**，`UNASSIGNED` 也是核心的一部分 |
| `../Dataset/dataset_train_20260921_lsz.csv` | 原始订单：`start_time/end_time/start_geohash/end_geohash` |
| `../xgb_bikeflow/data/hw6_gp_master.csv` | GP 属性；坐标用 `lon_gcj/lat_gcj`（GCJ-02） |
| `../Problem_Description.md` | MDP 的库存、容量、调度定义 |

### 1.2 空间口径

- 核心区域 = `hw6_grid_ownership.csv` 的 **全部 10,473 个 L8 格点**。
- 其中 **5,630 个格点分配给 90 个 GP**，**4,843 个是 `UNASSIGNED`**。
- GP 库存 `I_i` 只统计分配给 GP `i` 的格点；`UNASSIGNED` 格点里的车是另一个池子，是否进入 MDP state 需要单独决定（见 §10.2）。
- 坐标基准是 **GCJ-02**，不要转换；这和地图/geohash 无关，但涉及任何空间连接时要注意。

### 1.3 时间口径

- 早高峰窗口：`[06:30, 10:30)`，8 个 30 分钟 slot。
- slot `t = 0` 是 06:30–07:00，`t = 7` 是 10:00–10:30；10:30 整不属于窗口。
- 工作日：周一至周五，剔除八个法定节假日，并计入 2026-05-09（周六调休）：
  - 节假日：2026-04-04 至 04-06，2026-05-01 至 05-05；
  - 调休：2026-05-09。
- 数据集完整工作日为 2026-03-02 至 2026-05-22，共 **57 个工作日**。
- **2026-05-22 是半天数据**（早高峰只有约 600 单，正常约 2,300–2,400 单），做日均时必须排除或单独标记。

### 1.4 流的口径

- 到达 `A_i(t)`：`end_time` 落在 slot `t`、`end_geohash` 属于 GP `i` 的订单数。
- 离开 `D_i(t)`：`start_time` 落在 slot `t`、`start_geohash` 属于 GP `i` 的订单数。

注意：**GP 的到达/离开计数包含核心区域内部的行程**。一个从 GP `i` 骑到 GP `j` 的订单，会同时计入 `i` 的离开和 `j` 的到达；它不跨越核心边界，但仍然改变 GP 库存。只有两端都不在 GP 格点（都在 `UNASSIGNED` 或都在核心外）的订单，才不影响任何 GP 的 `A_i/D_i`。这与 §9.2 的“跨界流”口径不同，不要混用。

---

## 2. 符号与库存动力学

对 GP `i`、slot `t = 0..T-1`：

| 符号 | 含义 |
|---|---|
| `A_i(t)` | slot `t` 到达 GP `i` 的自行车数 |
| `D_i(t)` | slot `t` 从 GP `i` 离开的自行车数 |
| `I_i(t)` | slot `t` 开始时 GP `i` 的库存（06:30 为 `t=0`） |
| `R_i(t)` | slot `t` 的净调度补车（正=补入，负=搬出） |
| `S_i(t)` | slot `t` 因无车而未满足的离开需求 |
| `q_i` | GP `i` 的容量 |
| `N` | 06:30 核心区域总车数（GP + UNASSIGNED + 在途） |

库存递推：

```text
I_i(t+1) = I_i(t) + A_i(t) - D_i(t) + R_i(t) - S_i(t)
0 <= I_i(t) <= q_i
```

忽略调度和缺车时：

```text
I_i(t) = I_i(0) + cumA_i(t) - cumD_i(t)
```

**关键事实：OD 只能给出 `cumA - cumD`，给不出 `I_i(0)` 的水平。** 对每个 GP 只有一个方程、两个未知数（`I_i(0)` 和 `I_i(T)`）；即使把所有 GP 加起来，也只多一个总量约束，空间分布仍不唯一。

所以必须补一个假设。本文补的是：**总车数 `N` + 空间先验 `w_i` + 非负/容量约束**。

---

## 3. 方法一：净流下界 `m_i`

### 3.1 推导

假设：

- 早高峰没有调度（`R_i = 0`）；
- 所有观测到的离开都被满足（没有 `S_i`，即数据没有因为缺车而截断）；
- 库存不允许为负。

那么对所有 `s`：

```text
I_i(s) = I_i(0) + cumA_i(s) - cumD_i(s) >= 0
```

因此：

```text
I_i(0) >= max_s [ cumD_i(s) - cumA_i(s) ]
```

定义下界：

```text
m_i = max_{0 <= s <= T} [ sum_{t=0..s} D_i(t) - sum_{t=0..s-1} A_i(t) ]^+
```

这是**保守版**：假设同一 slot 的到达都晚于离开，到达只能从下一个 slot 开始使用。  
如果认定同一 slot 的到达可以服务同一 slot 的离开，把 `A` 的上限改成 `t=0..s`，得到的 `m_i` 会略小。差异通常很小，但要保持一致。

### 3.2 含义

- `m_i` 是“早高峰不出现负库存”的**最小初始库存**；
- `sum_i m_i` 是“不出现缺车”的**最小 GP 总车数**；
- 真实 `I_i(0)` 一定 `>= m_i`（在没有缺车的前提下）；
- 如果数据中存在缺车，观测到的 `D_i` 是截断后的需求，真实需求更大，`m_i` 只是下界。

### 3.3 参考代码

```python
# D, A: DataFrame, index = gp, columns = slot 0..7
cumD = D.cumsum(axis=1)
cumA = A.cumsum(axis=1).shift(axis=1, fill_value=0)   # 保守版
m = (cumD - cumA).max(axis=1).clip(lower=0)
```

### 3.4 经验数值（观测 OD，非模型预测）

- 完整工作日上，90 个 GP 的 `sum_i m_i` 约为 **1,050–1,250 辆**；
- 这是“早高峰所有观测离开都被满足”的最小总库存；
- 它不是实际库存，只是可行下界；实际库存取决于总车数和调度。

### 3.5 注意事项

- 如果 `m_i > q_i`，该 GP 即使 06:30 装满也不够，早高峰必须调度补车；这是“硬缺口”，应优先标记。
- 如果 `sum_i m_i > N`，总车数不足，必然有 GP 缺车或需要调度；问题不可行。
- 如果存在调度 `R_i`，下界应改为 `max_s [cumD - cumA - cumR]`；`R_i` 未知时，`m_i` 是无调度假设下的最保守下界。
- 这里只统计分配给 GP 的格点；`UNASSIGNED` 的库存不计入 `m_i`，但会消耗核心总车数 `N`。

---

## 4. 方法二：晚间到达先验 `w_i`

### 4.1 思路

06:30 的车绝大多数是前一天晚上停进来的。用前一个晚间（建议 18:00–24:00）的到达/离开构造空间权重：

**净积累版（推荐）：**

```text
raw_i = max(0, A_i(prev evening) - D_i(prev evening))
w_i   = raw_i / sum_j raw_j
```

**到达版（更稳健，防止净流噪声）：**

```text
raw_i = A_i(prev evening)
w_i   = raw_i / sum_j raw_j
```

如果两者差异大，用加权组合，并保留一个小 floor（例如 1% 均匀分布）避免某些 GP 权重为 0。

### 4.2 日期匹配

- 普通工作日：用前一个工作日晚间；
- 周一：不要直接用周日晚间（周末节律不同）；用上一个周五晚间，或最近 5 个工作日晚间的平均；
- 节假日后：用上一个同类型日（工作日/节假日）的晚间；
- 如果数据允许，最好对最近 3–5 个同类型晚间取平均，降低单日噪声。

### 4.3 经验验证（只用 OD，不用 bike_id）

以 2026-05-20 为例，用 2026-05-19 晚间 18:00–24:00 的流，对照 2026-05-20 早高峰离开量（90 个 GP）：

| 先验 | Pearson | Spearman |
|---|---:|---:|
| 晚间到达 `A(prev evening)` | **0.82** | **0.74** |
| 晚间净到达 `A - D` | **0.87** | — |

这说明晚间到达/净到达是有效的**空间先验**，但不是库存本身，不能直接当 `I_i(06:30)`。

### 4.4 参考代码

```python
prev = d - pd.Timedelta(days=1)
A_ev = (trips[(trips.end_time >= prev + pd.Timedelta(hours=18)) &
              (trips.end_time <  d)]
        .groupby("end_gp").size().reindex(gps, fill_value=0))
D_ev = (trips[(trips.start_time >= prev + pd.Timedelta(hours=18)) &
              (trips.start_time <  d)]
        .groupby("start_gp").size().reindex(gps, fill_value=0))

w = (A_ev - D_ev).clip(lower=0)
if w.sum() == 0:
    w = A_ev
w = w / w.sum()
```

### 4.5 注意事项

- 晚间窗口可测试 17:00–24:00、18:00–06:30；不同窗口的权重会变，做敏感性分析。
- 周末/节假日/工作日的晚间节律不同，必须按日型匹配。
- 晚间到达多不等于停了很多车（同一辆车可能多次出行）；它只是空间相关信号，所以必须和总量 `N`、下界 `m_i` 一起用。

---

## 5. 方法三：投影（先验 → 可行库存）

### 5.1 优化问题

给定：

- 先验权重 `w_i`（`sum_i w_i = 1`）；
- 总 GP 库存 `N_GP`；
- 下界 `m_i`；
- 容量 `q_i`（没有就设为 `inf`）。

求解：

```text
minimize   sum_i (I_i - N_GP * w_i)^2 / (2 * N_GP * w_i)
subject to m_i <= I_i <= q_i
           sum_i I_i = N_GP
```

这是一个可分离凸二次规划。先验说“应该在 `N_GP * w_i`”，下界/容量/总量把它拉到可行域内。

### 5.2 可行性

- `sum_i m_i <= N_GP <= sum_i q_i` 必须成立；
- 如果 `N_GP < sum_i m_i`：没有“无缺车”的可行解，必须允许缺车或调度；
- 如果存在 `m_i > q_i`：该 GP 硬缺口，单独标记；
- 如果 `w_i = 0`，用一个很小的 `epsilon` 替代，或把该 GP 固定在下界。

### 5.3 参考代码

```python
import numpy as np
from scipy.optimize import minimize

def project(prior, m, q, N):
    prior = np.asarray(prior, dtype=float)
    prior = np.maximum(prior, 1e-9)
    prior = prior / prior.sum()
    m = np.asarray(m, dtype=float)
    q = np.asarray(q, dtype=float)
    x0 = np.clip(N * prior, m, q)

    def obj(x):
        return np.sum((x - N * prior) ** 2 / (2 * N * prior))

    res = minimize(
        obj, x0, method="SLSQP",
        bounds=list(zip(m, q)),
        constraints=[{"type": "eq", "fun": lambda x: x.sum() - N}],
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not res.success:
        raise RuntimeError(res.message)
    return res.x
```

### 5.4 替代目标

- 最小化 `sum |I_i - N w_i|`（LP，更鲁棒）；
- 最小化 KL 散度 `sum I_i log(I_i / (N w_i))`（更适合比例数据）；
- 加一个平滑项，避免少数 GP 被投影到边界后其余 GP 剧烈变化。

---

## 6. 总车数 `N` 怎么定

### 6.1 有车队记录

直接用核心区 06:30 的可用车总数。注意 `N` 要区分：

- `N_GP`：停在 90 个 GP 的格点里；
- `N_unassigned`：停在 `UNASSIGNED` 格点里；
- `N_transit`：06:30 正在骑行、尚未到达的车；
- `N = N_GP + N_unassigned + N_transit`。

### 6.2 没有记录：先用下界

```text
N_min = sum_i m_i
```

以 `N = N_min` 投影，得到“刚好不出现缺车”的最小库存向量。这是基线，不是实际值。

### 6.3 仿真校准（推荐）

把 `N` 当作标量参数：

1. 给定 `N`，投影得到 `I_i(06:30)`；
2. 用 `xgb_bikeflow` 的 slot 级预测生成 `A_i(t), D_i(t)`；
3. 跑库存递推，统计缺车/未满足离开；
4. 调整 `N`，使模拟的“已服务离开量/缺车率”与观测 OD 最接近；
5. 可用一维求根/网格搜索；随机模拟时用多次重复的矩匹配。

这本质上是**模拟矩估计 / 间接推断**：OD 本身不给 `N`，但“模拟出的早高峰服务量”可以反推 `N`。

### 6.4 报告区间

如果既没有车队记录，也没有缺车观测，只能报告：

```text
N_min = sum_i m_i
N_upper = 车队记录 或 其他外部信息
# 若用“晚间到达总量”当粗上界：它假设每单都是不同车辆，
# 同一辆车多次出行时会高估，只能作为非常粗的参考
I_i(06:30; N) for N in [N_min, N_upper]
```

不要在没有 `N` 的情况下给单一 `I_i(06:30)`。

---

## 7. 端到端流程

1. 读 `hw6_grid_ownership.csv`，建立 `geohash -> gp`，固定 90 个 GP；
2. 读订单，解析 `start_time/end_time`，过滤工作日和 06:30–10:30；
3. 把订单聚合到 `(gp, slot)`：得到 `D_i(t)` 和 `A_i(t)`；
4. 计算 `m_i`（§3）；
5. 计算前一日晚间的 `w_i`（§4），必要时对最近几个同类型晚间平均；
6. 选 `N`（§6），做投影（§5）得到 `I_i(06:30)`；
7. 用 `xgb_bikeflow` + 库存递推模拟早高峰，检查：
   - `sum I_i = N_GP`；
   - `I_i >= m_i`；
   - `I_i <= q_i`；
   - 模拟缺车与观测服务量是否一致；
8. 输出诊断表：`date, gp, m_i, w_i, N, I_i, q_i, flag`；
9. 做敏感性分析：`w_i` 用净/总晚间到达、窗口 18–24 vs 17–06:30、`N` 取 `N_min` 与外部估计。

---

## 8. 为什么不用静态 Markov 稳态 / 闭环网络直接求库存

### 8.1 `π = πP` 不是库存

由 OD 构造 `P_ij = T_ij / sum_j T_ij`，解 `π = πP` 得到的是**长期访问频率**：一次行程落在 `i` 的比例。它不是停在 `i` 的自行车数。要变成库存，至少需要每个 GP 的平均停留时间 `h_i`：

```text
I_i ∝ π_i * h_i
```

没有 `h_i`，`N * π_i` 只是“假设每辆车在每个 GP 停留时间相同”的产物，没有依据。

### 8.2 早高峰是瞬态

早高峰的 OD 矩阵和晚高峰反向。用早高峰的 `P` 求时间齐次稳态，相当于假设早高峰方向永远重复；它不描述 06:30 的状态。06:30 是日周期系统的一个相位，不是时间齐次链的稳态。

### 8.3 网络不是封闭的

调度车、维修、新增/退出、核心区域外的车都会破坏“车辆数守恒”。闭环网络的乘积形式解还要求时齐、无阻塞、有限容量等条件；你的 MDP 里 `Δd` 是外生流，和“顾客在站点间按路由移动”的闭环模型不是同一个东西。

### 8.4 流量平衡本身欠定

如果强行令 `A_i = D_i`（稳态平衡），等于假设早高峰没有净流入——而净流入正是你要研究的现象。不要用稳态平衡去抹掉它。

### 8.5 如果一定要用稳态

正确的版本是**日周期不动点**：

```text
I(06:30)_{d+1} = F(I(06:30)_d)
I* = F(I*)
```

`F` 是“用全天 OD + 调度策略模拟一天”的映射。它需要：

- 全天 OD（不只早高峰）；
- 调度策略 `R_i(t)`；
- 总车数 `N`；
- 容量 `q_i`。

没有这些，稳态法只是把未知数换了个名字。

**如果只想借用闭环网络：** 把 `π_i`（或 `π_i h_i`）只当作先验 `w_i`，然后仍然用 `N` 和 `m_i` 做投影；不要把 `N * π_i` 直接当库存。

---

## 9. 数据事实与探索性结果

### 9.1 核心区域规模

| 项 | 数值 |
|---|---:|
| L8 格点总数 | 10,473 |
| 分配给 90 个 GP | 5,630 |
| `UNASSIGNED` | 4,843 |
| GP 数 | 90 |
| 完整工作日数 | 57（2026-03-02 至 2026-05-22） |

### 9.2 早高峰边界流（背景，不是库存）

按“骑出/骑入核心边界”计，57 个工作日 06:30–10:30：

- 骑出 15,279 单；
- 骑入 18,324 单；
- 骑入/骑出 = 1.199；
- 净流入约 53 单/工作日。

这说明早高峰核心区是净流入的；也说明不能假设 `A_i = D_i`。

### 9.3 为什么不能用 `bike_id` / 首程法

即使某个训练文件里有 `bike_id`，在本数据上也**不能**当物理车辆编号使用：

- 177,624 个 `bike_id`，43% 只出现一天；
- 相邻两天的 bike_id Jaccard 约 **0.14**；
- 2026-05-20 有 6,625 辆活跃车，但首程发生在 06:30 前的只有 **163 辆（约 2.5%）**；
- 因此“06:30 前最后一单的终点”无法重建 06:30 位置；“当天首程起点”也高度依赖后续是否被调度搬动。

**结论：不要用首程法；本文的方法完全不依赖 `bike_id`。**

### 9.4 `m_i` 的量级

用完整工作日的观测 OD 计算：

- `sum_i m_i` 约 **1,050–1,250 辆**；
- 这是“早高峰不出现缺车”的最小 GP 总库存；
- 实际库存取决于总车数 `N` 和调度。

### 9.5 晚间先验的相关性

2026-05-19 晚间 vs 2026-05-20 早高峰（90 个 GP）：

- 晚间到达 vs 早高峰离开：Pearson **0.82**，Spearman **0.74**；
- 晚间净到达 vs 早高峰离开：Pearson **0.87**。

### 9.6 在途车辆

06:30 正在骑行（`start_time < 06:30 < end_time`）的车，完整工作日约 **10–22 辆/天**，可以直接从 OD 数出来；它们属于 arc inventory，不属于 GP node inventory。初始化时不要把它们算进 GP 库存，除非你的 MDP 把在途并到节点。

### 9.7 半天数据

2026-05-22 的早高峰只有约 600 单（正常约 2,300–2,400），是半天数据。做工作日平均/校准时应排除或单独标记。

---

## 10. 实现计划与待定决策

### 10.1 建议新增脚本

`estimate_0630_inventory.py`：

1. 读 ownership + orders；
2. 工作日过滤；
3. 聚合 `(gp, slot)`；
4. 逐日计算 `m_i`；
5. 逐日计算 `w_i`（日型匹配）；
6. 输入 `N`（或 `N_min`），投影；
7. 输出 `out/workday_gp_i0.csv`；
8. 输出诊断：`m_i`、`w_i`、`I_i`、`flag`（`m_i > q_i`、`N < sum m_i` 等）。

建议输出列：

```text
date, gp, m_i, w_i, N_GP, I_i, q_i, flag
```

### 10.2 待定决策

| 决策 | 选项 | 影响 |
|---|---|---|
| GP 数量 | 90（ownership）还是 96（`network_v1`） | `network_v1` 多 6 个 repro-only GP；需用 `network_v1_gp_cells.json` 重新聚合，不能直接用 ownership 的 90 个 |
| `UNASSIGNED` 库存 | 忽略 / 单独池 / 并入最近 GP | 影响 `N_GP` 与总车数守恒；建议先单独池并报告 |
| 容量 `q_i` | 有记录 / 无记录设 `inf` | 影响可行性；没有容量时只能保证非负 |
| 总车数 `N` | 车队记录 / `N_min` / 仿真校准 | 决定库存水平；没有 `N` 时只能给区间 |
| 日型匹配 | 前一晚 / 最近 3–5 个同型晚间 | 影响 `w_i`，尤其周一和节假日后 |
| 流来源 | 观测 OD / `xgb_bikeflow` 预测 | 观测用于历史校准；预测用于未来初始化 |

### 10.3 验证清单

- [ ] `sum_i I_i = N_GP`；
- [ ] `I_i >= m_i`；
- [ ] `I_i <= q_i`；
- [ ] `sum_i m_i <= N_GP`，否则标记为不可行；
- [ ] 模拟早高峰，缺车率与观测服务量一致；
- [ ] 对 `w_i`（净/总）、时间窗、`N` 做敏感性分析；
- [ ] 排除 2026-05-22 半天数据后重算日均；
- [ ] 明确 `UNASSIGNED` 和在途车的处理。

---

## 附录 A：参考代码骨架

```python
import numpy as np
import pandas as pd
from scipy.optimize import minimize

OWN = pd.read_csv("../xgb_bikeflow/data/hw6_grid_ownership.csv")
GH2GP = dict(zip(OWN.geohash, OWN.gp))
GPS = sorted(set(OWN.gp) - {"UNASSIGNED"})
HOL = pd.to_datetime([
    "2026-04-04", "2026-04-05", "2026-04-06",
    "2026-05-01", "2026-05-02", "2026-05-03", "2026-05-04", "2026-05-05",
])
MK = pd.to_datetime(["2026-05-09"])


def workday(d):
    return ((d.dt.dayofweek < 5) & ~d.isin(HOL)) | d.isin(MK)


def gp_slot_counts(trips, d, gps):
    """Return D, A: index=gp, columns=slot 0..7."""
    cut = d + pd.Timedelta(hours=6, minutes=30)
    end = cut + pd.Timedelta(hours=4)
    dep = trips[(trips.start_time >= cut) & (trips.start_time < end)].copy()
    arr = trips[(trips.end_time >= cut) & (trips.end_time < end)].copy()
    dep["gp"] = dep.start_geohash.map(GH2GP)
    arr["gp"] = arr.end_geohash.map(GH2GP)
    dep = dep[dep.gp.isin(gps)]
    arr = arr[arr.gp.isin(gps)]
    dep["slot"] = ((dep.start_time.dt.hour * 60 + dep.start_time.dt.minute - 390) // 30).astype(int)
    arr["slot"] = ((arr.end_time.dt.hour * 60 + arr.end_time.dt.minute - 390) // 30).astype(int)
    D = dep.groupby(["gp", "slot"]).size().unstack(fill_value=0).reindex(index=gps, columns=range(8), fill_value=0)
    A = arr.groupby(["gp", "slot"]).size().unstack(fill_value=0).reindex(index=gps, columns=range(8), fill_value=0)
    return D, A


def lower_bounds(D, A):
    """m_i: minimum initial inventory to avoid negative stock (no rebalancing)."""
    cumD = D.cumsum(axis=1)
    cumA = A.cumsum(axis=1).shift(axis=1, fill_value=0)
    return (cumD - cumA).max(axis=1).clip(lower=0)


def evening_prior(trips, d, gps):
    """w_i: previous evening's net arrivals, normalised."""
    prev = d - pd.Timedelta(days=1)
    lo = prev + pd.Timedelta(hours=18)
    A = trips[(trips.end_time >= lo) & (trips.end_time < d)].copy()
    D = trips[(trips.start_time >= lo) & (trips.start_time < d)].copy()
    A["gp"] = A.end_geohash.map(GH2GP)
    D["gp"] = D.start_geohash.map(GH2GP)
    A = A[A.gp.isin(gps)].groupby("gp").size().reindex(gps, fill_value=0)
    D = D[D.gp.isin(gps)].groupby("gp").size().reindex(gps, fill_value=0)
    w = (A - D).clip(lower=0)
    if w.sum() == 0:
        w = A
    return w / w.sum()


def project(w, m, q, N):
    w = np.asarray(w, float)
    w = np.maximum(w, 1e-9)
    w = w / w.sum()
    m = np.asarray(m, float)
    q = np.asarray(q, float)
    x0 = np.clip(N * w, m, q)
    res = minimize(
        lambda x: np.sum((x - N * w) ** 2 / (2 * N * w)),
        x0, method="SLSQP",
        bounds=list(zip(m, q)),
        constraints=[{"type": "eq", "fun": lambda x: x.sum() - N}],
    )
    if not res.success:
        raise RuntimeError(res.message)
    return res.x
```

---

## 附录 B：文件地图

| 文件 | 内容 |
|---|---|
| `../xgb_bikeflow/data/hw6_grid_ownership.csv` | L8 格点 → GP；10,473 行，90 个 GP + `UNASSIGNED` |
| `../xgb_bikeflow/data/hw6_gp_master.csv` | GP 属性；坐标用 `lon_gcj/lat_gcj` |
| `../Dataset/dataset_train_20260921_lsz.csv` | 原始 OD 订单 |
| `../Problem_Description.md` | MDP 的 node/arc inventory、容量、调度定义 |
| `flow_balance.py` | 工作日早高峰边界流平衡（out/in 1.199） |
| `gp_amap_map.py` | GP/格点高德地图 |
| `out/workday_morning_boundary_flow.csv` | 57 个工作日的边界流 |
| 本文 | 06:30 库存估计的 handoff |

---

## 附录 C：一句话交接

**不要用静态 Markov 稳态直接给库存；用 `m_i` 保证可行、用晚间到达 `w_i` 给空间先验、用 `N` 和投影定水平，再用早高峰仿真校准 `N`。没有 `N` 就报告区间，不要报告单一精确值。**
