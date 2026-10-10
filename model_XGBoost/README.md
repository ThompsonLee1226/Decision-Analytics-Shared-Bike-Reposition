# XGBoost 需求模型 — 说明

对应作业 §2（第二套方法）。模型本身很薄，值得读的是**为什么是这些数字**，
所以这份文档把「数据口径的一次更正」「修掉的一个 bug」「结果怎么读」「产物怎么用」
「哪里不能信」都写清楚。

---

## 0. 数据口径更正：先读这一节

**本目录下所有数字都基于 2026-10-09 更正后的数据集，与更早的版本不可直接相比。**

### 原来错在哪

清洗策略写在 `EDA/01_data_cleaning.py` 的 `CLEAN_DROP_FLAGS` 里，它当时把两个 flag
也划进了"丢弃"：

| flag | 丢掉了什么 |
|---|---|
| `qc_anomaly_day` | **整个 `2026-05-17`**：2,458 次出发，占其 7 日滚动中位数（8,267）的 30%，低于 50% 阈值。一天被整段删掉 |
| `qc_zero_dur` | 5,834 条 `duration_min == 0` 的行程 |

两个都不该丢：

- **异常日仍然是系统在运营的一天。** 需求低可能是因为下雨、假日、或调度中断 ——
  这些恰恰是模型该学会的东西。删掉它等于从面板里凭空抽掉一个日历日。
- **零时长行程是真的把车骑走了。** 其中 77.6% 的起终点跨了 L6 格 —— 在 ~38 m 的
  8 位 geohash 精度下不可能，说明原因是**分钟级时间戳截断**把亚分钟骑行压成了 0，
  不是用户错误。它们是真实的出发。

### 代价

8,271 行（占 1.24% 的行程），其中一整天的数据。面板因此从 76 天变成 77 天。

### 改法与验证

把这两个 flag 从 `CLEAN_DROP_FLAGS` 移进 `CLEAN_KEEP_FLAGS`（它们现在是"标记但不删"），
然后重跑。**验证方式**是逐行比对重跑产物与外部提供的那份
`trips_clean_keepzerodur_keepanomday.csv.gz`：

```
重生成 : 666,575 行  md5(内容) = ee88a7a7ac3b0ea064695694a15dcfe1
外部件 : 666,575 行  md5(内容) = ee88a7a7ac3b0ea064695694a15dcfe1
```

指纹取自按 `order_id` 排序后的 17 个关键列（含全部相关 `qc_*`）。**逐行一致**，
所以这次更正不是"改完开关看着差不多"，而是可复现的。

### 对结果的影响：小，但不是零

| | 更正前 | 更正后 |
|---|---|---|
| 面板 | 54,720 cell-hours（30 × 76 × 24） | **55,440**（30 × **77** × 24） |
| 出发总量 | 592,139 | **599,992**（+1.3%） |
| 第 4 折长度 | 19 天 | **20 天**（`np.linspace(0,77,5)` 切不出四等份） |
| XGBoost Poisson deviance | 1.810 | **1.829**（+1.0%） |
| XGBoost 校准 | 0.989 | **0.990** |

**排名完全没变**（XGBoost < gbdt < nb_glm < naive），四个模型的相对位置一个都没动。
换句话说：这次更正修的是数据口径的诚实性，不是模型的结论。

### 这次更正暴露的第三个"写死"bug

notebook 里 `obs_per_day` 原本写成 `tf.obs_sum / 19` —— **硬编码 19 天**。
第 4 折变成 20 天后，它把该窗口的日均需求算低了 5%，于是"最后一个窗口比第一个高多少"
输出 **+23.4%**，正确值是 **+17.3%**。

这和 §1 那个 `day_index` 错位、以及 `README` 里手写的版本号是同一类问题：
**一个和现实脱钩的常数，不会报错，只会静默给出错的数**。现在改成 `tf.obs_sum /
tf.valid_days`，天数由折自己给出。

---

## 1. 另一个会影响全部数字的 bug：`day_index` 错位

**位置**：`EDA/09_demand_model.py`（原第 115 行，现第 123–126 行）。

原来的写法是在 `dropna` **之前**给 `day_index` 打序号：

```python
p["day_index"] = p.dt.rank(method="dense").astype(int) - 1   # 在 dropna 之前
...
p = p.dropna(subset=[f"lag{L}" for L in LAGS]).reset_index(drop=True)
```

`dropna` 砍掉每个格子的前 7 天（`lag168` 需要 168 小时历史），但 `day_index` 是在砍之前
算的，砍完后最小值变成 **7**，而折边界（`np.linspace(0, len(days), N_BLOCKS + 1)`）是从
**0** 开始的位置序号。两套坐标系差了整整 7 天：

```
声明:  block 2 = 2026-03-28 .. 2026-04-15
实际:  模型训练 2026-03-09 .. 2026-03-20，在 2026-03-21 .. 2026-04-08 上"验证"
```

**每个折的验证窗口都比声明的早 7 天**，训练窗口跟着早 7 天。

### 它有影响，但不是数据泄漏

训练日仍严格早于验证日 —— 错位是两套坐标系同时平移，不是把未来喂给模型。
四折结构没有失效，**但每一个已发布的数字都变了**。

### 证据：声明一直是对的，代码是错的

修完之后（在当时的 76 天面板上）`EDA/out/table_h3_folds.csv` **字节完全相同**。仓库里那张"折窗口声明表"
从来没写错，是跑数的代码没照着它跑。这类 bug 只有把"声明的窗口"和"实际切出来的窗口"
对起来比才会现形，看代码是看不出来的。
（§0 的更正把面板变成 77 天后，这张表也跟着变了 —— 那是因为天数变了，不是这个 bug 复发。）

### 现在怎么防住

`model_XGBoost/results.ipynb` 第 2 节把这条变成了断言，任何一次重跑都会检查：

```
day_index : 0 .. 76 (期望 0 .. 76)
训练日严格早于验证日：3/3 折通过
与已保存的 metrics_by_fold.csv：最大差 1.776e-15  ->  逐位一致
```

---

## 2. 数据与任务（H1）

| | |
|---|---|
| 目标 | 每个（L6 格子, 小时, 日）的**出发次数** |
| 范围 | 出发量 top 30 的 L6 热点 × 77 天 × 24 小时 = **55,440** cell-hours |
| 总出发量 | 599,992；其中 69.9% 的 cell-hour 非零 |
| 日期 | 2026-03-09 .. 2026-05-24（**连续 77 天，没有缺口**） |
| 特征 | `hour, dow, is_weekend, lag1, lag24, lag168, lag1h_ma3, cell_prior` + `cell`（类别型） |

**两点关于这份数据的说明：**

- **热点集合没有变。** 用更正后的数据重算 `n_touching`，top-30 的集合与顺序与更正前
  完全一致（重算能逐格复现 `cell6_summary.csv`，最大差 0）。所以 `cats` 没有迁移，
  这不是"换了 30 个格子重来一遍"。
- **`cell_prior` 在每一折里只从该折的训练日算** —— 用全期均值会让折退化成非
  walk-forward。全量模型（§5）没有这个约束，用的是全 77 天。

---

## 3. 模型与参数

`count:poisson` 目标（对计数数据，Poisson deviance 才是正确的评分规则），
`tree_method="hist"`，400 棵树，`learning_rate=0.06`，`max_depth=6`。

**假设。** 树模型不设函数形式假设——不假设线性或乘法结构，`hour × dow × cell` 的交互
由分裂自动学出。唯一的显式分布假设来自损失：`count:poisson` 最小化泊松负对数似然，
等价于「给定特征后计数服从均值 = 方差的泊松」。§F4 已测得真实斜率 1.39（泊松要求
1.00），所以这条假设不成立；但泊松 deviance 对计数仍是正确的评分规则，且树不做
log-link 外推——实测 pred_max 402、从未越过观测上界 495（§4.2），而 nb_glm 越到 1,082。

**软件实现。** XGBoost 用原生 `xgboost` 3.1.1（UBJ 序列化，不依赖 PyTorch）；三个参照
模型的库与版本见 `EDA/out/model_summary.md` §H4（`scikit-learn` 1.7.1、`statsmodels`
0.14.5）；数据侧 `pandas` 2.3.2、`numpy` 2.3.1。

**参数是固定的，没有做搜索，这是刻意的。** 它们镜像自流水线里已有的
`HistGradientBoostingRegressor(loss="poisson")`：同样的 `n_estimators=400` 和
`learning_rate=0.06`，树深取 scikit-learn `max_leaf_nodes=31` 的等价物（2⁵ = 32 叶）。
这样两个模型之间的差异**只是库的差异，不是调参的差异** —— "参数怎么选的"有一个真答案，
而不是一个要人相信的搜索结果。

**没有 early stopping。** 一折能空出来的带标签数据只有它自己的验证集，在上面停就会把
答案泄进模型。

**离群值处理是继承的，没有重新决策**：见 §0，`trips_clean.csv.gz` 丢弃 Tier 0/1 中
真正不可能的记录（负时长、速度 > 30 km/h、跨研究区）加 `qc_dur_gt_180`，保留 99.81%。

---

## 4. 结果

### 4.1 三折（walk-forward，块 1 只训练）

| fold | 训练行 | 训练日 | 验证窗口 | 验证日 | MAE | RMSE | Poisson dev | 校准 |
|---|---|---|---|---|---|---|---|---|
| 1 | 13,680 | 19 | 03-28 .. 04-15 | 19 | 2.807 | 7.929 | 1.805 | 1.035 |
| 2 | 27,360 | 38 | 04-16 .. 05-04 | 19 | 2.782 | 6.604 | 1.670 | 1.002 |
| 3 | 41,040 | 57 | 05-05 .. 05-24 | 20 | 3.094 | 9.028 | 2.002 | 0.944 |
| **池化** | | | 41,760 行 / 455,591 次 | | **2.898** | **7.937** | **1.829** | **0.990** |

### 4.2 和流水线三个模型比（同样这些折、同样这些格子、**同一份数据**）

| 模型 | MAE | RMSE | Poisson dev | 校准 | 预测上界 |
|---|---|---|---|---|---|
| **XGBoost (Poisson)** | **2.898** | **7.937** | **1.829** | 0.990 | **402** |
| Gradient-boosted trees (Poisson) | 2.919 | 8.498 | 1.855 | 0.989 | 403 |
| Negative binomial GLM | 4.032 | 17.101 | 2.834 | 1.034 | **1,082** |
| Seasonal naive（基线） | 3.304 | 9.005 | 3.136 | 0.937 | 375 |

- 对基线：Poisson deviance **−41.7%**。
- 对流水线里最好的那个（同族的 sklearn GBDT）：**−1.4%**。
  差得不多是正常的 —— 参数是镜像过来的，**这 1.4% 就是"换库"的净效应**。
- 验证集里实际最大单点 **495 辆/小时**。XGBoost 的预测上界 402，**从未越过实测上界**；
  `nb_glm` 却越到 **1,082**（2.2 倍）。这是两者之间**可操作性**的区别，不只是准确率。

### 4.3 五张图怎么读

| 图 | 文件 | 一句话 |
|---|---|---|
| 1 | `out/nb_fig01_pred_vs_actual.png` | log-log 预测 vs 实测，一折一栏。点在对角线**上方**就是高估。三折散点形状几乎一样，说明误差结构稳定、不是某一折的偶然 |
| 2 | `out/nb_fig02_calibration.png` | 校准漂移 vs 需求水平。**随需求单调下滑** —— 需求越高越低估，这是"平"的模型的特征 |
| 3 | `out/nb_fig03_by_hour.png` | 误差按小时。**几乎全在早 7–9 点和晚 17–18 点**，其余 20 个小时基本无事 |
| 4 | `out/nb_fig04_count_band.png` | 按真实计数的分箱收缩。占 30.1% 的"零出发"行上平均浪费 0.577 辆；>100 辆/小时 的峰值段漏掉 12.9% |
| 5 | `out/nb_fig05_by_cell.png` | 30 个格子逐个。**误差集中在最头部 1 个格子**（`wx4eqt` 一个就占 16.4% 的出发量、deviance 6.39 是池化值的 3.5 倍）；尾部 5 个格子 deviance ≈ 1、校准 ≈ 1，因为那儿的模型基本在输出常数 —— 指标好看不代表有用 |

### 4.4 四条可以直接进 deck 的结论

1. **偏差随时间单向漂移：+3.5% → +0.2% → −5.6%。** 三个窗口的日均需求是
   7,305 / 7,655 / 8,567，最后一个比第一个高 **17.3%**。特征集里没有趋势项、也没有天气，
   模型一直在用过去的水平预测更高的未来。**这是全表信息量最大的一条**，它同时就是
   limitation 里"没有天气"的量化形式。
2. **主要失效模式是均值收缩，且方向对运营不利。** 静的时候浪费运力，忙的时候不够用。
3. **误差不均匀**：集中在 30 个格里的头 1 个、24 小时里的 4 个（7–9、17–18）。
   MAE 2.90 这个汇总数字会把整个结构抹平 —— 这就是残差地图值得画的原因。
4. **预测上界从未越过实测上界**（402 vs 495），而 `nb_glm` 越到 1,082。

---

## 5. 产物清单

| 文件 | 是什么 |
|---|---|
| `xgboost_model.py` | **三折训练与评估**，产出 `out/` 里**所有报出的分数** |
| `train_final.py` | **全量重训**，产出 `models/` 里的可部署产物；不打印任何分数 |
| `models/xgboost_poisson_full.ubj` | 全量模型本体，原生 XGBoost UBJ 二进制，2,189 KB |
| `models/xgboost_poisson_full.meta.json` | `cats` + 超参 + 训练窗口 + 每格 prior + 警告 |
| `results.ipynb` | 从**已跟踪的** `EDA/out/table_h1_panel.csv.gz` 重算三折与全部图 |
| `out/metrics_by_fold.csv` · `out/report.json` | 三折与池化指标，超参 |
| `out/predictions.csv.gz` | 41,760 行逐行预测（cell, date, hour, actual, pred） |
| `out/pred_vs_actual.png` | 图 1，由 `xgboost_model.py` 产出 |
| `out/nb_fig01..05_*.png` | 图 1–5，由 `results.ipynb` 重跑产出 |
| `../model_common.py` | 共享脚手架：面板、折、指标、对比 |

**为什么存的是 UBJ 不是 `.pt`。** `.pt` 是 PyTorch 的约定，而这里没有 PyTorch 模型；
`torch.save` 只是把 sklearn 包装器 pickle 一遍，会引入一个约 2 GB 的加载期依赖，
并把产物绑在 pickle 与库版本上。UBJ 是 XGBoost 原生格式，跨语言、跨版本稳定，加载
不需要 sklearn。实测同规模 UBJ 2,189 KB vs pickle 2,192 KB，往返误差都是 0。

**为什么只存全量这一个。** 三折的模型是被评估的模型，不是要上线的模型；每个折的训练集
都是全量的子集，留三个子集模型只会让"该加载哪个"变成一个错误来源。它们的数字在
`report.json` 里，代码在 `xgboost_model.py` 里，随时能重建。

### 怎么加载

```python
import json, sys
import pandas as pd, xgboost as xgb

sys.path.insert(0, "<仓库根>")
import model_common as C

meta  = json.loads(open("model_XGBoost/models/xgboost_poisson_full.meta.json",
                        encoding="utf-8").read())
cats  = pd.Index(meta["cell_categories"])          # 这行不能省，见下
model = xgb.XGBRegressor()
model.load_model("model_XGBoost/models/xgboost_poisson_full.ubj")

mu = model.predict(C.tree_design(new_df, cats))     # new_df 要有 8 个数值特征 + cell
```

### 加载时必须知道的两条

**① `cats` 不能省。** 原生格式存的是树，不存 pandas 的类别集合。`cell` 是类别型特征，
如果放任 pandas 按新数据里实际出现的值重新编码 —— 少一个格子、顺序变一下 —— 模型会把
某个格子**当成另一个格子**读，而且不报错。`meta` 里的 `cell_categories` 就是为了堵这个；
传进 `C.tree_design(df, cats)` 即可，没见过的格子会变成 NaN，被当作缺失而不是别的格子。

**② 这个模型没有任何诚实的验证分数。**

```
要引用性能   ->  out/report.json    （三折模型）
要部署       ->  models/*.ubj       （全量模型，未评分）
```

全量模型用了全部 77 天，没有留出任何一天来给它打分，所以 §4 那张表里的
1.829 / 0.990 **属于三折模型，不能安到它头上**。`train_final.py` 因此不打印任何分数，
连训练集上的都不打印 —— 一个在全量上拟合的模型的训练误差接近 0，印出来一定会被当成
真数字引用。`meta.json` 里 `validation_score` 是 `null`，旁边写了原因。

另外，全量模型用的 `cell_prior` 是**全 77 天**算的，比任何一折的都"厚"，和三个折的值
都不一样。这个选择是对的（上线时本来就拥有全部历史），但意味着全量模型和三折模型
**不是同一个模型**，连特征口径都不同。sidecar 里逐格记下了它用的 prior 值，
所以仅凭 `.ubj` + `meta.json` 就能精确复放。

---

## 6. 已知局限

- **趋势没建模。** §4.4 第 1 条：需求涨了 17.3%，模型只看到"过去的水平"。
  最直接的补法是加天气与节假日。
- **极值系统性低估。** 最严重的一例是 fold 3 的 `wx4eqt` 2026-05-11 早 8 点：
  实测 495、预测 124。这类点会直接表现为调度时的缺车。
- **第 3 折校准 0.944**，即整体低估 5.6%；前两折分别 +3.5% 和 +0.2%。
  校准不是常数，随需求水平漂移。
- **异常日只保留了一天。** `2026-05-17`（2,363 次出发，全期日均 7,855）现在在面板里，
  而 20 天的第 4 折也只包含这一个异常日 —— 这点样本量不足以让模型学会"异常日长什么样"。
  这是保留它的代价，也是为什么第 4 折的 deviance 最高（2.002）。
- **`EDA/out/` 已整体重跑到更正后的数据上**（`02`–`08` 全部重跑，16 张图与配套表格/JSON 全部刷新）。
  重跑后 `cell6_summary.csv` 的 top-30 与更正前**完全一致**，所以模型不需要跟着重跑。
- **预测的是"实际发生"而不是"潜在需求"**（§G 的截尾问题），这是天花板的一半。

---

## 7. 怎么复现

```bash
python EDA/01_data_cleaning.py         # 清洗策略在脚本顶部的 CLEAN_DROP_FLAGS 里
python EDA/02_region_map.py --png      # 区域底图 + fig07（--png 不能省，默认不写图）
python EDA/03_region_flow.py           # fig08, fig09
python EDA/04_distributions.py         # fig01, fig02
python EDA/05_temporal.py              # fig03, fig04
python EDA/06_spatial.py               # fig05, fig06, cell6_summary.csv
python EDA/07_spatiotemporal.py        # fig10, fig11, fig12
python EDA/08_uncertainty.py           # fig13
python EDA/09_demand_model.py          # 面板与流水线三模型（fig14-16, table_h5_*）
python model_XGBoost/xgboost_model.py  # 三折训练 + 评估，写 out/
python model_XGBoost/train_final.py    # 全量重训，写 models/（会断言重载后逐位一致）
jupyter nbconvert --to notebook --execute --inplace model_XGBoost/results.ipynb
                                       # 重算三折 + 重画 5 张图，写 out/
```

`02`–`08` 必须按编号顺序跑：`06_spatial.py` 写出的 `cell6_summary.csv` 是 `07` / `08` /
`09` 的输入（top-30 热点就是从这里选的）。

**环境上没有依赖清单**，所以记一笔：`02_region_map.py` 需要 `folium`（本次装进了
`computer-vision` 环境：folium 0.20.0 + branca + xyzservices）。缺它时脚本会在写
`region_map.html` 那一步报 `ModuleNotFoundError` —— 注意那时 `fig07_study_area.png` 还没写，
因为 PNG 排在 HTML 之后，虽然它本身只用 PIL、与 folium 无关。

`results.ipynb` 从**已跟踪的** `EDA/out/table_h1_panel.csv.gz` 读面板，所以队友不装
`trips_clean.csv.gz` 也能跑；它还会把自己算出的指标和 `out/metrics_by_fold.csv` 对上
（最大差 1.776e-15），对不上就报错。
