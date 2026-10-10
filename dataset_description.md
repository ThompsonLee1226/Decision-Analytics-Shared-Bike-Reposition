`trips_clean.csv.gz` 共 41 列 = 原始 7 列(原样保留)+ **34 个新增列**。按来源分四组:

## 时间特征(8 列,全部以 `start_time` 为准)

| 列名 | 类型 | 含义 |
|---|---|---|
| `duration_min` | float | `end_time − start_time`,单位分钟。清洗后范围 1–180 |
| `date` | string | 出发日期 `YYYY-MM-DD`,如 `2026-03-02` |
| `hour` | Int16 | 出发小时 0–23 |
| `minute` | Int16 | 出发分钟 0–59 |
| `dow` | Int16 | 星期,0=周一 … 6=周日 |
| `is_weekend` | bool | `dow >= 5` |
| `slot_15` | float | 出发时刻所属的 15 分钟槽 0–95,= `(hour*60+minute)//15` |
| `slot_60` | Int16 | **就是 `hour` 的别名**(0–23),保留只为让 `slot_*` 命名统一 |

注意:时间特征只有出发侧,没有 `end_hour` 之类;终点只有 geohash 和坐标。

## 空间特征(10 列)

| 列名 | 含义 |
|---|---|
| `start_cell4` / `start_cell5` / `start_cell6` | 起点 geohash 截断到前 4/5/6 位,即三级可调粒度的网格单元。**L6 是分析单元(约 937 m × 607 m)** |
| `end_cell4` / `end_cell5` / `end_cell6` | 同上,终点侧 |
| `start_lat` / `start_lon` | 起点 8 位 geohash 单元**中心点**(约 38 m 单元),由脚本自带的纯 Python geohash 解码器算出 |
| `end_lat` / `end_lon` | 同上,终点侧 |

坐标基准是 **GCJ-02**,与原始数据和围栏一致,不要转换。

## 运动学特征(3 列)

| 列名 | 含义 |
|---|---|
| `crowfly_km` | 起终点中心点的**大圆(haversine)距离**,单位 km。是真实路距的**下界**;清洗后最大 29.4 |
| `implied_speed_kmh` | `crowfly_km / (duration_min/60)`,最小可能速度。清洗后最大 29.98 |
| `is_same_cell` | `start_geohash == end_geohash`,即起终点落在**同一个 ~38 m 单元**内 |

`implied_speed_kmh` 在 `duration_min ≤ 0` 时**故意置为 NaN**,这样零时长行程只触发 `qc_zero_dur` 一个标志,不会被速度规则二次计数。

## 质量标志(13 列,`qc_*`)

| Tier | 列名 | 判据 |
|---|---|---|
| 0 可解析性 | `qc_null_required` | 任一必需字段为空 |
| 0 | `qc_dup_order` | `order_id` 重复 |
| 0 | `qc_unparsed` | 起止时间无法解析 |
| 0 | `qc_bad_geohash` | geohash 不是合法 8 位 base32 |
| 1 自相矛盾 | `qc_neg_dur` | 时长 < 0 |
| 1 | `qc_zero_dur` | 时长 = 0(分钟截断产生的伪零) |
| 1 | `qc_speed_gt_30` | 大圆推算速度 > 30 km/h(物理不可能) |
| 1 | `qc_cross_region` | 任一端的 4 位前缀落在 `wx4e`/`wx4g` 之外 |
| 2  atyp但可能真实 | `qc_dur_gt_60` | 时长 > 60 min —— **仅提示,不删行** |
| 2 | `qc_dur_gt_180` | 时长 > 180 min(疑似"占车") |
| 2 | `qc_same_geohash` | 等同 `is_same_cell` —— **仅提示,不删行** |
| 2 | `qc_anomaly_day` | 当日总量 < 其中心 7 日滚动中位数的 50% |
| 2 | `qc_outside_window` | 在研究窗口之外(窗口未设置时**恒为 False**) |

## 两个使用上的坑

1. **11 个标志列在本文件里是恒为 False 的死列。** 因为凡是会置真标志的行都已被删掉,于是 `qc_zero_dur`、`qc_speed_gt_30`、`qc_cross_region`、`qc_dur_gt_180`、`qc_anomaly_day`、`qc_outside_window` 以及 4 个 Tier-0 标志和 `qc_neg_dur` 在本文件里全为常数 False。做特征矩阵时应剔除(零方差)。真正还"活着"的只有 `qc_dur_gt_60`(10,665 行)和 `qc_same_geohash`(3,417 行)。

2. **`action_dt` 不是新增列**,它是原始列且与 `date` 语义重复,但字符串形式不同(`2026/3/2` vs `2026-03-02`),按文本连接会匹配不上。