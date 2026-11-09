# inventory_distribution / GP 分布地图

用高德（Amap）底图画出 `xgb_bikeflow/data/hw6_grid_ownership.csv` 里的
L8 格点归属，以及全部 90 个 gathering point。

> 06:30 GP 库存的 OD-only 估计方法（下界 + 晚间先验 + 投影）见
> [`HANDOFF_0630_gp_inventory.md`](HANDOFF_0630_gp_inventory.md)。

## 数据口径

| 文件 | 内容 |
|---|---|
| `hw6_grid_ownership.csv` | 10,473 个 L8 格点；5,630 个已分配给 90 个 GP，4,843 个为 `UNASSIGNED` |
| `hw6_gp_master.csv` | 90 个 GP 的中心；本脚本用 `lon_gcj/lat_gcj` 两列 |

`hw6_grid_ownership.csv` 的 `lon/lat` 和 `hw6_gp_master.csv` 的
`lon_gcj/lat_gcj` 都是 **GCJ-02**，与高德瓦片一致，脚本不做任何坐标转换。
`hw6_gp_master.csv` 的 `lon/lat` 是 WGS-84，约偏 540 m，不能和高德底图一起用。

GP 点坐标取自主表的 GCJ 中心，而不是格点均值；其到自身全部格点的最远距离
约 247 m（中位数 112 m），脚本每次运行都会做这个一致性检查。

## 运行

```bash
cd inventory_distribution
python gp_amap_map.py
```

首次运行需要联网抓高德瓦片，之后缓存在 `.tilecache/`，可离线重跑。

常用参数：

```bash
python gp_amap_map.py --mode points       # 只画 Amap + 90 个 GP 点
python gp_amap_map.py --mode assignment   # 只画格点归属 + GP 点
python gp_amap_map.py --no-labels         # 不标 GP id
python gp_amap_map.py --no-html           # 不生成交互地图
python gp_amap_map.py --zoom 15 --pad-m 300
```

## 输出（`out/`）

| 文件 | 说明 |
|---|---|
| `fig01_gp_points_amap.png` | **最简视图**：高德底图 + 全部 90 个 GP 点（含 id 标签） |
| `fig02_grid_assignment_amap.png` | 高德底图 + 全部 10,473 个 L8 格点 + 90 个 GP 点；已分配格点按所属 GP 着色，`UNASSIGNED` 仅灰色描边 |
| `gp_points.csv` | 每个 GP 一行：GCJ 坐标、格点数、HW6 的 `dep_day/arr_day/out_p/in_p`、类别 |
| `gp_assignment_summary.csv` | 每个 GP 的格点数与格点经纬度包围盒 |
| `gp_points_amap.html` | 可交互的高德底图 + GP 点（Leaflet/Folium） |

## 早高峰边界流平衡

`flow_balance.py` 用同一套核心区域定义（`hw6_grid_ownership.csv` 的全部
10,473 个格点）计算工作日 06:30–10:30 跨越核心区域边界的流量：

```bash
python flow_balance.py
```

- **骑出**：起点在核心、终点在外，按 `start_time` 计；
- **骑入**：起点在外、终点在核心，按 `end_time` 计；
- 时间窗为 `[06:30, 10:30)`；工作日为周一至周五，剔除八个法定节假日，
  并计入 2026-05-09（周六调休）。

结果：57 个工作日，骑出 15,279、骑入 18,324，骑入/骑出 = **1.199**；
骑入比骑出平均多 **53.4 单/日**（95% CI 46.0–60.8），55/57 天骑入更多。
逐日结果见 `out/workday_morning_boundary_flow.csv`。

注意：`dataset_train_20260921_lsz.csv` 中 667,821 单里有 483,287 单是
**核心区域内部**行程（起终点都在核心），它们不跨越边界；上表只统计
94,855 条骑出 + 89,678 条骑入 + 1 条两端都在外的记录。

## 依赖

`pandas`, `numpy`, `matplotlib`, `Pillow`, `folium`, `scipy`。瓦片拼接复用
`../EDA/basemap.py`，格点矩形复用 `../EDA/cells.py`（脚本运行时自动把
`../EDA` 加入 `sys.path`）。
