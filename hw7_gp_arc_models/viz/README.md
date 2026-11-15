# GP / Arc 预测可视化

由 `scripts/build_prediction_viz.py` 从 `runs/gp_arc_models` 生成。

## 文件

| 文件 | 内容 |
|---|---|
| `index.html` | 单文件交互地图：实际 / 预测 / 残差，按 42 个样本外日期 × 8 个半小时播放 |
| `figures/residual_distribution.png` | 残差分布（裁剪到 99 分位） |
| `figures/pred_vs_actual.png` | 预测-实际 hexbin |
| `figures/calibration.png` | 按预测值分 12 箱的校准曲线 |
| `figures/error_by_slot.png` | 每个半小时的 MAE |
| `figures/totals_by_day.png` | 每日总量：实际 vs 预测 |
| `figures/top_units_bias.png` | 平均偏差最大的 10 个单元 |

## 怎么用 `index.html`

双击打开即可（Leaflet 已内联，不依赖 CDN；高德底图需要联网，断网时弧线和数值仍然可用）。
右侧可以切换：

- **面板**：Arc（151 条服务弧）与 GP（30 个集散点）；
- **通道**：车借 / 还车；
- **内容**：预测值 / 实际值 / 残差（预测 − 实际；红=高估，蓝=低估）；
- 拖动日期与半小时滑块，或点“播放”。

鼠标悬停一个弧/一个 GP 可以看到实际值、预测值与差值。右侧还会显示本帧的残差分布和残差最大的 6 个单元。

## 数据口径

- 数值来自三折**样本外**预测（42 个工作日），不是训练集拟合值；GP 与 Arc 的行数分别为 20,160 与 101,472（每通道各一半）。
- **Arc 之间共享服务 patch**（329 个 patch 中 275 个属于 ≥2 条弧），所以不同弧的预测不可相加当全网需求；Arc 44 无 patch，恒为 0。
- 残差 = 预测 − 实际。正值表示高估。
- 颜色按全局 99 分位归一，切换帧时色标不变，便于跨帧比较；打开时默认停在"总量最接近
  实际"的日期与 08:30–09:00，避免第一天作为离群帧误导第一印象。
- 残差模式下零值附近是浅灰蓝，红/蓝只表示方向与幅度。

## 已知问题

- 趋势（`volume_index`）是一个拟合后外推的点估计；它只进入 offset，且被 cap（GP 1.20/1.15，Arc 1.20）限制。
- 天气用实况而非预报，`is_raining` 取的是该半小时**之前**的天气。
- 超参和特征在同一组三折上选择，指标略偏乐观。
- 最后一个验证日 2026-05-22 的数据不完整（总量约为平时的 1/4），`totals_by_day.png` 里用灰色标注；
  单帧数字会因此偏低。
- 分时段校准存在系统差：slot 1 略高估（约 +11%），slot 5–7 略低估（约 −11%），这是模型行为，不是图表问题。

## 复现

```bash
PYTHONPATH=src python scripts/build_prediction_viz.py --run runs/gp_arc_models
```
