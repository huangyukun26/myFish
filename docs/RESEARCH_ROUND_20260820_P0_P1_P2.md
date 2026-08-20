# Research Round Status: P0 / P1 / P2

更新：2026-08-20

这份文档只上传可审核的汇总数字。原始 logits、feature cache、candidate packet、图片和运行目录仍保留在本地 `runs/`，不进入 GitHub。

## 当前线上基线

当前最佳包没有被本轮实验改动：

- seen：`0.779917`
- unseen：`0.150822`
- overall：`0.505313`（`18022 / 35665`）

整体提高 1 个百分点需要约 `+357` 个净正确样本。

## P0：瓶颈诊断

### Seen

固定 OOF 验证共 10,790 行，基线为 `9,883 / 10,790 = 0.91594`。专家 oracle（只说明潜在互补上限，不是可提交结果）为：

| panel | rows | base correct | oracle complement |
|---|---:|---:|---:|
| all | 10,790 | 9,883 | +375 |
| dev | 7,444 | 6,715 | +309 |
| sealed | 3,346 | 3,168 | +66 |

实际专家在 sealed 上没有正净增；最好的候选仍为 `-1`。结论：当前候选 bank 的理论互补存在，但现有 gate/校准不能可靠提取，不能生成 seen overlay。

### Unseen pseudo split

1000-class species/genus pseudo split 的 top-K 召回：

| method | species @1 | species @20 | species @50 | genus @1 | genus @20 | genus @50 |
|---|---:|---:|---:|---:|---:|---:|
| taxon | 26.6% | 70.2% | 80.7% | 23.4% | 69.9% | 80.0% |
| fish | 25.3% | 69.9% | 81.0% | 23.0% | 70.9% | 80.4% |
| description | 14.5% | 48.1% | 60.2% | 12.0% | 46.3% | 60.0% |
| rank fusion | 26.0% | 69.3% | 79.8% | 22.5% | 68.3% | 79.5% |
| oracle union | 42.0% | 82.2% | 88.1% | 38.6% | 81.6% | 88.1% |

结论：description 不是当前主解；unseen 仍有候选召回空间，但简单 rank fusion 没有把 oracle 空间转化为 top-1。

## P1：固定融合复播

`fish_taxon_avg` 在三个 seed 上的净变化：

| split | seed 42 | seed 43 | seed 44 |
|---|---:|---:|---:|
| species | -1 | +7 | +11 |
| genus | +9 | +18 | +8 |

visual-trait 方向明显负收益（species `-135`，genus `-117`）。P1 没有跨 seed 同向、足够大的锁定规则，因此不进入测试集。

## P2：层级校准小扫

- species 最好：`alpha=0.10, beta=0.00`，净增 `+2 / 1000`
- genus 最好：`alpha=0.05, beta=0.03`，净增 `+1 / 1001`

这是小样本 pseudo split 上的微弱信号，远低于提交门槛，不能外推为 leaderboard 增益。

## 提交决定

本轮没有生成新 submission，也没有消耗提交次数。原因是：

1. seen 严格 sealed 没有可复现正收益；
2. unseen 的正收益只出现在单个 seed，且幅度很小；
3. P0/P1/P2 都没有达到约 `+357` overall rows 的现实门槛。

## P3 / P4 / P5 为什么没有做

需要明确：**2026-08-20 的正式实验计划只定义了 P0、P1、P2，P3/P4/P5 当时没有被正式创建或排程。** 因此它们不是“跑失败”，而是尚未启动。

按后续工作分期，建议定义为：

- **P3：学习型候选重排/残差校正。** 需要先用 group-aware OOF 证明 P0 的 candidate recall 能被稳定转化；P1/P2 尚未满足，所以没有贸然训练。
- **P4：新的通用视觉专家或云端大显存实验。** 6GB 足以完成本轮诊断，当前阻塞点不是显存；只有 P3 证明目标方向后，云端新 encoder 才值得开机。仍只能使用规则允许的通用模型和官方数据。
- **P5：全量 test 推理、锁定 overlay、打包提交。** 只有 P3/P4 在 sealed 多 seed 同向净增后才能执行；本轮没有达到这个入口条件。

下一轮应先完成 P3 的 group-aware OOF 候选重排，未达到预设净增门槛就不进入 P4/P5。
