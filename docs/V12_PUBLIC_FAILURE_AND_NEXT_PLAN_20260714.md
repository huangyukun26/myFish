# v12 榜单失败复盘与下一步方案（2026-07-14）

## 结论

以 2026-07-02 包为唯一有效基线。停止提交 v10、v12、v13、v14、v15 以及 q0/q50/maxnet 变体；这些包都包含已经被真实榜单否定的 seen 918 行改动，且部分 unseen 改动会覆盖 7.2 已验证有效的结果。

下一次若需要用一个提交同时验证 seen 和 unseen，应当只组合“一条 seen 机制 + 一条 unseen 机制”。Codabench 分别返回 `accuracy_test` 与 `accuracy_unseen`，因此两侧可以在同一个 ZIP 中独立归因。不能在同一 split 内混入多个新机制。

## 真实结果

| 包 | seen | unseen | overall |
|---|---:|---:|---:|
| 7.2 baseline | 0.7654376275 | 0.1508221994 | 0.4971540726 |
| v12 balanced q25 | 0.7582723790 | 0.1492163412 | 0.4924155334 |
| v12 相对 7.2 | -144/20097 | -25/15568 | -169/35665 |

v12 相对 7.2 修改 seen 918 行、unseen 650 行。真实条件净收益分别为：

- seen：`-144/918 = -15.69%`；
- unseen：`-25/650 = -3.85%`。

这不是显示精度或随机波动，而是确定的负迁移。

## 原评估为什么失效

### 1. seen 分数外推公式错误

原报告把 OOF 的 `+264` 直接写成 `0.7654376 + 264/5795 = 0.81099`。但 `+264` 是 1280 次替换在另一个代理基线上的净收益，不能直接叠加到完整 7.2 成品分数。

即使乐观地假设 OOF 条件收益率 `264/1280=20.63%` 可以迁移到 public 的 918 次修改，估计也只是净增约 189 张，即 seen 约 `0.7749`，不是 `0.8110`。原公式实际暗含 918 次修改净增约 915 张，要求近 99.7% 的净命中率，与 OOF 本身矛盾。

### 2. seen 验证分布与榜单指标错位

5,795 张 holdout 是“一类一图”，优化的是 class-balanced/macro 指标；其中 `<=5-shot` 类占 79.36%。隐藏榜单是 20,097 张图片上的 micro accuracy，在训练图像权重下 `<=5-shot` 仅占 18.13%。

v12 的 918 次 seen 修改中：

- 690 次移向训练样本更少的 species；
- 676 次目标为 `<=5-shot` 类；
- 441 次目标恰好为 2-shot 类；
- 来源类训练样本中位数为 11，目标类中位数为 3。

因此路由器系统性放大了尾类先验，与隐藏 micro 指标相反。

### 3. 代理基线不是完整 7.2 pipeline

OOF 比较使用 `fixed_fusion_logits.argmax`，而 public 修改覆盖在已经经过 hflip/letterbox/concat/router/consensus 的完整 7.2 预测上。代理基线与提交基线没有逐图复现，导致“代理上的纠错”在真实基线上不再成立。

### 4. 重复调参与验证泄漏

同一 OOF 上枚举约 1,634 个阈值/专家顺序，再选最好配置，没有 untouched outer holdout。云端 balanced split 又将原 train 53,469 与原 validation 10,790 合并后再做一类一图 holdout，使原 validation 参与了模型训练。

### 5. pseudo-unseen 不是六个独立验证集

species/genus × seed 42/43/44 的 6,003 次评估仅涉及 1,823 张 unique 图。不同 seed 主要更换候选 distractors，真值图像重复；species 与 genus manifest 还重叠 178 张。阈值和 top-k 又在同一批图上反复选择，`+323/6003` 严重高估了泛化证据。

### 6. v12 覆盖了已由榜单证明有效的 7.2 行

7.2 相对上一包的真实收益为：

- seen router：修改 391 行，净增 52 张；
- unseen consensus：修改 3512 行，净增 125 张。

v12 在 seen 侧覆写其中 52 行；在 unseen 侧覆写其中 219 行，其中 48 行直接退回旧 letterbox 标签。后续所有实验必须冻结这两个 branch-level protected set。

## 下一包：v17 micro-aligned dual probe

路径：`runs/submission_20260714_v17_microaligned_dual_probe/submission.zip`

它从完整 7.2 逐行复制，只加入两个单机制分支：

- seen 43 行：历史候选血缘全部给出同一替代标签、与 7.2 当前标签同 genus，并且替代类训练数量不少于当前类；不再向更稀有类别移动；
- unseen 106 行：仅采用 structured taxonomy 的 known-genus → novel-genus 路由，并排除 7.2 proven consensus 的全部 3512 行；
- protected seen 391 行覆写数：0；
- protected unseen 3512 行覆写数：0；
- ZIP 仅含根目录 `prediction.json`，共 35,665 行，键顺序与 7.2 一致，标签全集合法；
- SHA256：`dfee2366f0c248dbefb369b3872880f20b9f7053786c3e620caf459425cc84e7`。

这里的“历史候选一致”只是共享血缘的 lineage consensus，不是 11 个独立模型证据。v17 是低覆盖榜单探针，不是 0.8/0.2 的分数承诺。其硬覆盖风险上界为 seen `43/20097=0.00214`、unseen `106/15568=0.00681`；实际正负必须由榜单决定。

## 一次提交后的明确决策

| v17 结果 | 下一步 |
|---|---|
| seen、unseen 都正向 | 保留两条分支，分别扩大覆盖；下一包仍各只扩一个预先锁定层级 |
| seen 正、unseen 负 | 保留 seen 43；unseen 完全退回 7.2，再换一条 unseen 机制 |
| seen 负、unseen 正 | 保留 unseen 106；seen 完全退回 7.2，再换一条 seen 机制 |
| 两边都负 | 完全退回 7.2；停止当前 cloud candidate family，完成新验证协议后再提交 |

任何一侧只有 1 张净提升也能由榜单精确检测。为了避免在噪声很小的结果上扩得过快，建议扩覆盖的门槛为净增至少 5 张且方向与预注册验证一致。

## 新实验准入协议

1. 模型只使用原始 train 53,469 张训练；原 validation 10,790 张完整冻结，禁止进入训练、阈值选择和专家顺序搜索。
2. 在 frozen validation 上逐图复现完整 7.2 pipeline；若不能逐图复现，不允许计算“相对 7.2 增益”。
3. 主指标使用 10,790 张的 micro accuracy；一类一图 macro 只作为尾类诊断。
4. inner folds 用于训练/阈值选择，untouched outer holdout 只在规则锁定后评估一次。
5. 每个候选必须报告 `changed / wins / losses / net / net-per-change`，并按训练频次、same/cross genus、图像问题类型分层。
6. public 风险估计只允许使用 `实际 public 覆盖量 × outer holdout net-per-change`；禁止再次使用 `baseline + OOF net / holdout_size`。
7. 每个 split 每个提交只允许一个新机制。seen 首次覆盖不超过约 50 行；unseen 首次覆盖不超过约 150 行。榜单正向后才扩展。
8. 8k 人工问题标签应作为分层评估维度：主体不完整、小目标、死亡/异常、位置不明、多鱼/杂鱼、保护色、复杂背景、花纹干扰、模糊、颜色偏移、绘图/非实物、水体/拍摄环境、手部/人物干扰；不能只看全局平均分。

## 目标判断

`0.85/0.25` 仍可作为最终目标，但当前证据不支持把它当下一包的可信预期。先恢复可校准的实验—榜单关系，再逐步扩大公开验证为正的分支，才有机会稳定超过 7.2。当前首要目标是“不再用错误代理消耗提交次数”，而不是用更大覆盖赌一次跳分。
