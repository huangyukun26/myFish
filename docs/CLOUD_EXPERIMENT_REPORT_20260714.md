# Fish Species Recognition Challenge：云 GPU 实验报告（2026-07-14）

> **已作废（2026-07-14 v12 public 结果后）**：v12 实测 seen `0.7582723790`、unseen `0.1492163412`，相对 7.2 分别净损失 144 与 25 张。本文的 `0.81099/0.20463` 代理、推荐提交顺序及 v13-v15 扩展建议均不得继续用于提交决策。完整原因和新方案见 `docs/V12_PUBLIC_FAILURE_AND_NEXT_PLAN_20260714.md`。

## 基线与目标

- 纠正后的基线为 2026-07-02 submission：seen `0.7654376`，unseen `0.1508222`。
- 期望目标为 `0.85/0.25`，可接受目标为 `0.80/0.20`。
- seen 选择全部使用 5,795 类 one-query-per-class 验证，并按 genus 哈希四折 OOF；unseen 使用 species/genus × seed 42/43/44 共六个伪 unseen 划分。

## 分数代理与目标差距

- seen balanced 的内部代理为 `0.7654376 + 264/5795 = 0.81099`；maxnet 为约 `0.81168`。要达到 0.85，仍需在同口径下再增加约 226 个净正确样本。
- seen balanced 四个 genus 哈希折均为正，最差折约为 `+52/~1449 = +3.59` 个点；把该最差折增益叠加到基线仍约为 0.801。因而 0.80 不只依赖平均值，在四折最差情形下也有内部支撑。
- unseen v12 q25 的内部代理为 `0.1508222 + 323/6003 = 0.20463`；v15 为 `0.1508222 + 330/6003 = 0.20580`。要达到 0.25，仍需再增加约 265 个净正确样本。
- `+323/+330` 只计 structured + genus gate 的核心逐样本组合；最终包同时保留伪划分合计约 `+8` 的 five-crop 微路由。为避免未知重叠被重复计数，分数代理没有额外加这 8 个净正确样本，因此属于较保守口径。
- 因隐藏 public 的物种/属构成和图像问题比例未知，这些是用于比较实验的线性代理，不是 leaderboard 分数预测或保证。当前证据支持把 `0.80/0.20` 作为现实目标，不支持宣称 `0.85/0.25` 已经达到。

v15 对 species-like unseen 平均提升约 8.47 个点，对 novel-genus-like unseen 平均提升约 2.53 个点。按隐藏 unseen 中 species-like 占比做敏感性分析：

| species-like 占比 | unseen 线性代理 |
|---:|---:|
| 0% | 0.176 |
| 25% | 0.191 |
| 50% | 0.206 |
| 75% | 0.221 |
| 100% | 0.236 |

达到 0.20 约需 species-like 占比至少 40%；即使全部为 species-like，当前证据也不足以支持 0.25。

## Seen 结果

### 多视图监督头

- 完整缓存了 64,259 张训练图的 BioCLIP five-crop 特征。
- 新输入为 `hflip + letterbox + fivecrop`，共 `3×1024` 维。
- 单模型 raw top-1：2027 `0.54461`、2028 `0.54513`、2029 `0.54668`；旧监督头为 `0.52545`。
- 三模型 raw top-1 为 `0.55720`；五模型固定融合为 `0.55789`，相对旧固定融合净增 132/5795。
- label-smoothing 模型单独不强，但提供互补；五模型加该模型的固定融合为 `0.56238`，净增 158/5795。
- 该专家的 top-15% OOF 路由净增 173，四折为 `[+46,+51,+46,+30]`。

### 消融

- hidden 8192、hidden 2048、dropout 0.1/0.5 都没有提高五模型集成边际收益。
- 普通交叉熵 raw top-1 仅 `0.48059`，淘汰。
- label smoothing 单模型 raw `0.53926`，但加入集成后固定融合从净增 132 提到 158，保留。
- 全量 refit 与留出模型在 public top-1 上一致率 `86.93%`；因缺少独立 OOF，仅保留共识实验包，不替代主包。

### 多专家组合

专家包括 DINO metric、tri-view、priority five-crop，以及多视图监督集成。最终穷举专家顺序与阈值后：

- `v10_maxnet`：OOF changed 1280、net `+268`，四折 `[+53,+84,+81,+50]`。
- `v10_balanced`：OOF changed 1280、net `+264`，四折 `[+54,+81,+77,+52]`。
- 旧 v5 为 net `+235`，因此 v10 是本轮 seen 主候选。
- balanced 与 maxnet 在全部 public 上只分歧 52 张（20 张同 genus、32 张跨 genus）。maxnet 仅多 4 个总 OOF 净正确，但最差折从 +52 降到 +50，因此主包选择 balanced，maxnet 只作小范围消融。

注意：OOF 增益不能机械外推成 leaderboard 分数；它只证明选择性替换在所有 genus 折上方向一致。

## Unseen 结果

- 最稳主线仍是 structured taxonomy/adapter + novel-genus 路由：六划分净增 `[+1,+1,+2,+27,+27,+22]`，合计 `+80/6003`，最差划分 `+1`。
- five-crop 只有在 `1.25 <= aspect ratio < 1.5` 且权重 0.1 时六划分非负，合计 `+8/6003`；public 严格门只改 39 张。
- 新做的 structured × five-crop 联合网格表明原图权重 0 最好；五裁剪混入后总净增降至 75 或更低，拒绝。
- 新增 supervised genus gate：seen 数据训练的 2,065-genus MLP 验证 top-1 为 `0.72770`，仅对当前预测属于已知 genus 的 unseen 样本 rerank。
- genus gate 的 top-100/q25 在 species42/43/44 上分别净增 `[+80,+82,+82]`，损失仅 `[2,1,2]`；novel-genus 三折保持不动。
- 与 structured 路由逐样本组合后的六折净增为 `[+81,+83,+83,+27,+27,+22]`，总净增 `+323/6003`、最差折 `+22`；共改 754 次，仅 5 次已知损失。
- q0 激进组合总净增 `+337/6003`，但 species 三折有 37 次损失，风险显著高于 q25。
- Transductive CLIP/EM-Dirichlet 总净增仅 21 且存在负折；SigLIP2、FG-CLIP 和全局 five-crop 均淘汰。
- v12 q25 public 相对 7.2 改 650 个 unseen；q50 改 276；q0 激进版改 2,924。
- seed 42/43/44 复用同一批图像，只改变候选集合，因此不能把六折当作独立图片。早期逐折 bootstrap `[+289,+358]` 偏窄；按同图最大相关性做保守压力测试后，v12 的 95% 范围为约 `[+265,+383]`。v15 用真实逐图 delta 做 image-cluster bootstrap，95% 区间为 `[+277,+385]`，抽样总净增仍全部为正。这些都只衡量内部伪 unseen 稳健性，不是隐藏榜单置信区间。
- 追加单调置信度子集 v13：保留 v12 q25 的 428 个 genus 改动中 margin 最高的 364 个。对应的 q35 伪划分参考为 species `[+73,+76,+76]`（三折均 0 loss），连同 genus 后总净增 `+301/6003`；public 相对 7.2 改 seen 918、unseen 589。它是 q25 与 q50 之间的平衡档，不替代主包。
- 失败样本显示负改主要是 `Etheostoma/Percina`、`Floridichthys/Cyprinodon`、`Plectropomus/Gracila` 等外观相近的跨属混淆，而不是单纯模糊或小目标。基于此增加 current-genus advantage gate：直接要求预测 genus 的 logit 高于当前基线 genus。q30 在 species 三折为 `[+83,+85,+86]`、损失 `[1,0,0]`；连同 genus 后总净增 `+330/6003`，逐图聚类 bootstrap 95% 区间 `[+277,+385]`。public 改 seen 918、unseen 1,210。该方案是在查看失败样本后产生，存在自适应选择乐观偏差，因此列为高潜实验包 v14，不替代预先验证的 v12 主包。
- 三随机种子 genus ensemble 复核后，q30 仍为 `[+83,+85,+87]`，说明 current-genus advantage 方向不依赖单一权重。最终 v15 要求单模型与三模型预测 genus 一致：伪 unseen 保持 `[+83,+85,+86,+27,+27,+22] = +330` 和 1 次损失不变，public genus gate 从 v14 的 1,016 降至 923，最终 public unseen 改动从 1,210 降至 1,121。因此高潜实验档优先 v15，v14 只作消融保留。
- v15 的 923 个 genus 改动覆盖 490 个源 genus、473 个目标 genus、765 个不同 genus 对；最大单一 genus 对仅占 0.87%，前 10 对合计 5.63%。未发现由少数映射对主导的类别坍缩。
- v12 的 428 个 genus gate 改动中有 299 个（69.9%）也被 v15 选中，且这些交集的目标 species 100% 相同；v15 另有 624 个新增改动，v12 有 129 个独有改动。因此 q25 正向能支持继续试 v15，但不能完全保证 v15 的新增覆盖也正向。
- v15-only 的 624 张与两方法交集的 299 张在图像质量代理上接近：中位最短边 379px vs 364px，长宽比、对比度和边缘能量基本一致。新增覆盖没有明显集中在小图或模糊图，不增加额外质量阈值。
- 进一步要求 hflip/letterbox 与 five-crop 的 genus 预测一致时，species 总净增从 254 降到 245–251，损失仍为 1，说明视图共识只删掉正确纠错而未额外消除错误；拒绝 v16，最终停在 v15。
- 再用“替代 species 的 BioCLIP 相似度不得明显低于当前 species”保护时，消除最后 1 次损失需要把 species 净增从 254 降到 152；说明 genus 监督头的有效纠错恰恰经常发生在原始 species 相似度不足处。拒绝该 guard。

q25 组合在伪 unseen 上约提升 5.4 个百分点，内部证据第一次达到 `0.20` 所需量级；但 public 的 species/genus 构成未知，仍不能承诺 leaderboard 一定达到 0.20，更不能承诺 0.25。

## 最终候选包

所有 ZIP 均已验证：根目录只含 `prediction.json`，键顺序与 7.2 一致，总计 35,665 行，并保存 SHA256。

1. `v12_balanced_q25_submission.zip`：首选；seen OOF `+264`，unseen 六折 `+323`；public seen 改 918、unseen 改 650；SHA256 `26f127570fc4f7173f2a39166f0aa9709f4ce06e12730876254b86bdcefda8d3`。
2. `v12_maxnet_q25_submission.zip`：seen 总收益优先；seen OOF `+268`，unseen 与上相同；SHA256 `e3a7f36803faea0324763ac134e63bb338911af1e0b79e632df63df23d007f3d`。
3. `v13_balanced_q35_subset_submission.zip`：中风险平衡档；public seen 改 918、unseen 改 589；SHA256 `f1b49e87cd384f6401a9a4d3ed8f7d9c07367aeebcad8cf0842bccce6d49ba16`。
4. `v15_balanced_current_genus_adv_q30_consensus_submission.zip`：高潜共识实验包；伪 unseen `+330`、public unseen 改 1,121；SHA256 `41f74c08da8207e3b6589da818173cfd0ffbd37d5a76c0eeafd6ff20f7af0259`。
5. `v14_balanced_current_genus_adv_q30_submission.zip`：v15 的无共识消融；public unseen 改 1,210；SHA256 `479cbbba3b92ac03aa79f5112105789cf20bd70d49dbf74b8cde8755de910336`。
6. `v12_balanced_q50_submission.zip`：更保守；public unseen 改 276；SHA256 `9fca69f9ca72e91f67d6a967210e5512fa9aac539fa9dd4399508d36007f69f0`。
7. `v12_balanced_q0_aggressive_submission.zip`：冲 unseen 覆盖的高风险版，public unseen 改 2,924；SHA256 `2d0d3b24a02eb5f1c01ab6c17338012dd489facaf70429626833e0251a7aed68`。

v10 严格包仍保留为“不启用 genus gate”的回退。
其 SHA256 为 `65988253f7cb9e5928385e05043b26fc44e141a0f0c83f105b2339939b372749`。

本地回传目录：`runs/cloud_20260714_return/`。

## 建议提交顺序

若提交次数有限：

1. 先交 `v12_balanced_q25`，同时观察 seen/unseen 两项。
2. 若 q25 unseen 明确正向且希望扩大覆盖，第 2 包优先试 `v15_balanced_current_genus_adv_q30_consensus`；它比 q0 保守得多，v14 仅作消融。
3. 若 q25 unseen 负向，第 2 包交 `v10_balanced_strict` 撤掉 genus gate；两者 seen 完全相同，可直接隔离 unseen 路由影响。
4. 若 q25 只有小幅波动、希望更保守，提交 `v13_balanced_q35_subset`。
5. 只有 q25/v15 均支持扩大覆盖时才尝试 `v12_balanced_q0_aggressive`；若 q25 负向，不要提交 q0。
6. `v12_maxnet_q25` 只用于验证 seen 专家顺序；保留 7.2 和 v5 回退。

## 前沿资料与实验关系

- [BioCLIP 2](https://arxiv.org/abs/2505.23883)：支持继续采用生物层级监督 backbone，而非通用 VLM。
- [Transductive Zero-/Few-Shot CLIP](https://openaccess.thecvf.com/content/CVPR2024/html/Martin_Transductive_Zero-Shot_and_Few-Shot_CLIP_CVPR_2024_paper.html)：本轮复现相关概率校准思路，但六划分稳定性不足。
- [SCAP](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_SCAP_Transductive_Test-Time_Adaptation_via_Supportive_Clique-based_Attribute_Prompting_CVPR_2025_paper.html)：支持局部属性与测试时适配方向；本轮有效部分体现为多视图和保守路由。
- [FishNet++](https://arxiv.org/abs/2509.25564)：其细粒度鱼类识别结论与本轮通用 VLM 表现弱、BioCLIP 更有效的观察一致。

## Workflow 修正

- Balanced Softmax 必须显式配 `class_weight=none`；自动队列已修正。
- 任何“full train”缓存必须核对行数和唯一 image id；旧缓存只有 53,469 行，已拒绝并重建为 64,259 行。
- novelty gate 的 known classes 必须使用 `seen_class_to_idx.json`；误用全类表会把 known genus 从约 1850 错增至约 3860，该无效网格已隔离保存且不计入结论。
- 最终包必须执行 ZIP 根目录、行数、键顺序、seen/unseen 改动数和 SHA256 五项审计。

## 下一轮优先实验

1. **Family/order gate，专攻 novel-genus。** 为全部 species 补齐 family/order taxonomy，用 seen 图像训练 family 与 order 监督头；在当前 genus 不可知时，只在预测 family/order 内重排 novel genus。必须沿用三个候选集 seed，并要求 genus 三折全部正向、最差折至少 `+10/1001` 才进入 public。它是突破 unseen 0.21–0.24 区间最直接的方向。
2. **鱼体检测/分割双视图，专攻小鱼、主体不完整、手和复杂背景。** 保留原图分支，同时增加 detector crop/mask 分支；按 8k 人工问题标签分别报告小目标、截断、遮挡、多人/手、复杂背景的增益。只在每个问题组非负且总 OOF 至少 `+100/5795` 时加入 seen ensemble，避免 crop 破坏体型与尾鳍信息。
3. **多鱼实例聚合。** 对检测到的多个鱼实例分别分类，用“最大置信度 + 图像级先验”或轻量 attention 聚合；同图异种必须单列评估，不能把整图平均特征当作默认方案。
4. **质量/域路由器。** 将死亡标本、绘图、浑水、船摄、白底标本、颜色偏移作为显式域标签，训练只决定“使用哪个专家”的路由器，不直接预测 species。按 genus 分组 OOF，最差折非负才保留。
5. **停止纯阈值搜索。** 当前阈值与专家顺序已多轮自适应选择；继续在同一 2,001 张伪 unseen 图上搜索会放大选择偏差。下一次计算预算优先生成全新的独立图像伪划分或人工冻结验证集，再比较 v12/v15 与新方法。
