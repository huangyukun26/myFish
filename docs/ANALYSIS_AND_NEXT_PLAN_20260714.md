# Fish Species Recognition Challenge：瓶颈复盘与下一轮实验方案

更新：2026-07-14

## 1. 结论先行

当前最佳 public 基线应修正为 2026-07-02 的组合包：

- submission：`runs/submission_20260702_seen_router_unseen_pair_o70species_avg_letterbox/submission.zip`
- seen：`0.765437627506593`，约 `15,383 / 20,097` 正确
- unseen：`0.15082219938335045`，约 `2,348 / 15,568` 正确
- overall：`0.4971540726202159`

目标 `seen=0.85, unseen=0.25` 对应：

- seen 还需要约 `+1,700` 个正确样本；
- unseen 还需要约 `+1,544` 个正确样本；
- 目标 overall 约为 `0.58810`，总共还需要约 `+3,244` 个正确样本。

因此，目标不能靠当前管线上的小权重、prompt 或几十到几百行 selective override 实现。下一轮必须同时改变：

1. public-like 验证设计；
2. 图像表示，尤其是 native-aspect / fine-grained / local-region 表示；
3. unseen 的概率空间 transduction 和类别语义表示；
4. 根据图像质量与拍摄域进行条件路由，而不是全局平均多视图。

今天没有生成 submission 不是 zip 命令故障。`work/cloud_20260713` 下的新 runner 均止于缓存、OOF gate、top-K 或预测分支，没有任何 submission assembly 调用；最新 run log 也明确规定在证据不足时不产包。仓库内真正负责打包的是 `tools/override_submission_with_csv.py` 和 `tools/override_submission_with_submission_split.py`。

## 2. 数据集的决定性结构

### 2.1 规模与极端长尾

- labeled train：`64,259` 图；
- test_seen：`20,097` 图；
- test_unseen：`15,568` 图；
- seen classes：`5,795`；
- unseen candidate classes：`11,598`；
- 总 competition classes：`17,393`。

训练类图像数中位数只有 `2`：

| 训练图数 | 类别数 |
|---|---:|
| 2 | 3,181 |
| 3-5 | 1,418 |
| 6-10 | 651 |
| 11-20 | 284 |
| 21-50 | 58 |
| 51+ | 203 |

旧的 image-distribution validation 只覆盖 `1,933 / 5,795` 类，约 `0.89` 的本地 top-1 会系统性高估 public seen。新的 one-query-per-class gate 才真正暴露 count-2 类问题。

### 2.2 unseen 有非常明显的宽幅域偏移

利用已经返回的 DINO foreground box 元数据，对所有三套数据读取原始宽高：

| split | 宽高比中位数 | `AR>=1.5` | `AR>=2.25` | 竖图 `AR<0.75` |
|---|---:|---:|---:|---:|
| labeled | 1.465 | 44.72% | 4.59% | 3.47% |
| test_seen | 1.499 | 49.46% | 6.73% | 2.87% |
| test_unseen | 1.744 | 73.32% | 22.60% | 0.33% |

这解释了 letterbox 为什么是迄今最确定的 public 正增益：test_unseen 的极宽图比例约为训练集的 `4.9` 倍，默认 shortest-side resize + center crop 会频繁截掉鱼体头尾和整体轮廓。

### 2.3 8k 图人工审查的正确用法

人工审查归纳了以下问题：

- 主体：不完整、过小、死亡/骨架、位置不明；
- 数量：同种多鱼、异种鱼混入；
- 环境：保护色、复杂背景、花纹/纹理干扰、低清、色偏、插画、浑水、水上/水下拍摄；
- 人为干扰：手、半身或整个人。

这份结果目前是“问题分类体系”，还不是可学习的统计数据，因为没有每类频率和 image ID 标签。不能据此删除图片或直接设置 router prior。最高价值的后续动作是保留 image ID，并给 `1,500-2,000` 张分层样本打多标签；至少包括 `主体尺寸、主体完整度、多鱼、背景复杂度、照片/插画、浑水/水下、人体干扰、模糊、色偏`。

## 3. 7 月 2 日最佳包的真实增益归因

相对上一版 public-positive 组合：

- seen router：只改 `391` 行，seen 从 `0.762850` 到 `0.765438`，净增约 `52` 个正确样本，平均每个 changed row 的净收益约 `13.3%`；
- unseen consensus：相对 letterbox 改 `3,512` 行，unseen 从 `0.142793` 到 `0.150822`，净增约 `125` 个正确样本，平均每个 changed row 的净收益约 `3.56%`。

这说明：

- seen 的小路由仍有价值，但规模不足以支持 `+8.46` 个点；
- unseen 的 broad change 已进入明显的低净收益区，继续放宽 consensus 会快速增加损失；
- 后续 public submission 必须分 seen-only / unseen-only 做因果归因，不再直接提交多分支大组合。

## 4. 7 月 13-14 日实验的有效结论

### 4.1 Seen

全 5,795 类 random-2027 gate：

| 方法 | top-1 |
|---|---:|
| hflip Balanced Softmax MLP | 0.467645 |
| hflip + letterbox concat Balanced Softmax | 0.525453 |
| concat + structured/text/prototype fusion | 0.537187 |
| concat strong base + full-support DINO metric router | 0.559103 |

DINO router 相对 strong concat base 的净增为 `+127 / 5,795`，四个 genus-grouped OOF fold 全正，且主要恢复 count-2 类。这是今天最值得进入 public seen-only inference 的信号。

Foreground 三视图在 raw MLP 上有 `+39` 个正确样本，但在 text fusion + DINO router 后只剩 `+1` 到 `+11`，说明信息高度重叠；不应把 foreground 当作全局第四个平均分支。

### 4.2 Unseen

最强稳健新信号是：

```text
0.95 BioCLIP taxon + 0.05 cleaned structured trait mapping
+ move-from-known-genus-to-novel-genus taxonomy router
```

相对当前 avg-H8192，在 species42/43/44 和 genus42/43/44 六个 gate 上均为正，总计 `+103 / 6,003`，约 `+1.72` 个点，worst split `+10` 个正确样本。

该结果足以生成一个 unseen-only 校准包，但不足以解释从 `0.1508` 到 `0.25` 的十点缺口。

Foreground unseen 全局融合最多只有 `+8 / 6,003`，且 worst split 为 `-3`；更高权重明显为负。正确方向是质量/宽高比条件路由，而不是全局 crop average。

### 4.3 类覆盖压力测试

当前最佳包只预测：

- seen：`4,464 / 5,795` 个不同类；
- unseen：`7,188 / 11,598` 个不同类。

官方说明将 `all_classes.pkl` 描述为全部 seen+unseen 类，因此应向主办方确认：每个 unseen class 是否至少在 test_unseen 中出现一次。

本地新增压力测试结果：

1. one-query-per-class、5,795 类均激活，BioCLIP taxon raw argmax 为 `0.2721`；一对一全局匹配为 `0.3619`，提升 `+8.97` 点；
2. 按 public 的 `N/K=15,568/11,598=1.3423` 构造 active-complete gate，三个 seed 的 raw argmax 约 `0.3003`，minimum-one assignment 为 `0.3473-0.3481`，稳定提升约 `+4.7-4.8` 点；
3. 但先应用当前 H8192 soft Sinkhorn 后，top-1 已到 `0.3791`，再强制 minimum-one 下降到 `0.3763`；只应用最小代价的一半覆盖改动最好为 `0.3810`，仅约 `+0.19` 点。

结论：类激活信息是真信号，但当前 Sinkhorn 已吸收大部分收益。硬 minimum-one 不是首要 public 候选；只有主办方确认全类出现，且新的 soft coverage regularizer 在多 seed、多 genus gate 全正后才进入提交队列。

## 5. 新实验主线

### 5.1 P0：重建 public-like 验证协议

优先级：最高，先于新模型。

建立四组固定 gate：

1. `seen-all-class`：每类一张 query，三 seed；
2. `seen-domain-hard`：每类一张最接近 test_seen 域的 query；
3. `unseen-active-complete`：held-out class 全激活，query/class 比固定为 `1.3423`；
4. `unseen-genus-novel`：整属 holdout，并分别统计 known-genus / novel-genus。

每个候选必须报告：top1/top5/top20、changed/wins/losses、class coverage、known/novel genus、宽高比桶、质量桶、worst fold。

### 5.2 U1：用 Dirichlet probability-space transduction 替换/补充 Sinkhorn

优先级：unseen 第一。

当前方法在余弦 logit 上做 SwAV-style balancing；Transductive Zero-Shot CLIP 的核心是先把 image-text 得分变成概率单纯形特征，再用 Dirichlet 分布联合估计类分布与 assignment。这比继续扫 Sinkhorn `tau/blend/prior` 更有结构差异。

实验输入固定为：

- image：hflip、letterbox、两者平均；
- class：BioCLIP taxon、taxon95+structured05；
- 不做 active pruning；
- 先在六个已有 species/genus gate 和 active-complete gate 上跑原论文算法；
- 只有 worst split 非负且平均至少 `+1.5` 点才做 public inference。

论文：<https://openaccess.thecvf.com/content/CVPR2024/papers/Martin_Transductive_Zero-Shot_and_Few-Shot_CLIP_CVPR_2024_paper.pdf>

### 5.3 U2：加入真正互补的通用编码器

优先顺序：

1. SigLIP2 NaFlex L/16 或 So400m：native aspect ratio 与更好的 dense/local features 正好对应 unseen 宽图和主体定位问题；当前只准备了 Base/256-patch 路径，尚未完成有效 gate；
2. FG-CLIP Large：通用 fine-grained 与 region/text alignment，不是鱼类专用模型，适合做 top-K candidate expert；
3. DINOv3 CLS：保留为视觉结构/metric expert，不作为直接 zero-shot text scorer。

第一阶段不看 standalone top1，先看它是否提高 BioCLIP union top20/top50 recall，尤其 genus holdout。通过条件：六个 gate 的 top20 recall 至少五个为正、worst 不低于 `-0.2` 点，且 genus gate 平均至少 `+2` 点。

资料：

- SigLIP2：<https://arxiv.org/abs/2502.14786>
- FG-CLIP：<https://arxiv.org/abs/2505.05071>
- BioCLIP2/2.5：<https://github.com/Imageomics/bioclip-2>

### 5.4 U3：从“整段描述 embedding”改成“判别式多属性匹配”

当前 structured trait 只允许 5% 权重，说明信息存在，但全局单向量与图像细节未对齐。

新做法：

1. 每类拆成多个原子属性组：body shape、head/mouth、fin/tail、color/pattern、scale/texture、habitat/view；
2. 去除否定、纯计数、不可见测量和重复 evidence；
3. 对候选 top20 内同属/近邻类生成 pairwise discriminative attributes，而不是 11.6k 类全局长描述；
4. image 使用 full letterbox + foreground/local patches；
5. 在 seen species/genus OOF 上训练 multi-instance attribute-to-visual adapter，并使用 sibling hard negatives；
6. 最终只在 candidate union top20 内 rerank。

这条线与 AdaptCLIPZS、Real Classification by Description、UniFGVC 的共同结论一致：测试时简单拼长描述通常不够，需要属性-视觉对齐、局部/多分辨率表示或 reference-guided 判别描述。

### 5.5 U4：一轮保守 GTA-CLIP

在 U1/U2/U3 建立更强 candidate recall 后，再做一次 Generate-Transduct-Adapt：

- 复用已经清洗的官方描述和 structured traits，不重新引入鱼类专用外部数据；
- 只用三分支高置信且跨视图一致的 pseudo labels；
- 不删除任何候选类；
- 只训练轻量 text/image projector，一轮后停止；
- 每轮必须在 genus holdout 独立评估，防止 pseudo-label collapse。

GTA-CLIP 在论文中相对普通 CLIP 和 transductive CLIP 有明显平均提升，但这里有 11.6k 类、每类约 1.34 图，不能直接套用论文增益预期。

论文：<https://openaccess.thecvf.com/content/ICCV2025/html/Saha_Generate_Transduct_Adapt_Iterative_Transduction_with_VLMs_ICCV_2025_paper.html>

### 5.6 S1：部署 DINO long-tail router，先做 seen-only public 探针

优先级：seen 第一，当前已有证据最完整。

保持 7 月 2 日 public seen prediction 作为 base，不重新 broad replace：

```text
仅当两个 DINO metric seed 同意
且 alternate 类训练图数正好为 2
且 concat base margin 位于低 70%
时替换
```

必须先在 public test_seen 上复现 concat scorer margin，并审计：changed count、alternate class frequency、两个 seed agreement、与 7 月 2 日 391-row router 的重叠。只生成 seen-only 包；如果 changed 过宽或无法复现 base scorer，则不提交。

合理期望是校准是否能带来约 `+0.5` 到 `+1.5` seen，而不是直接到 `0.85`。

### 5.7 S2：质量条件 Mixture-of-Experts

这是 8k 图审查最应转化出的模型。

Experts：

- current hflip+letterbox concat head；
- DINO metric tail expert；
- foreground crop expert；
- SigLIP2 NaFlex expert；
- 可选 grayscale/color-jitter robust expert。

Gate features：

- 人工/通用 VLM 质量多标签；
- aspect ratio、原始分辨率、crop area、fallback；
- 每个 expert 的 margin、entropy、top-K agreement；
- class train frequency；
- known/novel genus mass。

Cross-fitting 规则：expert 与 gate 不能在同一 query 图上拟合和评估；至少按 genus 分四 fold。重点学习条件：

- 极宽、完整长鱼：full letterbox / NaFlex；
- 小鱼、复杂背景、手/人干扰：foreground/local expert；
- 多鱼：实例级候选聚合；
- 插画、标本、死亡鱼：domain-specific prototype/text expert；
- count-2：DINO metric expert。

### 5.8 S3：多鱼与异种混入专线

不要把 detector crop 全局用于所有图。对训练图运行通用 detector/segmenter得到多个 proposal：

1. 用已知训练标签判断哪个 proposal 最能恢复全图标签，训练 target-fish selector；
2. 同种多鱼用 proposal logsumexp/mean；
3. 异种混入用 selector 选择主目标，并保留全图上下文分支；
4. 只在自动 multi-fish confidence 高的图上启用。

这比单框 foreground crop 更贴合人工审查中的第二大问题。

## 6. 实验与提交顺序

### 第一阶段：1-2 天，利用已有缓存

1. 完成 P0 active-complete / domain-hard gates；
2. 复现并生成 DINO seen router 的 public CSV；
3. 生成 structured+novel-genus unseen public CSV；
4. 移植 Transductive CLIP，跑六个旧 gate + active-complete gate；
5. 只在通过 gate 后打包。

建议 public 提交顺序：

1. seen-only DINO router；
2. unseen-only 最强新 transduction；如果 U1 尚未通过，则先用 structured+novel-genus router 做低风险校准；
3. 只有前两项分别 public-positive 后才提交 combo。

### 第二阶段：3-5 天，新编码器与质量路由

1. SigLIP2 NaFlex；
2. FG-CLIP；
3. candidate union recall；
4. 质量标签与 cross-fitted MoE；
5. discriminative multi-attribute rerank。

### 第三阶段：通过前两阶段后

1. 保守一轮 GTA-CLIP；
2. multi-fish selector；
3. soft coverage regularizer；
4. 最终分支 ensemble 与 reproducibility audit。

## 7. Submission assembly

当某一新分支已经有完整 public CSV 后，以 7 月 2 日最佳包为 base：

```powershell
python tools\override_submission_with_csv.py `
  --base runs\submission_20260702_seen_router_unseen_pair_o70species_avg_letterbox\submission.zip `
  --override-csv <new_branch_predictions.csv> `
  --out-dir runs\submission_20260714_<branch_name>
```

生成后必须检查：

- zip 根目录只有 `prediction.json`；
- 总 key 数 `35,665`；
- seen-only 包 unseen diff 必须为 0；
- unseen-only 包 seen diff 必须为 0；
- 所有 label 属于对应候选集合；
- changed、unique labels、max class count、known/novel genus 比例；
- 保存 base、命令、SHA256 和 public 结果。

## 8. 对目标的客观判断

- `seen=0.85`：当前证据不支持靠 DINO router 或三视图单独达到；需要 DINO tail + 新 fine-grained/native-aspect encoder + 质量 MoE 同时产生互补增益。短期更可信的阶段目标是 `0.79-0.81`，之后再判断是否有到 `0.85` 的斜率。
- `unseen=0.25`：比 seen 更难。当前 structured/novel-genus 只有约 `+1.72` pseudo 点。要接近 `0.25`，至少需要新 transduction、互补 encoder、判别式属性对齐三条线同时有效；`0.20-0.22` 是更合理的第一阶段目标，`0.25` 是 stretch target。

不建议继续：

- active class pruning；
- broad foreground/crop average；
- 无 trigger 的 Qwen/VLM override；
- 只改 prompt 或描述权重；
- 继续放宽 7 月 2 日 consensus；
- 用旧的 1,933-class 高分 gate 决定 public submission。
