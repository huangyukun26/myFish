# Seen 0.8+ 下一轮实验方案（2026-07-18）

## 1. 结论先行

本轮不生成 submission 是正确决定。DINOv3-L LoRA 单头已经失败；`frozen joint + adapted DINO` 只保留为一个小正信号，不再继续扫 LoRA rank、seed、MLP、ensemble 权重或 gate 阈值。

下一轮主线改为：

1. 先修复验证泄漏，建立按近重复图簇隔离的 paired-micro 评估；
2. 用 **PE-Core-L/14-336 frozen supervised** 获取与 BioCLIP、DINO 不同的新表示；
3. 不再只优化 standalone CE，而是用 **cross-fitted current-base logits + residual correction objective** 直接学习“纠正当前 0.780 包”；
4. PE frozen 明确互补后，才做 PE 上的 AdaptFormer；
5. 主视觉系统到约 0.793–0.797 后，才用 top-5 局部 rerank / NaFlex 宽图专家补最后缺口。

真正允许推理 test_seen 和生成 submission 的硬条件是：对当前规则模拟基线，grouped/nested OOF point net 至少 `+260/10790`，目标 `+300`，oracle complement 至少 `+324～350`，并能产生约 `2200～3000` 个新的 test_seen 改动。

## 2. 本轮 PEFT 的客观复盘

当前 public seen 为：

- `15685 / 20097 = 0.78046474598`
- 过 0.8 至少需要 `16078 / 20097`
- 仍差 `393` 张 public correct

最新 joint frozen+adapted 结果：

- current-rule-sim：`9823/10790 = 0.91037998`
- candidate：`9883/10790 = 0.91594069`
- raw changed：502
- wins/losses/net：`188 / 128 / +60`
- raw efficiency：11.95%
- oracle complement：`+188/10790 = 1.742pp`

关键判断：

- 即使知道每张 val 的真值并做完美 base/candidate 选择，该候选最多也只能增加 188 张；
- 直接按样本量换算，冲 0.8 至少需要约 `+211 val`；按上一轮 DINO 的实际 public/val 传递关系，需要约 `+258 val`；
- 因此当前候选在数学上就不足以承担 0.8，继续调 selector 也无法改变互补上限；
- 0.3/0.7 到 0.1/0.9 的 logit blend 仅得到 raw `+61～+66`，证明新旧信号高度相关。

DINO LoRA 本身也不值得续跑：

- direct head raw net `-208/10790`；
- val loss 在 epoch 3 最低 `0.59049`，epoch 4 已升到 `0.60163`；
- epoch 4 top1 只比 epoch 3 多 0.34pp，且学习率已到 0；
- 简单把 4 epoch 延长到 8–12 epoch，没有可信的放大依据。

## 3. 当前验证体系必须先修

### 3.1 已确认的问题

1. `train_embedding_mlp_classifier.py` 用完整 10,790 val 选择 best epoch，之后才做 dev/outer audit；outer 已参与 checkpoint selection。
2. `seen_current_base_tools.py` 的 dev/outer 只是同类内按 image_id 的 MD5 交替切分，不是 pHash/DINO 近重复簇切分。
3. 当前 dev 有 6,145 行、1,933 类，并吞掉全部 1,314 个 singleton 类；outer 有 4,645 行但只有 619 类，分布明显偏易。
4. gate trials 最后按 `outer.net` 排序，且只把 `outer.net > 0` 的 trial 用于打包；报告的“best gate +65、outer +29”存在 outer-peeking。
5. 当前 0.780 的 val base 只是 current-rule-sim，还未完整复现 7.2 的 391 行分支优先级。

因此本轮应暂时只认 raw `+60`，不能把 gate `+65` 当 sealed 证据。

### 3.2 新验证协议

在不开 GPU 的阶段完成：

1. 对 train+val 计算 pHash，并结合现有 DINO embedding 建近重复图簇；跨 train/val 的近重复簇单独标记，不能进入 final-sealed 主统计。
2. val 按图簇拆成：
   - selector-dev：约 60%；
   - family-holdout：约 20%；
   - final-sealed：约 20%。
3. 拆分同时约束 train-frequency、genus、AR/背景域桶；任何近重复簇不得跨集合。
4. epoch、视图、层、seed、head、gate 只能看 selector-dev。
5. 每个模型家族只允许一个锁定配置看 family-holdout；所有家族最终只允许一个 winner 看 final-sealed。
6. gate 采用 grouped 5-fold cross-fit：四折定阈值，一折统计，汇总 OOF；禁止按 outer 结果排序。
7. 完整重建 current-rule-sim：BioCLIP base、7.2 agreement branch、当前 frozen DINO branch及其优先级。
8. 重放历史符号控制：concat 正、7.2 正、v12 负、v20 负、5of6 负、当前 DINO-L 正。符号不能同时还原，不开昂贵 backbone。

主指标固定为相对 current-rule-sim 的 paired micro：`changed / wins / losses / net / efficiency / oracle complement`。macro、tail、AR、domain-hard 只作否决项，不参与主排序。

## 4. 阶段 1：PE-Core-L frozen supervised（下一轮云端第一优先级）

### 4.1 为什么是 PE-Core

它是尚未实验的新表示，不是继续加工 DINO。官方 PE-Core-L 是 336 分辨率、attention pooling 的大模型；官方研究还显示其中间层具有较强通用视觉信息。普通 SigLIP2-224 zero-shot 已失败，而 PE-Core supervised、PE 的官方视图与 native-aspect 对照均未做。

参考：

- PE 官方实现：https://github.com/facebookresearch/perception_models
- PE-Core-L14-336：https://huggingface.co/facebook/PE-Core-L14-336
- OpenCLIP/timm 权重：https://huggingface.co/timm/PE-Core-L-14-336
- PE 论文：https://arxiv.org/abs/2504.13181

### 4.2 固定实验，不做模型动物园

只缓存 train+val，不先推 test_seen。

模型与视图：

- backbone：PE-Core-L/14-336；
- view A：官方 336 squash；
- view B：336 native-aspect letterbox，padding 用模型归一化后的零色；
- 第一轮只用 final attention-pooled embedding；
- 同一次 forward 可顺手保存一个预锁定 penultimate 表示，但不做大规模 layer search。

依次只做四个固定对照：

1. PE squash standalone cosine/linear head；
2. PE letterbox standalone cosine/linear head；
3. `BioCLIP concat + frozen DINO-L + best PE view` joint head；
4. cross-fitted residual correction head。

所有实验 natural instance sampling、natural CE；禁止 Balanced Softmax、tail weighting、strong augmentation、broad threshold sweep。第一轮一个 seed。

### 4.3 关键新增：直接优化“纠正当前包”

上一轮 LoRA 的根本问题是优化 standalone CE，却与当前系统高度相关。下一轮使用现有缓存构造 5-fold OOF current-base train logits，再训练 PE residual：

`z_final = z_base_oof + alpha * r_PE(x)`

建议固定目标：

- 主损失：`CE(z_final, y)`；
- 对 OOF base-error 样本仅做温和 2x 权重，不能做类别平衡；
- 对 OOF base-correct 样本加入 trust/KL anchor，避免广泛破坏已正确预测；
- 加一个 `true class vs base wrong top1` 的 pairwise margin 项；
- alpha 在 selector-dev 只选一次；
- 最终重新以全 train 特征训练锁定配置。

这条线的目标是扩大 candidate/base oracle complement，而不是再把绝对 val top1 做高一点。

### 4.4 PE frozen 停止条件

- PE standalone 与 current base 的 oracle complement `<1.5pp`：PE 整条停止；
- locked joint/residual 在 honest grouped OOF `net < +120` 且 `oracle < +250`：不补 seed、不推 test；
- `+120～+259`：只算有效研究信号，可进入一次 PE AdaptFormer，但不能生成 submission；
- 进入 test_seen 的最低条件：
  - grouped/nested OOF net `>= +260`，目标 `>= +300`；
  - oracle complement `>= +324～350`；
  - 所有 fold 为正，worst fold 至少 `+0.3pp`；
  - cluster bootstrap 95% 下界至少 `+150 net`；
  - 有效 val changes 至少约 1,200，预计 test changes 2,200～3,000；
  - 净效率至少 18%，最后 500 行边际非负；
  - 训练样本数 >=50 的头类净正，3–9 张的尾类非负；
  - 单一 genus 不得贡献超过总 net 的 15%。

## 5. 阶段 2：只有 PE frozen 通过后才做 AdaptFormer

不要继续 DINO LoRA。若 PE frozen joint/residual 得到 `+120～+259` 且 oracle 足够大，则对胜出的 PE 视图做一次固定 AdaptFormer：

- 所有 transformer blocks 插入 bottleneck-16 adapter；
- adapter scale/末层初始化为 0；
- BF16、gradient checkpointing；
- micro-batch 2～4，梯度累积到有效 batch 32 左右；
- 6～8 epochs，natural sampling，label smoothing 0.05；
- 仅 hflip、轻尺度/平移；
- 第一 seed 通过后才补第二 seed；
- adapted PE 冻结后，再训练 residual/joint head；同时保留 frozen PE 作为独立输入。

AdaptFormer 必须相对 frozen PE 再增加 `+80～100 OOF net`，所有 fold 非负；否则停止。最终系统仍必须达到上一节的 submission 硬门槛。

参考：

- AdaptFormer 论文：https://arxiv.org/abs/2205.13535
- 官方仓库：https://github.com/ShoufaChen/AdaptFormer
- LIFT+ 的轻量微调证据：https://arxiv.org/abs/2504.13282

不要直接安装旧 AdaptFormer 环境污染当前依赖，只移植 adapter 模块。

## 6. 阶段 3：主视觉到 0.793–0.797 后才补最后缺口

最新 candidate 仍有 907 个 val top1 错误：

- true class 在 top-5 中的有 689 个；
- 405/907（44.7%）为同 genus 误判；
- 错误 top10 genera 只占 19.2%，不能做少数 genus 手工 router；
- AR>=2 只有 699 行、118 个错误，NaFlex 不能承担主目标。

因此补洞顺序是：

### 6.1 PE patch/local top-5 reranker

- 只在当前视觉 top-5 内比较；
- 使用 PE attention pool + 一个预锁定 patch-token GeM/局部表示；
- 训练 pairwise `true vs current wrong top1`，优先同 genus 对；
- 可以使用比赛提供的 species descriptions 作固定辅助特征；
- 不做全局 trait broad fusion，不做手工类别规则；
- 相对主视觉 grouped OOF 需额外 `>= +0.3～0.5pp`，worst fold 非负。

### 6.2 SigLIP2 NaFlex 宽图专家

仅处理 `AR>=2` 或 `padding>=45%` 的图：

- native-aspect、256 patches 起步；
- supervised head，不重跑普通 224 zero-shot；
- 宽图桶必须明显正，非宽图不得 broad override；
- overall sealed 至少 `+0.3pp` 才保留。

参考：

- Google 官方 SigLIP2/NaFlex 说明：https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/image_text/README_siglip2.md
- Hugging Face SigLIP2 文档：https://huggingface.co/docs/transformers/model_doc/siglip2

## 7. DINO E2 的地位

DINO 只允许一个有上限的对照，不是主线：

- 仅解冻最后 2 blocks；
- backbone LR `1e-6～3e-6`；
- 加 frozen-feature/logit anchoring；
- 比较 `BioCLIP+adapted DINO`、`frozen+adapted DINO` 和 late fusion；
- 一个 seed、固定 epoch；
- honest OOF `net < +120` 或 oracle `< +250`，立即永久停止 DINO PEFT。

只有 PE 环境临时不可用时才先跑 E2。不能再扫 LoRA rank/alpha/seed、MLP hidden/dropout、ensemble 权重或 margin gate。

## 8. 外部数据与规则边界

比赛规则允许 CLIP/OpenCLIP/DINO/SigLIP/BioCLIP 等通用 foundation models，但明确禁止为鱼类识别、鱼类专用数据或目标类别设计的模型/训练资源。

因此：

- **Fish-Vista 是鱼类专用数据，本方案移除，不得直接使用**；只有组织方书面明确许可后才可重新评估。
- Bio-DINO 属于广义 biodiversity 模型，不是鱼类专用，但其训练数据覆盖生物图像；若要使用，应先向组织方说明模型和训练数据并取得确认。
- 所有模型、权重、训练数据和微调过程必须保留可复现记录。

本地比赛规则快照：`work/codabench_16815_competition.json`。

## 9. 下一轮云资源执行顺序

### 云前本地门槛

- 完成 cluster split、nested/grouped evaluator、current-rule-sim 重建；
- 六个历史符号控制全部通过；
- PE-Core 权重和最小 forward 在本地/云镜像 dry-run；
- 所有命令、输出目录、停止条件写进单一 run manifest。

### 云 Run A：PE frozen scout

1. 缓存 train+val 的 squash、letterbox PE embedding；
2. 跑 standalone、joint、residual 四个固定对照；
3. 计算 grouped OOF 和 oracle complement；
4. 未达到 `+120 / oracle +250` 立即结束 PE，不推 test。

### 云 Run B：条件触发

- PE 达到 `+120～+259`：做一次 AdaptFormer；
- PE 已达到 `>=+260` 且统计门槛通过：先 final-sealed，再缓存 test_seen；
- 只有最终达到 submission-grade 才生成 S-main。

每轮必须带回：模型权重、train/val/test 特征摘要、逐行 logits、逐行 paired audit、cluster split、命令和环境锁定文件。禁止只带 summary。

## 10. 最终 submission 规则

- exact base：当前 `seen=0.78046474598` 的 prediction.json；
- unseen 完全不动，保持 `0.15082219938`；
- 第一枚包保守冻结 7.2 的 391 行分支和当前 DINO public-positive 分支，但不能把这些逐行当作“已知正确标签”；
- 第一枚包只引入 PE 这一种新机制；
- 新增 test_seen changes 目标 2,200～3,000；
- 离线预计 public net 至少 `+450`，为 0.8 留约 57 张缓冲；
- S-safe 可以预生成留作审计，但不因小正信号浪费一次提交；
- 若 S-main public 为负或只涨 <0.3pp，停止整个配置，不扫邻近阈值；
- 若涨到约 0.793～0.799，再用独立通过验证的 top-5/NaFlex 补洞；
- 达到 0.8 后冻结 seen 主线。

## 11. 明确禁止重复的实验

- DINOv3-L LoRA rank/alpha/seed 扩展；
- MLP hidden/dropout/epoch 网格；
- old/new logit 权重扫描；
- margin/consensus gate 扫描；
- Balanced Softmax、tail broad override；
- DINO metric router；
- broad foreground/five-crop/NaFlex；
- ordinary SigLIP2-224 zero-shot；
- trait 全局融合；
- 手工多个 genus router；
- Fish-Vista 外部训练。

客观预期：PE frozen 先验证是否存在 `+120～+250` 的新互补；真正越过 0.8，更可信的组合是“PE 新表示 + complementarity-aware residual training + 条件触发的 AdaptFormer”，必要时再加一个严格局部的 top-5/NaFlex 专家。若 PE 的 honest oracle complement 仍低于约 300，则应承认仅靠现有 target 数据和通用 frozen backbone 很难稳定过 0.8，而不是继续用阈值网格制造假进展。
