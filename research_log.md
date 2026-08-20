# FishNet Research Log

## 2026-08-20 — P0/P1 restart

- **假设**：seen 停滞主要来自已有专家高度相关和验证选择偏差；unseen 需要先判断真类是否进入 top-20/50，不能盲目换大模型。
- **失败现象（既有证据）**：普通多 crop、foreground、SigLIP2 zero-shot、简单文本/Dirichlet 融合没有稳定收益；当前 public 约 seen 0.78、unseen 0.15。
- **原因判断**：seen 的 76 通道候选 bank 已有 dev/sealed 分割但互补上限有限；unseen 的 BioCLIP prompt 与结构化分支需要在固定 species/genus pseudo 上重新量化召回。
- **修改内容**：新增本轮独立目录、实验计划和 P0 诊断入口；只读复用现有 logits/features/text caches，不改当前最佳 baseline。
- **结果变化**：待 P0 运行后填写。
- **是否保留**：保留诊断和所有原始 baseline；新候选需通过 sealed 条件后才保留。

## 2026-08-20 — Honest seen + full-universe unseen gate

- **验证修复**：`tools/build_honest_seen_audit_split.py` 使用 exact BLAKE2 + pHash-Hamming 1 的连通组件，5 个固定 seed 生成 grouped dev/sealed。旧 sealed 只有 278/1,933 个验证类且 1-shot/2-shot 类为 0；新 sealed 每 seed 覆盖 1,244–1,305 类，并同时覆盖 singleton、2-shot 和长尾类。最大 pHash 组件 204 行，已在 split summary 中单独记录。
- **冻结 seen bank 重放**：固定 `orig:old_crossfit:final_scores` 在 5 个 honest sealed fold 的净值为 +20、+15、+14、+1、+13（按 seed 42–46），最差仍为正；每折事后挑最优 expert 虽为 +20、+19、+15、+5、+15，但不作为规则。因此 seen overlay 暂不扩展，不按 fold 选 gate、不生成 submission。
- **全宇宙 recall**：`build_full_universe_pseudo_unseen.py` 构造 64,259 张官方 query、5,795 species、2,065 genera，候选固定为全部 17,393 官方类，无 held-out prototype/head/adaptation。taxon/fish/union 的 species-held-out top-1 为 62.32/60.87/65.43%，top-50 为 90.54/90.02/91.67%；genus-held-out 各 fold top-50 最低为 88.70/87.70/90.09%。因此 unseen 主瓶颈不是 top-50 retrieval。
- **旧 pseudo 对照**：同一 17,393-way pool 上，1,000-query proxy 的 taxon/fish/union top-50 为 76.2/76.7/80.5%，明显低于全规模 query；旧 11,598-way pseudo 不能外推真实候选空间。
- **P3 candidate reranker**：group-aware logistic 5-fold 总净值 species +79、genus +88，但 species fold 最差 -33、bootstrap 下界 -74；两层 MLP species -187/genus -92。均未达到“所有 fold 非负、bootstrap 下界 >=0”。
- **P4 expert selector**：只使用 margin/entropy/top1/overlap/cross-rank。species 净值 [+37,+94,-17,+42,-25]，genus [-105,+34,+4,+57,+12]，多折 bootstrap 下界为负。停止 unseen overlay。
- **结论**：真实停滞原因是开放集域偏移下的 top-1 选择/校准，不是候选召回不足；固定轻量 rerank/selector 不稳定。P5 新 encoder 不触发（full top-50 >60%），不继续扫 alpha/beta、description 或大模型，不生成 submission。
