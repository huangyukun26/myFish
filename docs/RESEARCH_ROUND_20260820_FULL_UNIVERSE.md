# FishNet 2026-08-20 honest split and full-universe round

本轮只复用官方图像、BioCLIP2.5 frozen image features 和官方 descriptions/text caches；未生成 submission。

## 结果摘要

线上 base 仍为 overall `0.505313`、seen `0.779917`、unseen `0.150822`。

### Seen validation repair

旧 sealed 的 3,346 行只覆盖 278/1,933 个验证类，单例类为 0；它不能代表公开测试。新协议用 exact BLAKE2 + pHash-Hamming 1 的 group assignment，5 个固定 seed，避免近重复跨 panel。每个新 sealed 覆盖 1,244–1,305 个类，并保留 singleton、2-shot 和长尾 bucket。

冻结 expert bank 的 `orig:old_crossfit:final_scores` 在 seed 42–46 honest sealed 上净值为 `+20,+15,+14,+1,+13`；最差折仍为正。每折事后挑最优 expert 虽为 `+20,+19,+15,+5,+15`，但不作为规则，因此没有按 seed 选择 gate，也没有产生 seen overlay。

### Full 17,393-way retrieval

`64,259` 张官方训练图作为 query，候选池固定为全部 `17,393` 官方类别；species/genus 各 5 折，未使用 held-out prototype、分类头或适配样本。

| expert | species top-1 | species top-50 | genus-fold top-50 最低 |
|---|---:|---:|---:|
| taxon | 62.32% | 90.54% | 88.70% |
| fish | 60.87% | 90.02% | 87.70% |
| taxon+fish union | 65.43% | 91.67% | 90.09% |

全规模 top-50 远高于 60%，所以 unseen 的主瓶颈不是 retrieval。旧 1,000-query proxy 在相同 17,393-way pool 的 top-50 只有 taxon 76.2%、fish 76.7%、union 80.5%，不能当作真实规模代理。

### P3/P4 gate

- Logistic candidate reranker：species 总净值 +79、genus +88，但 species 最差折 -33，bootstrap 下界 -74；不通过。
- 两层小 MLP：species 总净值 -187、genus -92；不通过。
- Expert selector：species 五折 `+37,+94,-17,+42,-25`；genus `-105,+34,+4,+57,+12`；多折 bootstrap 下界低于 0；不通过。

因此停止 unseen overlay，不跑 P5 新 encoder，不生成 submission。真实原因是开放集域偏移下的 top-1 选择/校准不稳定，而不是 top-K 候选召回不足。

## 代码与复核文件

- `tools/build_honest_seen_audit_split.py`
- `tools/replay_seen_expert_bank_honest.py`
- `tools/build_full_universe_pseudo_unseen.py`
- `tools/evaluate_full_universe_recall.py`
- `tools/build_candidate_training_table.py`
- `tools/train_group_aware_candidate_reranker.py`
- `tools/train_group_aware_expert_selector.py`

本地生成结果（被 `.gitignore` 排除，不上传数据/权重/大结果）：

- `runs/research_next_20260820/honest_seen_phash1/replay/replay_summary.json`
- `runs/research_next_20260820/honest_seen_phash1/replay/expert_replay.csv`
- `runs/research_next_20260820/honest_seen_phash1/replay/honest_sealed_best_net_by_seed.png`
- `runs/research_next_20260820/full_universe/manifest_summary.json`
- `runs/research_next_20260820/full_universe/recall_summary.json`
- `runs/research_next_20260820/full_universe/recall_metrics.csv`
- `runs/research_next_20260820/full_universe/proxy_1000_vs_full_universe_recall.png`
- `runs/research_next_20260820/full_universe/candidate_union_gain.png`
- `runs/research_next_20260820/full_universe/reranker_logistic/reranker_summary.json`
- `runs/research_next_20260820/full_universe/reranker/reranker_summary.json`
- `runs/research_next_20260820/full_universe/selector/selector_summary.json`
