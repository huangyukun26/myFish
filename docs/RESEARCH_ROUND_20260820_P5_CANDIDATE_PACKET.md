# P5 Candidate Packet Handoff

更新：2026-08-20

## 交付内容

本轮 P5 完成“全量候选集交付”，不生成最终分类 prediction：

- split：`test_seen + test_unseen`
- 图片数：`35,665`（seen `20,097`，unseen `15,568`）
- 每张图片：top-5 和 top-10 候选物种
- 候选顺序：`candidate_1` 为当前正常 submission 的 base prediction，后续依次为候选源排名

本地文件（因包含完整测试行，按约定不上传 GitHub）：

- `runs/research_next_20260820/p5_candidate_packet_20260820/test_all_candidates_top5.csv`
- `runs/research_next_20260820/p5_candidate_packet_20260820/test_all_candidates_top10.csv`
- `runs/research_next_20260820/p5_candidate_packet_20260820/README_all.json`

CSV 字段格式：

```text
image_id,split,candidate_1,candidate_2,...,candidate_K
```

## 选择依据

seen 的候选补全来自现有最强可复用的 BioCLIP MLP top-K 文件；由于 7/16 joint 模型的完整 test logits 不在当前归档中，没有把旧 top-K 冒充成 joint 排名。为保证下游模型不丢掉线上答案，每行先放入当前 online-best 的 base prediction，再用该候选源补足到 top10。

unseen 的候选补全来自 P1 `fish_taxon_avg` top50 packet。P1 在固定 pseudo split 上测试时，species 三个 seed 的 top-1 净变化为 `-1/+7/+11`，genus 为 `+9/+18/+8`；它不是已经验证的 leaderboard 增益，但在现有候选生成器中比 description-only 和 visual-trait 分支更适合作为第二轮模型的候选池。

全量文件已做以下一致性检查：35,665 个 image_id 唯一；seen/unseen 行数正确；`candidate_1` 与当前 base prediction 全部一致；每行 top10 无重复；候选标签全部属于对应 split 的官方候选池。

## 使用限制

这不是最终 prediction，也不包含置信度或分数；下游模型应在每行的候选集合内重新打分。候选池覆盖完整 submission 的 seen 和 unseen 测试集。seen 的后续候选分数来自历史可复用 top-K 源，只有 `candidate_1` 是当前 online-best 的精确答案；下游模型若需要精确 joint top-K 排名，应在拥有对应 test logits 后重新导出。

P5 的最终提交仍需由下游模型生成完整 `prediction.json`，并在提交前通过当前 best/ancestor guard。
