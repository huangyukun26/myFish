# P5 Candidate Packet Handoff

更新：2026-08-20

## 交付内容

本轮 P5 先完成“候选集交付”，不生成最终分类 prediction：

- split：`test_unseen`
- 图片数：`15,568`
- 每张图片：top-5 和 top-10 候选物种
- 排序来源：P1 `fish_taxon_avg` candidate ranking
- 候选顺序：`candidate_1` 为 rank 1，依次递减

本地文件（因包含完整测试行，按约定不上传 GitHub）：

- `runs/research_next_20260820/p5_candidate_packet_20260820/test_unseen_candidates_top5.csv`
- `runs/research_next_20260820/p5_candidate_packet_20260820/test_unseen_candidates_top10.csv`
- `runs/research_next_20260820/p5_candidate_packet_20260820/README.json`

CSV 字段格式：

```text
image_id,candidate_1,candidate_2,...,candidate_K
```

## 选择依据

P1 在固定 pseudo split 上测试了 `fish_taxon_avg`。species 三个 seed 的 top-1 净变化为 `-1/+7/+11`，genus 为 `+9/+18/+8`；它不是已经验证的 leaderboard 增益，但在现有候选生成器中比 description-only 和 visual-trait 分支更适合作为第二轮模型的候选池。

## 使用限制

这不是最终 prediction，也不包含置信度或分数；下游模型应在每行的候选集合内重新打分。候选池只覆盖 unseen 测试集。当前没有把历史 seen top-K 文件混入，避免不同版本模型和验证协议造成候选污染。

P5 的最终提交仍需由下游模型生成完整 `prediction.json`，并在提交前通过当前 best/ancestor guard。
