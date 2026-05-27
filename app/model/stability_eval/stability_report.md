# RankIC Stability Report

## Required Answers

1. Fold 1 和 Fold 3 是否是主要不稳定来源？是。Fold 1 rank_ic=-0.033492, Fold 3 rank_ic=-0.001508.
2. 当前模型是否属于“收益正但 RankIC 不稳”？是。 top5_return_mean=0.007502, worst_fold_rank_ic=-0.033492。
3. 后续实验采用与否的硬性门槛：top5_return_min_by_fold > 0；worst_fold_rank_ic 不低于当前主线；negative_day_rank_ic_ratio 下降。
4. 推荐门槛已写入本报告，并应同步用于 experiment_leaderboard 的 adopted 判断。

## Summary

| model | feature_set | sl | rank_ic_mean | rank_ic_std | worst_fold | neg_folds | neg_day_ratio | top5_mean | top5_min_fold | stability_score |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lstm | base_alpha_v3_rs_crowding_mini4 | 20 | 0.027982 | 0.065633 | -0.033492 | 2 | 0.400000 | 0.007502 | 0.003244 | -0.037332 |
