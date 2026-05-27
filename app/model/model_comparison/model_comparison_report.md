# 统一模型对比报告

## 对比口径

- 特征集：`base_alpha_v3_rs_crowding_mini4`
- 训练目标：`cross_section_rank`
- 选股逻辑：`risk_adjusted sort + pred weight + max_turnover=1.00`
- 回测口径：统一使用当前正式默认风险过滤与执行约束

## 总结论

- 当前同口径下综合表现最优模型：`LSTM sl20`
- 成本后累计收益：`1.171246`
- 成本后夏普：`4.019488`
- 成本后最大回撤：`-0.090067`
- 平均换手率：`0.956671`

## 模型排序

| 排名 | 模型 | rank_ic_mean | top5_mean_return_mean | cumulative_return_after_cost | sharpe_after_cost | max_drawdown_after_cost | avg_turnover | 是否最优 |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | LSTM sl20 | 0.027982 | 0.007502 | 1.171246 | 4.019488 | -0.090067 | 0.956671 | 是 |
| 2 | LightGBM | -0.027417 | 0.002183 | 0.295839 | 1.711682 | -0.113166 | 1.000000 | 否 |
| 3 | Transformer | 0.008137 | 0.003885 | 0.243178 | 1.537092 | -0.084388 | 0.994547 | 否 |
| 4 | XGBoost | -0.030145 | 0.000341 | 0.137837 | 1.045790 | -0.097146 | 1.000000 | 否 |
| 5 | Linear Regression | -0.028608 | 0.001038 | -0.041794 | -0.138544 | -0.186285 | 0.976856 | 否 |

## 结果解读

- `rank_ic_mean` 和 `top5_mean_return_mean` 反映 walk-forward 预测排序能力。
- `cumulative_return_after_cost`、`sharpe_after_cost`、`max_drawdown_after_cost`、`avg_turnover` 反映同一执行逻辑下的真实组合表现。
- 如果某个模型回归误差不差，但成本后收益明显弱，通常说明它在 Top-K 排序稳定性上不如更优模型。
