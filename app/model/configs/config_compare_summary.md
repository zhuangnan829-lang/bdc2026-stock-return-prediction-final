# Aggressive vs Robust Configs

This directory separates two valid but different operating goals.

## Use Cases

| Config | Use when | Do not use when |
|---|---|---|
| `submission_aggressive.json` | Competition score chasing, single-submission leaderboard runs, and explaining the highest-return default. | The discussion is about practical stability, lower turnover, or lower single-name exposure. |
| `submission_robust.json` | Stability demonstrations, risk review, and practical deployment discussion. | The only objective is maximizing one-shot competition score. |

## Config Difference

| Dimension | Aggressive | Robust |
|---|---:|---:|
| candidate_size | 180 | 100 |
| risk_penalty_weight | -0.30 | -0.25 |
| weighting_scheme | pred | pred_equal_blend |
| weight_blend_alpha | 1.00 | 0.25 |
| max_single_weight | 0.18 | 0.16 |
| max_turnover | 1.00 | 0.65 |
| use_previous_result_when_available | false | true |

## Evidence

- Candidate pool search: `cs180 + rp=-0.30` ranked `2/35` by return and `2/35` by Sharpe, so it remains the aggressive score-chasing profile.
- Candidate pool search: `cs100 + rp=-0.25` ranked `1/35` by composite score, with better drawdown behavior, so it anchors the robust profile.
- Weight blend search: `alpha=0.25` is the robust pred/equal mix because it keeps most of the return while reducing reliance on raw prediction scale.
- Weight cap search: lower caps reduce drawdown and contribution concentration; robust uses `0.16`, aggressive keeps the current `0.18`.

## Decision Rule

- For competition submission and score comparison, use `submission_aggressive.json`.
- For stability charts, risk discussion, and deployment-style analysis, use `submission_robust.json`.

The two configs should not be averaged or treated as one default. They answer different questions: aggressive asks "what scores best now?", while robust asks "what is easier to defend under volatility, turnover, and concentration risk?"
