# Demo 流程图

```text
                [原始行情 / 历史数据]
                           |
                           v
              [featurework.py -> train_features.csv]
                           |
                           v
              [run_research_pipeline.sh]
                           |
        +------------------+-------------------+
        |                  |                   |
        v                  v                   v
   [train.sh]        [test.sh]         [backtest.py / diagnostics]
        |                  |                   |
        |                  v                   v
        |           [app/output/result.csv]  [回测与阶段分析结果]
        |                  
        +------------------+-------------------+
                           |
                           v
                 [run_submission.sh]
                           |
                           v
                [正式推理 / 冻结模型产物]
                           |
                           v
              [aggressive 变体结果同步]
                           |
                           v
                [freeze_submission.sh]
                           |
        +------------------+-------------------+
        |                                      |
        v                                      v
 [result_validator.py]                [pre_submit_check.py]
        |                                      |
        +------------------+-------------------+
                           |
                           v
                 [最终 result.csv / 提交产物]
                           |
                           v
                 [Docker build / compose up]
                           |
                           v
                   [容器内正式提交入口]
```

## 一句话说明

- 左半段是研究链路：训练、验证、回测、诊断
- 右半段是提交链路：正式推理、冻结、自检、Docker 复现
- 两条链路共享同一套输入口径和正式推理产物

## 老师视角的核心表达

> 这个项目不是“脚本堆叠”，而是把研究、提交和 Docker 复现串成了一条完整主流程。
