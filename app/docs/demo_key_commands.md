# Demo 关键命令页

## 1. 研究主流程

```bash
bash /app/run_research_pipeline.sh
```

用途：

- 训练模型
- 做 walk-forward 验证
- 生成冻结推理快照
- 做本地回测
- 输出诊断结果

## 2. 正式推理入口

```bash
bash /app/run_submission.sh
```

用途：

- 默认使用冻结模型
- 直接推理生成 `app/output/result.csv`
- 适合展示“正式提交主流程”

## 3. 冻结提交流程

```bash
bash /app/freeze_submission.sh
```

用途：

- 同步正式配置
- 运行推理
- 校验 `result.csv`
- 执行 `pre_submit_check`

## 4. Docker 演示

```bash
docker build -t bdc2026 .
docker compose up
```

用途：

- 展示本地流程和容器流程一致
- 说明项目具备可复现提交能力

## 5. 现场最推荐的演示顺序

```bash
bash /app/run_research_pipeline.sh
bash /app/run_submission.sh
bash /app/freeze_submission.sh
docker compose up
```

## 讲解提示

- 第一条命令讲“研究”
- 第二条命令讲“正式推理”
- 第三条命令讲“冻结提交”
- 第四条命令讲“Docker 复现”
