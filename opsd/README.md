# OPSD (Open Source Distillation)

基于 TRL 的类 GKD 自蒸馏训练框架，支持 vLLM 高速生成、多种教师策略（固定教师 / EMA / 动态）和可插拔的数据整理器。

## 目录结构

```
opsd/
├── scripts/                          # 训练启动脚本
│   ├── run_sft_*.sh                  # SFT 训练脚本（1B / 4B / 8B）
│   ├── run_opsd_*.sh                 # OPSD 自蒸馏训练脚本
│   └── run_opsd_1b_combine.sh        # combine 模式训练脚本
│
├── data/                             # 数据集（Parquet 格式）
│   └── *.parquet
│
├── sft_train.py                      # SFT 训练入口
├── opsd_train.py                     # OPSD 主训练入口
├── opsd_train_answer.py              # answer 模式训练入口
├── opsd_train_combine.py             # combine 模式训练入口
│
├── opsd_trainer.py                   # 核心 Trainer（OPSD）
├── opsd_trainer_answer.py            # answer 专用 Trainer
├── opsd_trainer_combine.py           # combine 专用 Trainer
│
├── data_collator.py                  # 标准数据整理器
├── data_collator_answer.py           # answer 专用数据整理器
├── data_collator_combine.py          # combine 专用数据整理器
│
└── accelerate.yaml                   # Accelerate 分布式配置
```

## 核心模块说明

| 模块 | 说明 |
|------|------|
| `opsd_trainer.py` | 核心自蒸馏 Trainer，继承自 `SFTTrainer`，实现 JSD 损失、KL 散度计算、vLLM 同步、EMA / 固定教师策略 |
| `data_collator.py` | 构造学生/教师输入对，支持 `reason_first` 模式（教师先推理再教学） |
| `opsd_train.py` | 训练入口脚本，负责模型加载、数据集读取、Trainer 初始化与训练 |
| `scripts/` | 按模型规模组织的启动脚本，包含完整的超参数配置 |

### 关键特性

- **vLLM 集成**：支持 `colocate` 模式，训练与生成共享 GPU，大幅提升采样效率
- **教师策略**：
  - `fixed_teacher`：固定初始策略作为教师（需配合 PEFT）
  - `use_ema_teacher`：EMA 平滑学生权重作为教师
  - 动态教师：默认使用当前学生模型作为教师
- **损失函数**：广义 Jensen-Shannon 散度（JSD），支持 `top_k` 裁剪和 `token_clip` 稳定训练
- **日志系统**：支持 `wandb` / `swanlab` / `none` 三种日志模式

## 运行方式

### 1. 环境准备

```bash
# 安装依赖（示例）
pip install torch transformers trl accelerate peft vllm
```

### 2. 修改脚本配置

编辑 `scripts/` 下对应脚本，替换以下占位符：

```bash
--model_name_or_path "<YOUR_MODEL_PATH>"   # 模型路径
--output_dir "<YOUR_OUTPUT_DIR>"           # 输出目录
--dataset_path "<YOUR_DATASET_PATH>"     # 数据集路径（支持多文件，逗号分隔）
```

### 3. 启动训练

```bash
# SFT 训练（以 8B 模型为例）
bash scripts/run_sft_8b.sh

# OPSD 自蒸馏训练（以 8B 模型为例）
bash scripts/run_opsd_8b.sh

# 其他规模类似，替换对应脚本即可
```

### 4. 数据集格式与路径配置

数据集为 Parquet 格式，每行包含：

```json
{
  "problem": "题目内容...",
  "solution": {
    "reference_answer": "参考答案...",
    "rubrics": [...]
  }
}
```

**数据集路径配置**：
- 在脚本中通过 `--dataset_path` 参数指定数据集路径
- 支持传入多个文件，使用逗号分隔：
  ```bash
  --dataset_path "/path/to/data1.parquet,/path/to/data2.parquet"
  ```
- 如果不指定 `--dataset_path`，脚本会抛出 `ValueError` 提示必须提供

## 配置说明

### Accelerate 配置 (`accelerate.yaml`)

- 默认使用 DeepSpeed ZeRO Stage 2
- 支持多卡分布式训练
- 混合精度：`bf16`

### 常用参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model_name_or_path` | 预训练模型路径 | - |
| `--output_dir` | 输出目录 | - |
| `--learning_rate` | 学习率 | 5e-6 |
| `--lora_r` / `--lora_alpha` | LoRA 秩和缩放系数 | 64 / 128 |
| `--temperature` | 采样温度 | 1.1 |
| `--beta` | JSD 插值系数 | 0 |
| `--lmbda` | 损失权重 | 1 |
| `--fixed_teacher` | 启用固定教师策略 | false |
| `--use_ema_teacher` | 启用 EMA 教师策略 | false |
| `--use_vllm` | 启用 vLLM 加速生成 | true |
| `--log_with` | 日志系统 | swanlab |

## 注意事项

1. **路径配置**：所有脚本中的 `model_name_or_path`、`output_dir` 和 `dataset_path` 已替换为占位符，使用前需自行配置
2. **GPU 内存**：vLLM 共置模式下需合理设置 `vllm_gpu_memory_utilization`（默认 0.6）
3. **数据集路径**：通过 `--dataset_path` 参数传入，支持单个文件或多个文件（逗号分隔），不再在代码中硬编码
