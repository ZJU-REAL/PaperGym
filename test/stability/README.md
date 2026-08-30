# Stability

评测 LLM 评分稳定性的工具集，通过多轮独立评估分析评分一致性。

## 目录结构

```
stability/
├── stability_eval.py              # 稳定性评测主脚本
├── analyze_consistency.py         # 一致性分析
├── analyze_rubric_agreement.py    # Rubric 一致性分析
├── compare_scoring_models.py    # 评分模型对比
├── run_generate_stability.sh      # 便捷运行脚本
├── data/
│   └── stability.parquet          # 输入数据集
└── output/                        # 输出目录（自动创建）
```

## 快速开始

```bash
# 1. 安装依赖
pip install pandas numpy openai

# 2. 编辑 run_generate_stability.sh 配置参数

# 3. 运行稳定性评测
bash run_generate_stability.sh
```

## 核心功能

通过多轮独立评估（如 10 轮），分析：
- **评分一致性**：同一回答在多轮评估中的分数波动
- **Rubric 一致性**：不同 rubric 的评分稳定程度
- **模型对比**：不同评分模型的一致性差异

## 主要脚本

| 脚本 | 功能 |
|------|------|
| `stability_eval.py` | 主评测脚本，执行多轮独立评分 |
| `analyze_consistency.py` | 分析评分一致性（均值、方差、标准差） |
| `analyze_rubric_agreement.py` | 分析各 rubric 的评分一致程度 |
| `compare_scoring_models.py` | 对比不同评分模型的稳定性 |

## 典型流程

```bash
# 运行稳定性评测
bash run_generate_stability.sh

# 分析结果
python analyze_consistency.py --input output/stability_results.json
python analyze_rubric_agreement.py --input output/stability_results.json
```

## 使用自己的 LLM API 服务

本工具支持通过 OpenAI 兼容 API 调用任意 LLM 服务进行稳定性评测。

### 1. 修改脚本中的 API 配置

编辑 `run_generate_stability.sh`，设置你的 API 地址和密钥：

```bash
# LLM 模型参数
MODEL="your-model-name"
API_KEY="sk-your-api-key"
BASE_URL="http://your-api/v1"
```

### 2. 命令行参数方式

```bash
python stability_eval.py \
    --input data/stability.parquet \
    --output output/stability_results.json \
    --api_key sk-your-api-key \
    --base_url http://your-api/v1 \
    --model your-model-name \
    --rounds 10
```

### 3. 本地 vLLM 服务

```bash
python stability_eval.py \
    --input data/stability.parquet \
    --output output/stability_results.json \
    --api_key EMPTY \
    --base_url http://127.0.0.1:8000/v1 \
    --model your-local-model-name \
    --rounds 10
```
