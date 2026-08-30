# ResearchQA

基于 ResearchQA 数据集的 LLM 回答生成与评分框架。

## 目录结构

```
ResearchQA/
├── generate_responses.py              # 多线程生成 LLM 回答
├── compute_coverage_ds.py             # 评分脚本（支持并行）
├── show_results.py                    # 结果展示
├── researchqa_loader.py               # 数据加载工具
├── run_generate_responses.sh          # 本地 vLLM 生成脚本
├── run_generate_responses_remote.sh   # 远程 API 生成脚本
├── run_compute_coverage.sh            # 评分脚本
├── data/
│   └── research_qa.json               # 数据集
├── answer/                            # 生成的回答（自动创建）
└── output/                            # 评分结果（自动创建）
```

## 快速开始

```bash
# 1. 安装依赖
pip install openai tqdm retry

# 2. 生成回答（本地 vLLM）
bash run_generate_responses.sh

# 3. 评分
bash run_compute_coverage.sh

# 4. 查看结果
python show_results.py --dir output/
```

## 核心功能

**生成回答**：多线程调用 LLM API 为 ResearchQA 数据集中的每个问题生成回答。

**评分**：使用 LLM 按 rubric 逐条评判回答质量，输出 5 级评分：
- Not at all → Barely → Moderately → Mostly → Completely

**结果展示**：汇总各模型的平均覆盖率，以表格形式输出。

## 主要脚本

| 脚本 | 功能 |
|------|------|
| `generate_responses.py` | 多线程生成回答 |
| `compute_coverage_ds.py` | 并行 rubric 评分 |
| `show_results.py` | 结果汇总展示 |
| `researchqa_loader.py` | 加载 ResearchQA JSON 数据 |

## 典型流程

```bash
# 生成回答 → 评分 → 查看结果
bash run_generate_responses.sh
bash run_compute_coverage.sh
python show_results.py --dir output/
```

## Mock 测试

```bash
python test_mock.py
```

无需调用 LLM API，验证核心逻辑和路径配置。

## 使用自己的 LLM API 服务

本工具支持通过 OpenAI 兼容 API 调用任意 LLM 服务完成回答生成和评分。

### 1. 生成回答

#### 修改脚本配置

编辑 `run_generate_responses_remote.sh`，设置你的 API 地址和密钥：

```bash
API_BASE="http://your-api/v1"
API_KEY="sk-your-api-key"
MODEL="your-model-name"
```

#### 命令行参数方式

```bash
python generate_responses.py \
    --data data/research_qa.json \
    --output answer/responses.json \
    --api_base http://your-api/v1 \
    --api_key sk-your-api-key \
    --model your-model-name
```

### 2. 评分

#### 修改脚本配置

编辑 `run_compute_coverage.sh`，设置评分模型的 API 地址和密钥：

```bash
API_BASE="http://your-api/v1"
API_KEY="sk-your-api-key"
MODEL="your-model-name"
```

#### 命令行参数方式

```bash
python compute_coverage_ds.py \
    --input answer/responses.json \
    --output output/scores.json \
    --api_base http://your-api/v1 \
    --api_key sk-your-api-key \
    --model your-model-name
```

### 3. 本地 vLLM 服务

如果使用本地 vLLM 部署：

```bash
# 生成回答
python generate_responses.py \
    --data data/research_qa.json \
    --output answer/responses.json \
    --api_base http://127.0.0.1:8000/v1 \
    --api_key EMPTY \
    --model your-local-model-name

# 评分
python compute_coverage_ds.py \
    --input answer/responses.json \
    --output output/scores.json \
    --api_base http://127.0.0.1:8000/v1 \
    --api_key EMPTY \
    --model your-local-model-name
```
