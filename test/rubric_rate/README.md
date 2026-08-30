# Rubric Rate

分析学术论文数据集中 **Rubric** 的类型构成占比（Method vs Experiment）。

## 目录结构

```
rubric_rate/
├── analyze_rubric_types.py      # 核心分析脚本
├── run_analyze_rubric_types.sh  # 便捷运行脚本
├── test_mock.py                 # Mock 测试脚本
├── data/
│   └── rubric_rate.parquet      # 输入数据集
└── output/                      # 输出目录（自动创建）
```

## 快速开始

```bash
# 1. 安装依赖
pip install pandas numpy openai retry

# 2. 编辑 run_analyze_rubric_types.sh 配置 API 参数

# 3. 运行分析
bash run_analyze_rubric_types.sh
```

## 核心功能

使用 LLM 将每条 rubric 自动分类为 **Method**（方法）或 **Experiment**（实验），输出：
- 总体 Method/Experiment 占比
- 各学科类别内的占比统计

## 主要参数

```bash
python analyze_rubric_types.py \
    --parquet_path data/rubric_rate.parquet \  # 输入数据
    --output_dir output \                        # 输出目录
    --rubric_key rubrics \                       # rubric 集合键名
    --categories cs,econ,physics \                # 指定学科（可选）
    --num_workers 10                             # 并行线程数
```

## 输出文件

- `rubric_classification_detail.json` — 逐条分类结果
- `rubric_classification_stats.json` — 统计汇总

## Mock 测试

```bash
python test_mock.py
```

无需调用 LLM API，验证核心逻辑和路径配置。

## 使用自己的 LLM API 服务

本工具支持通过 OpenAI 兼容 API 调用任意 LLM 服务进行 rubric 类型分析。

### 1. 修改脚本中的 API 配置

编辑 `run_analyze_rubric_types.sh`，设置你的 API 地址和密钥：

```bash
# LLM 模型参数
MODEL="your-model-name"
API_KEY="sk-your-api-key"
BASE_URL="http://your-api/v1"
```

### 2. 命令行参数方式

```bash
python analyze_rubric_types.py \
    --parquet_path data/rubric_rate.parquet \
    --api_key sk-your-api-key \
    --base_url http://your-api/v1 \
    --model your-model-name
```

### 3. 本地 vLLM 服务

```bash
python analyze_rubric_types.py \
    --parquet_path data/rubric_rate.parquet \
    --api_key EMPTY \
    --base_url http://127.0.0.1:8000/v1 \
    --model your-local-model-name
```
