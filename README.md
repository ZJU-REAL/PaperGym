# PaperGym: Rubric-Centered Evolution for Research-Plan Generation

<div align="center">

[![Project Page](https://img.shields.io/badge/Project%20Page-PaperGym-blue)](https://zju-real.github.io/PaperGym)
[![Hugging Face Model](https://img.shields.io/badge/Model-PaperGym--Model-orange)](https://huggingface.co/CabbageWyh/PaperGym-Model)
[![Hugging Face Dataset](https://img.shields.io/badge/Dataset-PaperGym--20k-green)](https://huggingface.co/datasets/CabbageWyh/PaperGym-Data)

*Yuhan Wang · Zhengxi Lu · Yuchen Yan · Kaitao Song · Wenqi Zhang · Weiming Lu · Jun Xiao · Yueting Zhuang · Yongliang Shen*

</div>

## 🔥 Overview

Research planning is the decisive capability of AI scientists — yet a research plan admits no *verifiable* answer, so reinforcement learning lacks the environment it requires: tasks paired with a critic. **PaperGym** turns each scientific paper into a complete training environment by exploiting the structure of the paper itself:

- The **research question** is synthesized from the paper's *research goal + background*;
- The **evaluation criteria (rubrics)** are derived from the *method + experiments*, spanning two complementary dimensions — **methodological innovation** and **experimental design**.

Because the question and the criteria come from **disjoint sections** of the paper, criterion leakage drops to **3.7%** (versus 12–34% in existing benchmarks) — the reward can no longer be earned by paraphrase.

The rubrics are then used **twice** during training: first as privileged context for a rubric-conditioned self-distillation teacher, then as the reward for GRPO — a *teach–explore* paradigm.

<p align="center">
  <img src="docs/images/framework.png" width="880" alt="PaperGym framework">
</p>

## ✨ Highlights

- **3.7%** criterion leakage — roughly **3–9× lower** than existing rubric-based benchmarks (12%–34%).
- **20,000** instances in **PaperGym-20k** across five domains, with held-out **PaperGym-Innov / PaperGym-Design** test sets.
- **+5.6 / +5.0 / +4.8** average gains over base models across five benchmarks for Qwen3-1.7B / 4B / 8B — beating SFT, either training stage alone, and the reversed ordering.
- **73.48** on ResearchQA with fine-tuned Qwen3-8B, surpassing the much larger Kimi K2.6 (73.19).
- **58.1%** win rate in three-way blind comparisons on ResearchPlanGen-ML, vs. 28.2% for a RubricHub-trained baseline.

## 🧱 Method

PaperGym couples a **data-generation** pipeline that turns each paper into a low-leakage training environment with a **two-stage training** scheme that uses the extracted rubrics twice — first as privileged context, then as reward.

### 1 · Data generation — four-stage knowledge extraction

| Stage               | What happens                                                                                                                                                           |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Parse**     | Decompose each paper into*Research Goal · Background · Research Method · Experimental Design* (excluding numerical results) via map-reduce.                       |
| **Decouple**  | Synthesize the question from*Goal + Background* and the reference answer from *Method + Design* — disjoint sections, so leakage is prevented at the source.       |
| **Rubricify** | Merge, deduplicate, rank, and filter specialized rubrics to the top*n = 10* criteria per instance (innovation + design); apply a validated general rubric uniformly. |
| **Output**    | **PaperGym-20k** training set + held-out **PaperGym-Innov / PaperGym-Design** test sets.                                                                   |

### 2 · Rubric-centered training — teach then explore

Both stages share the same rubrics. **OPSD** (warm-up) has a rubric-conditioned teacher guide the student via token-level KL on on-policy rollouts — a rubric states evaluation *principles* rather than one realized answer, preserving more valid continuations than answer-conditioned distillation. **GRPO** (explore) then uses the rubrics as reward: a frozen base model grades each candidate criterion-by-criterion, with `r = α·r_spec + (1−α)·r_gen` (α = 0.7).

**Implementation.** Qwen3-1.7B / 4B / 8B (bf16). OPSD: LoRA (r=64, α=128), lr 5e-6. GRPO (verl): 8 rollouts per prompt via vLLM, KL penalty 0.01, lr 1e-6. The 4B / 8B models verify themselves; the 1.7B model is scored by the 4B model.

## 📊 Results

### Main results

<p align="center">
  <img src="docs/images/table.png" width="920" alt="Main results across model scales and training strategies">
</p>

### Win-rate comparisons

| Criterion               | Win rate |
| ----------------------- | -------- |
| **Overall Score** | <img src="docs/images/winrate/overall.png" width="680" alt="Overall-score win rate: PaperGym-20k 58.1%, RubricHub 28.2%, Base 13.7%"> |
| Goal Alignment          | <img src="docs/images/winrate/goal_alignment.png" width="680" alt="Goal-alignment win rate: PaperGym-20k 55.7%, RubricHub 29.0%, Base 15.3%"> |
| Novel Insight           | <img src="docs/images/winrate/novel_insight.png" width="680" alt="Novel-insight win rate: PaperGym-20k 58.4%, RubricHub 29.3%, Base 12.3%"> |
| Scientific Soundness    | <img src="docs/images/winrate/scientific_soundness.png" width="680" alt="Scientific-soundness win rate: PaperGym-20k 57.3%, RubricHub 29.0%, Base 13.7%"> |
| Execution Quality       | <img src="docs/images/winrate/execution_quality.png" width="680" alt="Execution-quality win rate: PaperGym-20k 54.6%, RubricHub 31.2%, Base 14.2%"> |
| Expected Impact         | <img src="docs/images/winrate/expected_impact.png" width="680" alt="Expected-impact win rate: PaperGym-20k 57.1%, RubricHub 29.4%, Base 13.5%"> |

Each bar shows the three-way win-rate split per criterion: <span style="background:#F2D96C">&nbsp;&nbsp;&nbsp;&nbsp;</span> baseline trained on PaperGym-20k, <span style="background:#A8D5D5">&nbsp;&nbsp;&nbsp;&nbsp;</span> baseline trained on RubricHub Science, <span style="background:#2B3F7B">&nbsp;&nbsp;&nbsp;&nbsp;</span> untrained Qwen3-1.7B. For each of the 685 ResearchPlanGen-ML questions, expert judges (DeepSeek-V4-Flash, Kimi K2.6, GLM-5.1; Gemini 3.7 Flash as tie-breaker) rank the three plans; the win rate is the share of questions where each model's plan is ranked best.

### Criterion leakage

<p align="center">
  <img src="docs/images/leakage.png" width="1000" alt="Criterion leakage: PaperGym-20k 3.73%, PaperGym-Innov 4.71%, PaperGym-Design 4.97%, HealthBench 11.90%, RubricHub Science 17.39%, ResearchQA 19.22%, ResearchPlanGen-ML 31.29%, ResearchPlanGen-ArXiv 34.10%">
</p>

See the [project page](https://zju-real.github.io/PaperGym) for the paper and complete experiments, ablations, and analysis.

## 📚 Benchmarks

The in-domain test sets are shipped in this repository:

| Benchmark                 | Scoring dimension         | Data                                  |
| ------------------------- | ------------------------- | ------------------------------------- |
| **PaperGym-Innov**  | Methodological innovation | `test/PaperGym/data/innov.parquet`  |
| **PaperGym-Design** | Experimental design       | `test/PaperGym/data/design.parquet` |

## 📦 Released Resources

- **Model**: fine-tuned models on [Hugging Face](https://huggingface.co/CabbageWyh/PaperGym-Model) (`CabbageWyh/PaperGym-Model`).
- **Corpus & benchmarks**: **PaperGym-20k** (20,000 instances) with held-out **PaperGym-Innov** / **PaperGym-Design** test sets, released on [Hugging Face](https://huggingface.co/datasets/CabbageWyh/PaperGym-Data) (`CabbageWyh/PaperGym-Data`).
- **Code**: the complete pipeline — data generation, OPSD / GRPO training, and the evaluation suite — lives in this repository.

## 🗂️ Repository Structure

```
PaperGym/
├── data_generation/          # Build the corpus from arXiv papers
│   ├── get_paper.py          #   batch-download paper TeX sources
│   ├── query_category.py     #   query arXiv categories via API
│   ├── process_paper/        #   merge / extract / section the LaTeX
│   └── problem_generation/   #   synthesize questions, answers, rubrics
├── opsd/                     # Stage 1 · rubric-conditioned OPSD (TRL)
│   ├── sft_train.py · opsd_train*.py
│   ├── opsd_trainer*.py      #   JSD loss, vLLM sync, EMA / fixed teacher
│   ├── data_collator*.py     #   student–teacher input pairs
│   └── scripts/              #   1B / 4B / 8B launch scripts
├── grpo/                     # Stage 2 · GRPO with rubric-as-reward (verl)
│   ├── reward_fn/            #   offline rubric scoring functions
│   ├── scripts/              #   training launch script
│   └── data/                 #   train / val sets (Parquet)
├── docs/                     # Project page (deploy via GitHub Pages)
│   ├── index.html            #   project homepage
│   └── images/               #   page figures
└── test/                     # Evaluation & analysis
    ├── PaperGym/             #   in-domain Innov / Design evaluation
    ├── ResearchQA/ · ResearchPlanGen/ · RubricHub/   # external benchmarks
    ├── compare/              #   blind two- / three-way win-rate comparison
    ├── data_leakage/         #   criterion-leakage detection
    ├── entropy/              #   response uncertainty (Shannon entropy)
    ├── rubric_rate/          #   Method vs. Experiment rubric analysis
    └── stability/            #   scorer-consistency analysis
```

## 🚀 Quick Start

Model, dataset, and API paths are placeholders in the launch scripts — replace them per your environment before running. Configure scoring APIs via `.env` (see [`.env.example`](.env.example)).

```bash
# 1. (Optional) build data from arXiv papers
python data_generation/get_paper.py
python data_generation/problem_generation/solve_dataset.py

# 2. Stage 1 — rubric-conditioned OPSD
bash opsd/scripts/run_opsd_8b.sh

# 3. Stage 2 — GRPO with rubric-as-reward
bash grpo/scripts/grpo_1b_offline.sh
```

See [opsd/README.md](opsd/README.md) and the per-module READMEs under `test/` for detailed instructions.

## 🧪 Evaluation

All benchmarks follow the same two-step workflow — **generate → score**. Model, dataset, and API paths are configurable variables in each script; replace the placeholders for your environment (remote API keys also come from `.env`, see [`.env.example`](.env.example)).

### 1 · Generate answers

Each benchmark's `run_generate_*` scripts start a **local vLLM** service (`.sh` without `_remote`) or call a **remote OpenAI-compatible API** (`*_remote.sh`), producing answer JSON files under `answer/<benchmark>/`.

```bash
bash test/PaperGym/run_generate_innov.sh                # in-domain Innov (local vLLM)
bash test/ResearchQA/run_generate_responses_remote.sh   # ResearchQA (remote API)
bash test/ResearchPlanGen/run_rubric_eval_remote.sh     # ResearchPlanGen-ML
bash test/RubricHub/run_generate_responses.sh           # RubricHub Science
```

### 2 · Score against rubrics

A judge model grades each generated response criterion-by-criterion. Configure the judge via `SCORE_BASE_URL` / `SCORE_API_KEY` / `SCORE_MODEL` in the script.

```bash
bash test/PaperGym/run_score.sh innov design     # in-domain scoring
bash test/ResearchQA/run_compute_coverage.sh     # ResearchQA coverage scoring
bash test/RubricHub/run_score_responses_batch.sh # batch rubric scoring
```

Analysis scripts (blind win-rate comparison, leakage detection, entropy, stability, rubric composition) live under `test/` — see the code and the per-module READMEs (`test/*/README_EN.md`, `test/*/README_ZH.md`).

## 📢 News

- **`2026-08`**: Released the PaperGym pipeline, the **PaperGym-20k** corpus, the **PaperGym-Innov / PaperGym-Design** benchmarks, and the fine-tuned models on [Hugging Face](https://huggingface.co/CabbageWyh/PaperGym-Model). *(arXiv link will be added upon release.)*

## ⭐ Citation

If you find PaperGym useful in your research, please consider citing:

```bibtex
@misc{wang2026papergym,
  title   = {PaperGym: Rubric-Centered Evolution for Research-Plan Generation},
  author  = {Yuhan Wang and Zhengxi Lu and Yuchen Yan and Kaitao Song and
             Wenqi Zhang and Weiming Lu and Jun Xiao and Yueting Zhuang and
             Yongliang Shen},
  year    = {2026},
  url     = {https://zju-real.github.io/PaperGym/},
  note    = {arXiv ID to be added}
}
```

## 🤝 Acknowledgements

This project builds on [verl](https://github.com/verl-project/verl), [vLLM](https://github.com/vllm-project/vllm), and [TRL](https://github.com/huggingface/trl). We thank the authors of these projects for their open-source contributions.
