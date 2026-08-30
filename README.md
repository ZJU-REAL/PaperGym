<div align="center">

# PaperGym: Rubric-Centered Framework for Research-Plan Generation

[![Paper](https://img.shields.io/badge/Paper-Preprint_PDF-b31b1b.svg)](https://zju-real.github.io/PaperGym/paper.pdf)
[![Project Page](https://img.shields.io/badge/Project-Page-2f6f9f.svg)](https://zju-real.github.io/PaperGym)
[![Model](https://img.shields.io/badge/Model-HuggingFace-FFD21E.svg)](https://huggingface.co/CabbageWyh/PaperGym-Model)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow.svg)](https://huggingface.co/datasets/CabbageWyh/PaperGym-Data)

**Yuhan Wang<sup>1,\*</sup> · Zhengxi Lu<sup>1,\*</sup> · Yuchen Yan<sup>1</sup> · Kaitao Song<sup>2</sup> · Wenqi Zhang<sup>1</sup> · Weiming Lu<sup>1</sup> · Jun Xiao<sup>1</sup> · Yueting Zhuang<sup>1</sup> · Yongliang Shen<sup>1,†</sup>**

<sup>1</sup>Zhejiang University &nbsp;·&nbsp; <sup>2</sup>Apple

<sup>\*</sup>Equal contribution &nbsp;·&nbsp; <sup>†</sup>Corresponding author

</div>

## 🔥 Overview

Research planning is the decisive capability of AI scientists — yet a research plan admits no *verifiable* answer, so reinforcement learning lacks the environment it requires: tasks paired with a critic. **PaperGym** turns each scientific paper into a complete training environment by exploiting the structure of the paper itself:

- The **research question** is synthesized from the paper's *research goal + background*;
- The **evaluation criteria (rubrics)** are derived from the *method + experiments*, spanning two complementary dimensions — **methodological innovation** and **experimental design**.

Because the question and the criteria come from **disjoint sections** of the paper, criterion leakage drops to **3.7%** (versus 12–34% in existing benchmarks) — the reward can no longer be earned by paraphrase.

The rubrics are then used **twice** during training: first as privileged context for a rubric-conditioned self-distillation teacher, then as the reward for GRPO — a *teach–explore* paradigm.

<p align="center">
  <img src="assets/framework.png" width="880" alt="PaperGym framework">
</p>

## ✨ Highlights

- **3.7%** criterion leakage — roughly **3–8× lower** than existing rubric-based benchmarks (12%–34%).
- **20,000** instances in **PaperGym-20k** across five domains, with held-out **PaperGym-Innov / PaperGym-Design** test sets.
- **+5.6 / +5.0 / +4.8** average gains over base models across five benchmarks for Qwen3-1.7B / 4B / 8B — beating SFT, either training stage alone, and the reversed ordering.
- **73.48** on ResearchQA with fine-tuned Qwen3-8B, surpassing the much larger Kimi K2.6 (73.19).
- **58.1%** win rate in three-way blind comparisons on ResearchPlanGen-ML, vs. 28.2% for a RubricHub-trained baseline.

## 🧱 Method

### Data Generation — a four-stage knowledge-extraction pipeline

Each arXiv paper is decomposed into disjoint sections, and questions / answers are synthesized from **non-overlapping sources**, eliminating data contamination at its origin.

| Stage | What happens |
|---|---|
| **Parse** | Each paper is decomposed into *Research Goal, Background, Research Method, Experimental Design* (excluding concrete numerical results) via a map-reduce procedure. |
| **Decouple** | The question is synthesized from *Goal + Background*; the reference answer from *Method + Design* — disjoint sections, so the reward cannot be earned by paraphrase. |
| **Rubricify** | Specialized rubrics are merged, deduplicated, ranked, and filtered to the top *n = 10* criteria per instance, covering both **innovation** and **design**; a validated general rubric is applied uniformly. |
| **Output** | The resulting **PaperGym-20k** training set and the held-out **PaperGym-Innov / PaperGym-Design** test sets. |

### Rubric-Centered Training — a teach–explore paradigm

Both training stages are supervised by the same rubrics.

- **Stage 1 · Warm-up — Rubric-Conditioned OPSD.** A rubric-conditioned teacher guides the unconditioned student on on-policy rollouts, minimizing token-level KL divergence. Because a rubric specifies evaluation *principles* rather than one realized solution, it preserves a broader set of valid continuations than answer-conditioned distillation.
- **Stage 2 · Explore — GRPO with Rubric-as-Rewards.** A frozen copy of the base model self-grades each candidate response criterion by criterion; the reward combines the specialized and general rubric scores as `r = α · r_spec + (1 − α) · r_gen` with `α = 0.7`.

**Implementation.** Qwen3 models at the 1.7B / 4B / 8B scales (bf16). OPSD fine-tunes with LoRA (r=64, α=128) at lr 5e-6; GRPO runs on verl with 8 responses per prompt via vLLM, a KL penalty of 0.01, and lr 1e-6. The 4B / 8B models serve as their own rubric verifiers, while the 1.7B model is scored by the 4B model.

## 📊 Results

### Main results (Qwen3-8B)

| Training | RubricHub Sci. | ResearchPlanGen-ML | ResearchQA | Innov | Design | Avg |
|---|--:|--:|--:|--:|--:|--:|
| Base | 46.07 | 20.28 | 66.65 | 18.69 | 17.03 | 33.74 |
| SFT | 45.29 | 17.91 | 67.53 | 17.52 | 15.68 | 32.79 |
| OPSD | 47.41 | 23.17 | 69.17 | 21.42 | 18.04 | 35.84 |
| GRPO | 48.09 | 20.76 | 71.02 | 20.26 | 19.17 | 35.86 |
| **OPSD + GRPO** | **49.41** | **23.53** | **73.48** | **24.47** | **21.88** | **38.55** |

### Criterion leakage

| Dataset | Leakage |
|---|---|
| **PaperGym-20k** | **3.73%** |
| PaperGym-Innov | 4.71% |
| PaperGym-Design | 4.97% |
| HealthBench | 11.90% |
| RubricHub Science | 17.39% |
| ResearchQA | 19.22% |
| ResearchPlanGen-ML | 31.29% |
| ResearchPlanGen-ArXiv | 34.10% |

See the [paper](https://zju-real.github.io/PaperGym/paper.pdf) for complete experiments, ablations, and analysis.

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

## 📢 News

- **`2026-08`**: Released the PaperGym pipeline, the **PaperGym-20k** corpus, the **PaperGym-Innov / PaperGym-Design** benchmarks, and the fine-tuned models on [Hugging Face](https://huggingface.co/CabbageWyh/PaperGym-Model). *(arXiv link will be added upon release.)*

## ⭐ Citation

If you find PaperGym useful in your research, please consider citing:

```bibtex
@misc{wang2026papergym,
  title   = {PaperGym: Rubric-Centered Framework for Research-Plan Generation},
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
