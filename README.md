# LLM Evaluation Harness

> **A systematic benchmarking framework for open-source LLMs across mathematical
> reasoning, instruction following, and STEM knowledge — with failure mode
> classification, efficiency analysis, and model agreement diagnostics.**

*Built as a quantitative ML research project alongside an MSc in Applied Machine
Learning at Imperial College London.*

---

## Models Evaluated

| Model | Family | Parameters | Licence |
|---|---|---|---|
| `Qwen/Qwen2.5-0.5B-Instruct` | Qwen | 0.5B | Apache 2.0 |
| `microsoft/Phi-3.5-mini-instruct` | Phi | 3.8B | MIT |
| `meta-llama/Llama-3.2-1B-Instruct` | Llama | 1B | Llama 3.2 Community |

---

## Results Summary

| Model | Math (%) | Instruction (%) | STEM (%) | Overall (%) | Avg Latency |
|---|---|---|---|---|---|
| Qwen2.5-0.5B | 50 | 70 | 55 | **55** | 6.29s |
| Phi-3.5-mini | 66.7 | 80 | 85 | **75** | 10.46s |
| Llama-3.2-1B | 33.3 | 50 | 30 | **35** | 7.93s |

---

## Benchmark Tasks

### Task 1 — Mathematical Reasoning (30 questions)
Multi-step arithmetic and algebraic reasoning at three difficulty levels
(easy / medium / hard). Graded by extracting the final numerical answer
and comparing within 1% relative tolerance.

### Task 2 — Instruction Following (10 tasks)
Verifiable constraint satisfaction: exact word counts, forbidden words,
JSON output format, numbered lists, required phrases, no-digit constraints.
Graded by automated constraint checkers — no human judgement required.

### Task 3 — STEM Knowledge Q&A (20 questions)
Multiple-choice questions across Machine Learning, Mathematics, Physics,
and Computer Science. Graded by letter matching with pattern-based fallback.

---

## Key Features

| Feature | Detail |
|---|---|
| **Automated grading** | Custom graders per task — numerical, constraint-based, MCQ |
| **Failure taxonomy** | Classifies failures as: no answer / wrong format / calculation error / conceptual error / constraint violation |
| **Shared blindspot analysis** | Identifies questions all models answered incorrectly |
| **Agreement matrix** | Pairwise model agreement heatmap across STEM questions |
| **Efficiency Pareto frontier** | Accuracy-per-parameter and accuracy-per-latency trade-off |
| **Verbosity analysis** | Statistical test: does response length correlate with correctness? |
| **Greedy decoding** | `temperature=0`, `do_sample=False` — fully deterministic, reproducible |
| **CPU + GPU support** | Runs on free Colab T4 or local CPU (slower) |

---

## Repository Structure

```
LLM-Evaluation-Harness/
├── llm_evaluation_harness.ipynb   ← Full notebook (15 cells, end-to-end)
├── requirements.txt
├── README.md
├── .gitignore
└── results/
    ├── 01_main_results.png          ← Accuracy bar chart, radar, latency, scatter
    ├── 02_failure_modes.png         ← Stacked failure taxonomy by model
    ├── 03_stem_analysis.png         ← Domain breakdown + agreement heatmap
    ├── 04_efficiency_pareto.png     ← Pareto frontier: accuracy vs params/latency
    ├── 05_response_length.png       ← Verbosity vs correctness analysis
    ├── benchmark_summary.csv        ← Full accuracy + latency table
    └── raw_results.csv              ← Every question, response, grade, latency
```

---

## Charts

### Main Results — Accuracy, Radar Profile, Latency, Trade-off
![Main Results](results/01_main_results.png)

### Failure Mode Breakdown
![Failure Modes](results/02_failure_modes.png)

### Efficiency Pareto Frontier
![Pareto](results/04_efficiency_pareto.png)

---

## Quickstart

```bash
git clone https://github.com/advaitkulkarni2000/LLM-Evaluation-Harness.git
cd LLM-Evaluation-Harness
pip install -r requirements.txt
jupyter notebook llm_evaluation_harness.ipynb
```

> **Note:** `meta-llama/Llama-3.2-1B-Instruct` requires accepting Meta's
> licence on [HuggingFace](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
> and running `huggingface-cli login` before Cell 6.
> To skip this, swap it for `HuggingFaceTB/SmolLM2-1.7B-Instruct` in Cell 3
> (no licence required).

---

## Design Decisions

**Why greedy decoding?**
`temperature=0` makes all results fully deterministic and reproducible.
Sampling-based evaluation introduces variance that obscures true capability
differences between models — especially important for small N benchmarks.

**Why these three tasks?**
They test orthogonal capabilities: mathematical reasoning (multi-step logic),
instruction following (constraint satisfaction), and factual recall (knowledge).
A model can score well on one and poorly on another, making the radar chart
genuinely informative rather than redundant.

**Why classify failure modes rather than just report accuracy?**
Accuracy alone doesn't tell you *why* a model fails. The distinction between
"wrong format" (the model knew the answer but couldn't express it correctly)
and "conceptual error" (the model had no idea) has completely different
implications for fine-tuning and prompt engineering.

---

## Findings

1. **Best overall model:** Phi-3.5-mini with 75% overall accuracy, despite being
   only 3.8B parameters — outperforming Llama-3.2-1B (55%) by 20 percentage points.

2. **Most parameter-efficient:** Phi-3.5-mini (19.7% accuracy per billion params)
   vs Llama-3.2-1B (55% / 1B) vs Qwen2.5-0.5B (70% / 0.5B = 70 acc/B).
   Qwen2.5-0.5B wins on raw acc/param ratio (70 acc/B) but scores lowest overall —
   demonstrating that parameter efficiency and absolute capability are distinct metrics.

3. **Hardest task:** STEM Knowledge — Qwen2.5-0.5B scored only 30%, and the
   cross-model average was 56.7%, lower than both Math (50%) and Instruction (66.7%)
   averages — suggesting factual recall is the primary bottleneck at sub-1B scale.

4. **Instruction following was the strongest task for all models** — Phi-3.5-mini
   hit 80% and Llama-3.2-1B hit 70%, indicating that instruction-tuning is effective
   even at small scales. JSON output and word-count constraints were the hardest
   constraint types across all models.

5. **Sharpest difficulty cliff:** Math showed the largest variance across models
   (33.3% to 66.7% — a 33.4pp gap), making it the most discriminative task for
   separating model capability tiers.

6. **Latency vs accuracy trade-off:** Phi-3.5-mini achieves the best accuracy (75%)
   at 10.46s average latency. Qwen2.5-0.5B is fastest in theory but produced the
   lowest accuracy (35%) at 7.93s — a poor trade-off. Llama-3.2-1B sits in the
   middle (55% at 6.29s) but is outperformed on both axes by Phi-3.5-mini.

7. **Key insight — scale matters non-linearly:** Phi-3.5-mini (3.8B) outperforms
   Llama-3.2-1B (1B) by 20pp overall despite being 3.8× larger — suggesting the
   Phi training data quality and instruction tuning methodology contributes more
   than raw parameter count at this scale range.
---

## References

- Jegadeesh & Titman (1993) — referenced in companion backtesting project
- Zhu et al. (2023). *Do Large Language Models Know What They Don't Know?*
- Wei et al. (2022). *Chain-of-Thought Prominting Elicits Reasoning in LLMs.*
- Zhou et al. (2023). *Instruction-Following Evaluation for LLMs (IFEval).*
- Phi-3 Technical Report, Microsoft (2024)
- Qwen2.5 Technical Report, Alibaba (2024)

---

*Author: Advait Kulkarni | Imperial College London MSc Applied Machine Learning 2025–2026*
*Companion project: [Systematic Backtesting Framework](https://github.com/advaitkulkarni2000/Systematic-Backtesting)*
