# English Bias Framework

Research pipeline for analyzing LLM scoring bias on IELTS/TOEFL essays. Compares a Mixture-of-Experts model (`openai/gpt-oss-120b`) against a Dense model (`qwen/qwen3-32b`) on reliability, consistency, and systematic bias using ICC, Bland-Altman, and paired t-test metrics with HITL routing simulation.

---

## Research Summary

| Item | Detail |
|---|---|
| Dataset | 50 respondents × 3 essay prompts (150 theoretical max) |
| Evaluation runs | 5 runs per essay per model |
| Rubric | 4 criteria: Grammar, Lexical, Coherence, Task (max 32 pts) |
| GPT submitted / valid | 130 submitted / 97 valid scored (33 JSON parse failures, 25.4%) |
| Qwen submitted / valid | 128 submitted / 128 valid scored (0 failures) |
| Valid pairs for comparison | 95 essays scored validly by both models |

---

## Key Findings

| Metric | GPT-oss-120b | Qwen3-32b | Advantage |
|---|---|---|---|
| ICC (reliability) | **0.943** — Excellent | 0.844 — Good | GPT |
| Mean score / 32 | 25.47 | **28.32** | Qwen |
| Consistency (mean CV) | **0.047** | 0.092 | GPT |
| Mean within-essay range | **2.4 pts** | 5.56 pts | GPT |
| Auto-approve rate | **55.4%** | 7.8% | GPT |
| JSON parse failure rate | 25.4% (33/130) | **0%** | Qwen |

**Key insights:**

- **GPT-oss is the more reliable scorer** — ICC 0.943 (excellent) vs 0.844 (good). Its within-essay score range of 2.4 pts is less than half of Qwen's 5.56 pts.
- **Qwen is the more format-compliant model** — Perfect JSON output on all 128 submitted essays. GPT failed to return parseable scores for 33 essays (25.4%) under zero-shot prompting conditions.
- **Qwen scores higher but far less consistently** — Mean 28.32 vs 25.47, but 92.2% of its essays were flagged for human review due to high score variance. GPT required review for only 44.6%.
- **Significant severity bias between models** — GPT scores systematically lower than Qwen. This gap is consistent across all three prompt categories.
- **Conservatism bias on data_report tasks** — Both models scored data_report essays most strictly (GPT: 24.20, Qwen: 27.37), likely due to the absence of visual context in the prompt. Neither model hallucinated visual content; both applied conservative scoring instead.

---

## Statistical Results

### Severity Bias — Paired t-test
Computed on **95 valid paired essays** (essays scored by both models with no parse failure).

| Statistic | Value |
|---|---|
| t-statistic | −7.43 |
| p-value | 5.03 × 10⁻¹¹ |
| GPT mean total | 25.47 (SD = 7.02) |
| Qwen mean total | 28.32 (SD = 6.00) |

GPT scores significantly lower than Qwen. The systematic gap is ~2.85 points on a 32-point rubric.

### Individual Agreement — Bland-Altman Analysis
Computed on **95 valid paired essays**.

| Metric | Value |
|---|---|
| Mean difference (GPT − Qwen) | −2.07 |
| Lower limit of agreement | −7.38 |
| Upper limit of agreement | +3.25 |

The limits of agreement span 10.6 points, indicating wide individual-level disagreement beyond the systematic gap. The asymmetric bounds (wider on the negative side) reflect GPT's tendency toward occasional extreme strictness rather than symmetric noise.

### Reliability — Intraclass Correlation (ICC-3)
5 runs per essay, two-way mixed consistency model.

| Model | ICC | n essays | n balanced |
|---|---|---|---|
| GPT-oss-120b | 0.943 | 97 | 81 |
| Qwen3-32b | 0.844 | 128 | 123 |

### HITL Routing Simulation
Threshold: CV < 0.15 AND score range ≤ 2 → Auto-Approve; else Flagged for Review.

| Model | Auto-Approve | Flagged for Review |
|---|---|---|
| GPT-oss-120b | 72 / 130 (55.4%) | 58 / 130 (44.6%) |
| Qwen3-32b | 10 / 128 (7.8%) | 118 / 128 (92.2%) |

### Topic Bias

| Prompt Category | GPT mean | Qwen mean | Gap |
|---|---|---|---|
| data_report | 24.20 | 27.37 | 3.17 |
| social_policy_opinion | 26.22 | 28.97 | 2.75 |
| tech_society_opinion | 27.18 | 28.85 | 1.67 |

---

## Prerequisites

- Python 3.10+
- See `requirements.txt` for dependencies

## Installation

```bash
pip install -r requirements.txt
```

---

## Pipeline

Three stages: preprocess → evaluate → analyze

| Stage | Command |
|---|---|
| Preprocess | `python src/preprocess.py --input data/raw/essays.xlsx --out-dir data/processed` |
| Evaluate | `python src/evaluate.py --inputs data/processed --models qwen3-32b gpt-oss-120b` |
| Analyze | `python src/analyze.py --inputs data/processed --out-dir data/processed/analysis` |

Run all three in sequence to generate reports.

---

## Outputs

Located in `--out-dir` (default: `data/processed/analysis`):

| File | Description |
|---|---|
| `report.html` | Interactive dashboard with KPI cards and embedded charts |
| `report.md` | Markdown summary with all tables |
| `report.json` | Machine-readable metrics |
| `metrics_summary.csv` | Per-model aggregated metrics |
| `hitl_routing.csv` | Per-essay HITL routing decisions |
| `image/*.png` | Static chart exports (Bland-Altman, heatmaps, boxplot, HITL) |

---

## Configuration

Edit constants in `src/analyze.py`:
- `HITL_CV_THRESHOLD = 0.15` — auto-approve if CV below this
- `HITL_RANGE_THRESHOLD = 2` — auto-approve if score range ≤ this

---

## Caching

The analyze stage skips computation if `metrics_summary.csv` and `hitl_routing.csv` already exist. Force recompute:

```bash
python src/analyze.py --inputs data/processed --out-dir data/processed/analysis --force
```

---

## Project Structure

```
english-bias-framework/
├── data/
│   └── processed/
│       ├── analysis/       # Generated reports and charts
│       ├── eval/           # Per-question model evaluation outputs
│       └── samples/        # Preprocessed essay samples (JSONL)
├── dataset/                # Source dataset (Excel)
├── notebook/               # Exploratory analysis notebook
├── scripts/
│   ├── extract_metrics.py  # Standalone metric extraction for reproducibility
│   ├── diag_checkpoint.py
│   ├── inspect_icc.py
│   └── inspect_xlsx.py
├── src/
│   ├── analyze.py          # Main analysis pipeline
│   ├── evaluate.py         # Model evaluation runner
│   ├── preprocess.py       # Data preprocessing
│   └── cli.py
├── requirements.txt
└── LICENSE
```

---

## Reproducing Metrics

To independently verify the reported numbers:

```bash
pip install pandas numpy
python scripts/extract_metrics.py
```

This reads directly from the evaluation JSONL files and prints total observations, paired intersection, and Bland-Altman metrics — matching `data/processed/analysis/report.json`.
