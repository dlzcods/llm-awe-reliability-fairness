# English Bias Framework

Research pipeline for analyzing LLM scoring bias on IELTS/TOEFL essays. Compares model reliability using ICC, CV, and t-test metrics with HITL routing.

## Key Findings

From testing `openai/gpt-oss-120b` vs `qwen/qwen3-32b` on 150 essays:

| Metric | GPT-oss-120b | Qwen3-32b | Winner |
|--------|-------------|-----------|--------|
| ICC (reliability) | **0.943** | 0.844 | GPT |
| Auto-approve rate | **55%** | 8% | GPT |
| Mean score | 25.5 | **28.3** | Qwen |
| Consistency (CV) | **0.047** | 0.092 | GPT |

**Key insights:**
- **GPT-oss is more reliable** — ICC 0.94 (excellent) vs 0.84 (good). More consistent scoring across repeated runs.
- **GPT needs less human review** — 55% auto-approve vs 8%. Saves reviewer time.
- **Qwen scores higher but less consistently** — Mean 28.3 vs 25.5, but 92% of essays flagged for manual review.
- **Significant severity bias** — Models disagree by ~2.8 points on average (p < 0.001). GPT is stricter.

> Full interactive dashboard with charts: https://english-bias-framework.tiiny.site

## Prerequisites

- Python 3.10+
- See `requirements.txt` for dependencies

## Installation

```bash
pip install -r requirements.txt
```

## Pipeline

Three stages: preprocess → evaluate → analyze

| Stage | Command |
|-------|---------|
| Preprocess | `python src/preprocess.py --input data/raw/essays.xlsx --out-dir data/processed` |
| Evaluate | `python src/evaluate.py --inputs data/processed --models qwen3-32b gpt-oss-120b` |
| Analyze | `python src/analyze.py --inputs data/processed --out-dir data/processed/analysis` |

Run all three in sequence to generate reports.

## Outputs

Located in `--out-dir` (default: `data/processed/analysis`):

| File | Description |
|------|-------------|
| `report.html` | Interactive dashboard with KPI cards |
| `report.md` | Markdown summary with tables |
| `report.json` | Numerical metrics |
| `metrics_summary.csv` | Per-model aggregated metrics |
| `hitl_routing.csv` | Per-essay HITL decisions |
| `images/*.png` | Static chart exports |

## Caching

The analyze stage skips computation if `metrics_summary.csv` and `hitl_routing.csv` already exist. Force recompute:

```bash
python src/analyze.py --inputs data/processed --out-dir data/processed/analysis --force
```

## Configuration

Edit constants in `src/analyze.py`:
- `HITL_CV_THRESHOLD = 0.15` — auto-approve if CV below this
- `HITL_RANGE_THRESHOLD = 2` — auto-approve if score range ≤ this
- Model colors and output paths

## Testing

```bash
python -m pytest tests/
```
