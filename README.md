# IPS Discovery Analysis

Pipeline for Round 1 IPS transition discovery: extract pain points and expectations from notes + worksheets, score sentiment, categorize themes, and build shareout charts.

## Setup

```bash
cd simple
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional: put a Hugging Face token in `.env` as `HF_TOKEN=...` (faster model downloads).

**Inputs** (already under `data/`):
- `data/notes.docx` — discovery session notes
- `data/worksheets/*.docx` — focus group worksheets

## Run

### 1. Analyze (sentiment + categories)

Open and run **`notebook/analysis.ipynb`** top to bottom.

Writes scored / categorized CSVs under `output/processed/`.

### 2. Visualize

| Notebook | Purpose |
|----------|---------|
| `notebook/visualize.ipynb` | Interactive Plotly charts |
| `notebook/analyze_model.ipynb` | Sentiment model bake-off |
| `shareout.ipynb` | Matplotlib charts for decks → `output/figures/shareout/` |

## Layout

```
data/           source notes & worksheets
scripts/        extract_records.py, sentiment_analysis.py, categorize_records.py
notebook/
  analysis.ipynb       sentiment + categorization
  visualize.ipynb      interactive charts
  analyze_model.ipynb  sentiment model comparison
output/
  processed/    challenges, expectations, categorized CSVs
  figures/      chart exports
shareout.ipynb  shareout / slide figures
```
