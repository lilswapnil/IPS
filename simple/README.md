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

Open and run **`analysis.ipynb`** top to bottom.

Writes scored / categorized CSVs under `output/processed/`.

### 2. Visualize

| Notebook | Purpose |
|----------|---------|
| `visualize.ipynb` | Interactive Plotly charts |
| `shareout.ipynb` | Matplotlib charts for decks → `output/figures/shareout/` |

## Layout

```
data/           source notes & worksheets
setup/          extract_records.py, analyze_records.py
output/
  processed/    challenges, expectations, categorized CSVs
  figures/      chart exports
analysis.ipynb  sentiment + categorization
visualize.ipynb interactive charts
shareout.ipynb  shareout / slide figures
```
