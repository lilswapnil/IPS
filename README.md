# IPS Discovery Analysis

Pipeline for Round 1 IPS transition discovery. It turns focus-group worksheets and session notes into structured pain points and expectations, scores sentiment, assigns theme categories, and produces charts for shareouts.

## What it does

1. **Extract** — Pull one record per line from Word worksheets and discovery notes; tag source (`worksheet` | `meeting_notes`); redact stakeholder names.
2. **Score sentiment** — Label each record negative / neutral / positive, then realign misfiled meeting-note rows.
3. **Categorize** — Assign each record to one of ten municipal themes via a hybrid keyword + embedding approach.
4. **Visualize** — Explore category mix, focus-group volume, and sentiment in notebooks / Plotly figures.

## Requirements

- Python 3.10+ recommended
- ~2–4 GB disk for Hugging Face / SentenceTransformer model downloads on first run

## Setup

From the project root:

```bash
cd /path/to/IPS
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Optional — put a Hugging Face token in `.env` for faster / authenticated model downloads:

```bash
echo 'HF_TOKEN=hf_...' > .env
```

### Inputs (already under `data/`)

| Path | Role |
|------|------|
| `data/notes.docx` | Discovery session notes (split by Heading 3 focus groups) |
| `data/worksheets/*.docx` | Focus-group worksheets (pain / technology tables) |
| `dictionary.txt` | Stakeholder first names to redact as `[PERSON]` |

## Run

Activate the venv, then either use the CLI extract step or jump straight into the analysis notebook (it calls extract for you).

### Option A — Full analysis notebook (recommended)

Open and run **`notebook/analysis.ipynb`** top to bottom. It:

1. Extracts records → `output/processed/challenges.csv`, `expectations.csv`
2. Scores sentiment → `*_scored.csv`
3. Realigns misclassified meeting notes
4. Categorizes themes → `categorized_*.csv`, `category_summary.csv`
5. Builds exploratory Plotly charts under `output/figures/`

First run downloads DistilBERT (sentiment) and `all-mpnet-base-v2` (categorization); later runs reuse the local cache.

### Option B — Extract only from the terminal

```bash
source .venv/bin/activate
python3 scripts/extract_records.py
```

Writes:

- `output/raw/worksheets.csv`
- `output/raw/docx_sections/` (one notes section per focus group)
- `output/processed/challenges.csv`
- `output/processed/expectations.csv`

### Visualizations & experiments

| Notebook | Purpose |
|----------|---------|
| `notebook/analysis.ipynb` | End-to-end extract → sentiment → categorize → explore |
| `notebook/notebook.ipynb` | Plotly charts from categorized CSVs (run analysis first) |
| `notebook/analyze_model.ipynb` | Sentiment model bake-off on a hand-labeled gold set |
| `thrash/shareout.ipynb` | Matplotlib deck figures → `output/figures/shareout/` |

## Algorithm

### 1. Extraction & prep

- **Worksheets** — Pain tables yield left-column challenges and right-column suggested improvements; technology tables contribute “wish list” expectation lines. Each non-empty line becomes its own row.
- **Meeting notes** — Notes are split by Heading 3 (focus group). Under Heading 4 sections matching *Challenges* / *Future capabilities*, each paragraph line becomes a row.
- **Short-note merge** — Meeting-note lines with ≤3 words are prepended onto the next longer line in the same focus group (worksheets are left alone).
- **Privacy** — Names in `dictionary.txt` are replaced with `[PERSON]` before scoring.

### 2. Sentiment (DistilBERT SST-2 + domain rules)

The production scorer is **`distilbert-base-uncased-finetuned-sst-2-english`**, chosen after a gold-set bake-off in `notebook/analyze_model.ipynb` (~0.77 accuracy vs ~0.60 for the previous 7-class DistilBERT).

Because SST-2 is binary and discovery phrasing is noisy, the pipeline adds:

| Step | Behavior |
|------|----------|
| Soft neutral | Top-class score &lt; 0.65 → treat as neutral |
| Dataset prior | Challenges default negative; expectations default positive when the model is flat or unsure |
| Keyword hints | Pain / wish lexicons override uncertain labels (e.g. “crash”, “manual” vs “wish”, “automate”) |
| Realignment | Positive meeting-note “challenges” move to expectations; negative meeting-note “expectations” move to challenges. Worksheet rows stay in their original tables. |

### 3. Hybrid categorization

Ten fixed themes (workflow, case management, data visibility, records, integration, reporting, communication, scheduling, UX/performance, training).

For each record the pipeline:

1. Scores **weighted keywords** on the title (text before `:`) and full body.
2. Embeds the text with **`all-mpnet-base-v2`** and compares cosine similarity to natural-language category blurbs.
3. Picks a label with a priority ladder: strong title keywords → strong body keywords → semantic match → weak keywords → semantic fallback. When keyword and semantic agree, that label wins.

Reliability symptoms (crash, freeze, lag, …) are biased toward **User Experience & Performance** even when schedule/process nouns are present.

## Project layout

```
data/                 source notes & worksheets
dictionary.txt        names to redact
scripts/
  extract_records.py      Word → challenges / expectations CSVs
  sentiment_analysis.py   DistilBERT SST-2 + priors / realign
  categorize_records.py   keyword + embedding themes
notebook/
  analysis.ipynb          main pipeline
  notebook.ipynb          interactive charts
  analyze_model.ipynb     sentiment model comparison
output/
  raw/                    worksheets.csv, docx_sections/
  processed/              scored & categorized CSVs
  figures/                HTML / image chart exports
thrash/                   experiments & shareout notebook
diagrams/                 static reference charts
requirements.txt
```

## Tips

- Run notebooks with the project root (or `notebook/`) as the working directory; cells resolve the repo root by looking for `scripts/` + `requirements.txt`.
- Re-run `extract_records` (or the first section of `analysis.ipynb`) whenever you add or edit files under `data/`.
- If model download fails, set `HF_TOKEN` in `.env` and retry.
