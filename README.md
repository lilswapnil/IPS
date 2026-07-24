# IPS Discovery Analysis

Pipeline for the Round 1 IPS transition discovery. It turns focus-group
worksheets and discovery-session notes into structured **challenges** (pain
points) and **expectations**, scores sentiment, assigns theme categories,
clusters the records, reconciles a `final_category`, and produces charts for
shareouts.

## What it does

1. **Extract** — Pull one record per line from Word worksheets and discovery
   notes; tag each row with its source (`worksheet` | `meeting_notes`), map it to
   a focus group + department, and redact stakeholder names.
2. **Score sentiment** — Label each record negative / neutral / positive
   (neutral is kept, not forced into a polarity), then realign misfiled
   meeting-note rows between the two datasets.
3. **Categorize** — Assign each record to one of ten themes via a hybrid
   keyword + embedding approach, recording the method and confidence used.
4. **Cluster** — Reduce embeddings with UMAP and cluster with HDBSCAN (with
   soft-assignment of noise), for both challenges and expectations.
5. **Reconcile** — Build `final_category` by comparing each record's (normalized)
   category confidence against its cluster's purity.
6. **Visualize** — Explore category mix, focus-group / department volume,
   sentiment, clusters, and drill-down tables in the notebooks / Plotly figures.

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

### Inputs (under `data/`)

| Path | Role |
|------|------|
| `data/notes.docx` | Discovery session notes (split by Heading 3 focus groups) |
| `data/worksheets/*.docx` | Focus-group worksheets (pain / technology tables) |
| `dictionary.txt` | Stakeholder first names to redact as `[PERSON]` |

## Run

Open and run **`notebook/analysis.ipynb`** top to bottom. It calls the extractor
for you, so there is no separate CLI step required. The notebook:

1. Extracts records → `output/raw/challenges.csv`, `expectations.csv`
2. Loads + prepares them (aliases, source tags, short-note merge, name redaction)
3. Scores sentiment → `output/processed/*_scored.csv`
4. Realigns misclassified meeting notes
5. Categorizes themes and evaluates the categorization method mix
6. Clusters challenges and expectations (UMAP + HDBSCAN)
7. Reconciles `final_category`, then saves `categorized_*.csv`, `category_summary.*`
8. Builds exploratory Plotly charts under `output/figures/`

First run downloads DistilBERT (sentiment) and `all-mpnet-base-v2`
(categorization); later runs reuse the local cache.

You can also run the extractor directly:

```bash
python3 scripts/extract_records.py
```

### Notebooks

| Notebook | Purpose |
|----------|---------|
| `notebook/analysis.ipynb` | End-to-end: extract → sentiment → categorize → cluster → reconcile → explore & save |
| `notebook/notebook.ipynb` | Presentation charts from the categorized CSVs (category mix, heatmaps by focus group **and** department, purity-vs-confidence, interactive drill-downs) |
| `notebook/analyze_model.ipynb` | Sentiment model bake-off on a hand-labeled gold set |

> Run `analysis.ipynb` first — `notebook.ipynb` reads the categorized CSVs it
> produces. `notebook.ipynb` recomputes `final_category` from those columns, so it
> works even before the columns are persisted.

## Algorithm

### 1. Extraction & prep (`scripts/extract_records.py`)

- **Worksheets** — Pain tables yield left-column challenges and right-column
  suggested improvements (expectations); technology tables contribute "wish
  list" expectations, plus features to retain, labeled `(to keep)`. Each
  non-empty line becomes its own row.
- **Meeting notes** — Notes are split by Heading 3 (focus group). Under Heading 4
  sections matching *Challenges* / *Future capabilities*, each line becomes a row.
- **Short-note merge** — Meeting-note lines with ≤3 words are merged into the next
  longer line in the same focus group (worksheet cells are left alone).
- **Focus group → department** — Each focus group maps to a department (DOCE, BAA,
  NBD, CPO, FPB, Law, Assessment, CPC, IT, OCHD).
- **Privacy** — Names in `dictionary.txt` are replaced with `[PERSON]` before scoring.

### 2. Sentiment (`scripts/sentiment_analysis.py`)

The scorer is **`distilbert-base-uncased-finetuned-sst-2-english`**, chosen after
a gold-set bake-off in `notebook/analyze_model.ipynb` (~0.77 accuracy vs ~0.60 for
a 7-class DistilBERT).

Because SST-2 is binary and discovery phrasing is noisy, the pipeline adds:

| Step | Behavior |
|------|----------|
| Soft neutral | Low top-class score (< 0.65) or a nearly flat margin → `neutral` |
| Keep neutral | Uncertain records **stay neutral** unless a clear, unambiguous keyword signal points one way |
| Keyword hints | Pain / wish lexicons resolve uncertain cases and flip confident-but-conflicting ones (e.g. a "positive"-scored challenge full of pain words → negative) |
| Realignment | Positive meeting-note *challenges* move to expectations; negative meeting-note *expectations* move to challenges. Worksheet rows stay put. |

### 3. Hybrid categorization (`scripts/categorize_records.py`)

Ten fixed themes: Workflow & Business Processes, Case Management, Data Management
& Visibility, Records & Document Management, System Integration, Reporting &
Decision Support, Communication & Collaboration, Scheduling & Resource
Management, User Experience & Performance, Training & Documentation.

For each record the pipeline:

1. Scores **weighted keywords** on the body text.
2. Embeds the text with **`all-mpnet-base-v2`** and compares cosine similarity to
   natural-language category descriptions.
3. Picks a label on a priority ladder: strong keywords → semantic match → weak
   keywords → semantic fallback. When keyword and semantic agree, that label wins.

Each record keeps the chosen `Category`, the `Category_Method` used, and a
`Category_Confidence`. Reliability symptoms (crash, freeze, lag, …) are biased
toward **User Experience & Performance** even when process/schedule nouns appear.

### 4. Clustering (in `notebook/analysis.ipynb`)

Both challenges and expectations are reduced with **UMAP** and clustered with
**HDBSCAN**. A small parameter sweep picks settings by silhouette score minus
penalties for noise and for drifting from the target cluster count; leftover
noise points are soft-assigned into the nearest cluster when they fall inside its
envelope. Each cluster is labeled by its dominant category (`Cluster_Label`).

### 5. `final_category` reconciliation

`Category_Confidence` mixes units (keyword-score sums vs 0–1 cosine similarity),
so it is not directly comparable to cluster purity. It is rescaled to 0–1
(`Category_Confidence_Norm`, by its 95th percentile). **Cluster purity** is the
share of a cluster whose per-record `Category` matches `Cluster_Label`. A record
keeps its own `Category` unless it sits in a pure enough cluster whose purity
beats its normalized confidence — then it adopts the cluster's label. Tune with
the single `min_purity` argument (default `0.6`).

## Outputs

| Path | Contents |
|------|----------|
| `output/raw/worksheets.csv` | Flattened worksheet rows |
| `output/raw/docx_sections/` | One notes section per focus group |
| `output/raw/challenges.csv`, `expectations.csv` | Extracted records (pre-scoring) |
| `output/processed/challenges_scored.csv`, `expectations_scored.csv` | Records with `sentiment` |
| `output/processed/categorized_challenges.csv`, `categorized_expectations.csv` | Final records: `Category`, `Category_Method`, `Category_Confidence`, `Category_Confidence_Norm`, `Cluster`, `Cluster_Label`, `Cluster_Purity`, `Assign_Status`, `final_category` |
| `output/processed/category_summary.csv` / `.xlsx` | Category rollups (Excel has per-dataset + categorized sheets) |
| `output/processed/ips_likes.csv` | Worksheet "(to keep)" features |
| `output/figures/` | HTML / PNG chart exports |

## Project layout

```
data/                    source notes & worksheets
dictionary.txt           stakeholder names to redact
scripts/
  extract_records.py       Word → challenges / expectations CSVs (+ prep, redaction)
  sentiment_analysis.py    DistilBERT SST-2 + neutral handling / realignment
  categorize_records.py    keyword + embedding themes
notebook/
  analysis.ipynb           main end-to-end pipeline
  notebook.ipynb           presentation charts & interactive drill-downs
  analyze_model.ipynb      sentiment model comparison
output/
  raw/                     worksheets.csv, challenges.csv, expectations.csv, docx_sections/
  processed/               scored, categorized & summary outputs
  figures/                 chart exports (HTML / PNG)
diagrams/                  static reference charts
thrash/                    experiments, shareout & model-eval notebooks
requirements.txt
```

## Tips

- Run notebooks with the project root (or `notebook/`) as the working directory;
  cells resolve the repo root by looking for `scripts/` + `requirements.txt`.
- Re-run `scripts/extract_records.py` (or the first section of `analysis.ipynb`)
  whenever you add or edit files under `data/`.
- The interactive heatmaps in `notebook.ipynb` use dropdowns (not click-to-filter),
  which work in the Cursor / VS Code notebook renderer as well as Jupyter.
- If a model download fails, set `HF_TOKEN` in `.env` and retry.
```
