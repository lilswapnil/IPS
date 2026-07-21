"""Multi-model matrix analysis for IPS discovery sentiment.

Compares candidate models on challenges/expectations with:
  - source-prior alignment (challenges→negative, expectations→positive)
  - heuristic gold labels (strong pain/wish keyword cues)
  - pairwise agreement matrix
  - latency + confidence

Each model runs in an isolated subprocess by default so file handles
(and the Jupyter ZMQ "Too many open files" failure mode) are released
between candidates.

Run via model_matrix.ipynb or:
  python3 -m setup.model_matrix
"""

from __future__ import annotations

import gc
import os
import re
import resource
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Shared workplace framing (same hypothesis family as production)
# ---------------------------------------------------------------------------
ZS_HYPOTHESIS = "This workplace discovery note indicates {}."
ZS_LABELS = [
    "process friction that needs fixing",
    "a desired future capability",
    "neither friction nor a request",
]
ZS_MAP = {
    "process friction that needs fixing": "negative",
    "a desired future capability": "positive",
    "neither friction nor a request": "neutral",
}

PAIN_HINTS = re.compile(
    r"\b("
    r"manual|can't|cannot|unable|missing|lack|broken|fail|slow|crash|freeze|"
    r"duplicate|rework|bottleneck|delay|inefficient|problem|issue|frustrating|"
    r"hard to|difficult|workaround|outdated|fragment|disconnect|error|"
    r"too many|no way|without|paperwork|spreadsheet|cumbersome|tedious"
    r")\b",
    re.I,
)
WISH_HINTS = re.compile(
    r"\b("
    r"wish|want|need|should|would like|would be|ability|able to|capability|"
    r"automate|automation|integrat|improve|better|enable|allow|feature|"
    r"dashboard|notify|helpful if|ideal|future"
    r")\b",
    re.I,
)

# Practical bake-off set (large MNLI models included; each runs isolated)
MODEL_CANDIDATES = [
    (
        "deberta-zs-v1.1",
        "MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33",
        "zero-shot",
    ),
    (
        "deberta-mnli",
        "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        "zero-shot",
    ),
    (
        "distilbert-mnli",
        "typeform/distilbert-base-uncased-mnli",
        "zero-shot",
    ),
    (
        "twitter-roberta",
        "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "sentiment3",
    ),
    (
        "distilbert-7class",
        "Thi144/sentiment-distilbert-7class",
        "sentiment7",
    ),
    (
        "sst2-distilbert",
        "distilbert-base-uncased-finetuned-sst-2-english",
        "sentiment2",
    ),
]

SENT7_MAP = {
    "LABEL_0": "negative",
    "LABEL_1": "negative",
    "LABEL_2": "negative",
    "LABEL_3": "neutral",
    "LABEL_4": "positive",
    "LABEL_5": "positive",
    "LABEL_6": "positive",
    "Very Negative": "negative",
    "Negative": "negative",
    "Slightly Negative": "negative",
    "Neutral": "neutral",
    "Slightly Positive": "positive",
    "Positive": "positive",
    "Very Positive": "positive",
}

SENT3_MAP = {
    "negative": "negative",
    "Neutral": "neutral",
    "neutral": "neutral",
    "positive": "positive",
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive",
}


def configure_runtime(*, nofile: int = 8192) -> None:
    """Reduce FD pressure before loading Hugging Face models."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = min(nofile, hard) if hard > 0 else nofile
        if soft < target:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    except (ValueError, OSError):
        pass


def _load_hf_token() -> None:
    env_path = Path(".env")
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key == "HF_TOKEN" and val and "HF_TOKEN" not in os.environ:
            os.environ["HF_TOKEN"] = val


def heuristic_gold(text: str, source: str) -> str | None:
    """High-confidence auto label, or None if ambiguous."""
    t = (text or "").strip()
    if not t:
        return None
    pain = bool(PAIN_HINTS.search(t))
    wish = bool(WISH_HINTS.search(t))
    if pain and not wish:
        return "negative"
    if wish and not pain:
        return "positive"
    return None


def load_eval_frame(
    challenges_csv: Path,
    expectations_csv: Path,
    *,
    n_per_source: int | None = 80,
    seed: int = 42,
) -> pd.DataFrame:
    """Build eval rows with text, source, prior, optional gold.

    n_per_source=None → use the full challenges + expectations dataset.
    """
    ch = pd.read_csv(challenges_csv)
    ex = pd.read_csv(expectations_csv)

    ch_rows = pd.DataFrame(
        {
            "text": ch["pain_points"].fillna("").astype(str),
            "source": "challenges",
            "prior": "negative",
        }
    )
    ex_rows = pd.DataFrame(
        {
            "text": ex["expectations"].fillna("").astype(str),
            "source": "expectations",
            "prior": "positive",
        }
    )
    frame = pd.concat([ch_rows, ex_rows], ignore_index=True)
    frame = frame.loc[frame["text"].str.strip().ne("")].copy()

    if n_per_source is not None:
        parts = []
        for _, g in frame.groupby("source"):
            n = min(n_per_source, len(g))
            parts.append(g.sample(n, random_state=seed))
        frame = pd.concat(parts, ignore_index=True)

    frame["gold"] = [
        heuristic_gold(t, s) for t, s in zip(frame["text"], frame["source"])
    ]
    return frame.reset_index(drop=True)


@dataclass
class ModelResult:
    name: str
    model_id: str
    kind: str
    labels: list[str]
    scores: list[float]
    margins: list[float]
    seconds: float
    error: str | None = None


def _release_pipeline(pipe) -> None:
    """Drop model weights / tokenizer handles before loading the next candidate."""
    if pipe is None:
        return
    try:
        if hasattr(pipe, "model"):
            del pipe.model
        if hasattr(pipe, "tokenizer"):
            del pipe.tokenizer
        if hasattr(pipe, "modelcard"):
            del pipe.modelcard
    except Exception:
        pass
    try:
        del pipe
    except Exception:
        pass
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _build_pipe(model_id: str, kind: str):
    from transformers import pipeline

    _load_hf_token()
    common = dict(model=model_id, device=-1)
    if kind == "zero-shot":
        return pipeline("zero-shot-classification", multi_label=False, **common)
    return pipeline(
        "sentiment-analysis",
        truncation=True,
        max_length=512,
        **common,
    )


def _predict_batch(
    pipe, texts: list[str], kind: str, *, batch_size: int = 16
) -> tuple[list[str], list[float], list[float]]:
    labels: list[str] = []
    scores: list[float] = []
    margins: list[float] = []

    if kind == "zero-shot":
        for text in texts:
            result = pipe(
                text,
                candidate_labels=ZS_LABELS,
                hypothesis_template=ZS_HYPOTHESIS,
                truncation=True,
                max_length=512,
            )
            lab = ZS_MAP[result["labels"][0]]
            sc = float(result["scores"][0])
            mg = (
                float(result["scores"][0] - result["scores"][1])
                if len(result["scores"]) > 1
                else sc
            )
            labels.append(lab)
            scores.append(sc)
            margins.append(mg)
        return labels, scores, margins

    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        raw = pipe(chunk)
        if isinstance(raw, dict):
            raw = [raw]
        for item in raw:
            raw_label = str(item["label"])
            sc = float(item["score"])
            if kind == "sentiment7":
                lab = SENT7_MAP.get(raw_label, "neutral")
            elif kind == "sentiment3":
                lab = SENT3_MAP.get(raw_label, SENT3_MAP.get(raw_label.lower(), "neutral"))
            else:
                lab = (
                    "positive"
                    if raw_label.upper() in {"POSITIVE", "LABEL_1"}
                    else "negative"
                )
            labels.append(lab)
            scores.append(sc)
            margins.append(sc)
    return labels, scores, margins


def run_model(
    name: str,
    model_id: str,
    kind: str,
    texts: list[str],
) -> ModelResult:
    """Score texts with one model; always release weights afterward."""
    configure_runtime()
    pipe = None
    try:
        pipe = _build_pipe(model_id, kind)
        t0 = time.perf_counter()
        labels, scores, margins = _predict_batch(pipe, texts, kind)
        elapsed = time.perf_counter() - t0
        return ModelResult(name, model_id, kind, labels, scores, margins, elapsed)
    except Exception as exc:  # noqa: BLE001 — keep matrix going
        n = len(texts)
        return ModelResult(
            name,
            model_id,
            kind,
            ["error"] * n,
            [0.0] * n,
            [0.0] * n,
            0.0,
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        _release_pipeline(pipe)


def run_model_isolated(
    name: str,
    model_id: str,
    kind: str,
    texts: list[str],
    *,
    timeout: float | None = None,
) -> ModelResult:
    """Run one model in a fresh interpreter so open files cannot accumulate.

    Uses ``python -m setup.model_matrix --worker`` (subprocess) instead of
    multiprocessing spawn — more reliable under Jupyter / IPython.
    """
    import json
    import subprocess
    import sys
    import tempfile

    with tempfile.TemporaryDirectory(prefix="ips_model_") as tmp:
        tmp_path = Path(tmp)
        in_path = tmp_path / "texts.json"
        out_path = tmp_path / "result.json"
        in_path.write_text(json.dumps(texts), encoding="utf-8")
        cmd = [
            sys.executable,
            "-m",
            "setup.model_matrix",
            "--worker",
            "--name",
            name,
            "--model-id",
            model_id,
            "--kind",
            kind,
            "--texts",
            str(in_path),
            "--out",
            str(out_path),
        ]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(Path.cwd()),
            )
        except subprocess.TimeoutExpired:
            n = len(texts)
            return ModelResult(
                name,
                model_id,
                kind,
                ["error"] * n,
                [0.0] * n,
                [0.0] * n,
                0.0,
                error="Timeout",
            )

        if proc.returncode != 0 or not out_path.exists():
            err = (proc.stderr or proc.stdout or "worker failed").strip()
            n = len(texts)
            return ModelResult(
                name,
                model_id,
                kind,
                ["error"] * n,
                [0.0] * n,
                [0.0] * n,
                0.0,
                error=err[-400:],
            )
        payload = json.loads(out_path.read_text(encoding="utf-8"))
        return ModelResult(**payload)


def _worker_main(args) -> int:
    import json

    configure_runtime()
    texts = json.loads(Path(args.texts).read_text(encoding="utf-8"))
    result = run_model(args.name, args.model_id, args.kind, texts)
    Path(args.out).write_text(json.dumps(asdict(result)), encoding="utf-8")
    return 0  # errors are carried in result.error


def agreement_rate(a: list[str], b: list[str]) -> float:
    if not a:
        return float("nan")
    return float(np.mean([x == y for x, y in zip(a, b)]))


def cohen_kappa(a: list[str], b: list[str]) -> float:
    labels = sorted((set(a) | set(b)) - {"error"})
    if not labels:
        return float("nan")
    idx = {lab: i for i, lab in enumerate(labels)}
    mat = np.zeros((len(labels), len(labels)), dtype=float)
    for x, y in zip(a, b):
        if x not in idx or y not in idx:
            continue
        mat[idx[x], idx[y]] += 1
    if mat.sum() == 0:
        return float("nan")
    po = np.trace(mat) / mat.sum()
    pe = np.sum(mat.sum(axis=0) * mat.sum(axis=1)) / (mat.sum() ** 2)
    if pe >= 1.0:
        return float("nan")
    return float((po - pe) / (1.0 - pe))


def scorecard(frame: pd.DataFrame, result: ModelResult) -> dict:
    if result.error:
        return {
            "model": result.name,
            "model_id": result.model_id,
            "kind": result.kind,
            "error": result.error,
            "n": len(frame),
            "prior_align": np.nan,
            "gold_acc": np.nan,
            "gold_n": 0,
            "neutral_rate": np.nan,
            "mean_confidence": np.nan,
            "mean_margin": np.nan,
            "sec_per_text": np.nan,
            "total_sec": np.nan,
        }

    pred = pd.Series(result.labels, index=frame.index)
    prior_align = float((pred == frame["prior"]).mean())
    gold_mask = frame["gold"].notna()
    gold_n = int(gold_mask.sum())
    gold_acc = (
        float((pred[gold_mask] == frame.loc[gold_mask, "gold"]).mean())
        if gold_n
        else float("nan")
    )
    return {
        "model": result.name,
        "model_id": result.model_id,
        "kind": result.kind,
        "error": None,
        "n": len(frame),
        "prior_align": round(prior_align, 4),
        "gold_acc": round(gold_acc, 4) if gold_n else np.nan,
        "gold_n": gold_n,
        "neutral_rate": round(float((pred == "neutral").mean()), 4),
        "mean_confidence": round(float(np.mean(result.scores)), 4),
        "mean_margin": round(float(np.mean(result.margins)), 4),
        "sec_per_text": round(result.seconds / max(len(frame), 1), 4),
        "total_sec": round(result.seconds, 2),
    }


def pairwise_matrices(
    results: list[ModelResult],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    names = [r.name for r in results if not r.error]
    ok = [r for r in results if not r.error]
    agree = pd.DataFrame(np.nan, index=names, columns=names)
    kappa = pd.DataFrame(np.nan, index=names, columns=names)
    for i, ri in enumerate(ok):
        for j, rj in enumerate(ok):
            agree.iloc[i, j] = round(agreement_rate(ri.labels, rj.labels), 4)
            kappa.iloc[i, j] = round(cohen_kappa(ri.labels, rj.labels), 4)
    return agree, kappa


def composite_rank(scorecard_df: pd.DataFrame) -> pd.DataFrame:
    df = scorecard_df.copy()
    df = df.loc[df["error"].isna()].copy()
    if df.empty:
        return df

    def _z(s: pd.Series) -> pd.Series:
        if s.std(ddof=0) == 0 or s.isna().all():
            return pd.Series(0.0, index=s.index)
        return (s - s.mean()) / s.std(ddof=0)

    gold = df["gold_acc"].fillna(df["prior_align"])
    df["score"] = (
        0.45 * _z(gold)
        + 0.35 * _z(df["prior_align"])
        + 0.10 * _z(df["mean_margin"].fillna(df["mean_confidence"]))
        - 0.05 * _z(df["neutral_rate"])
        - 0.05 * _z(df["sec_per_text"])
    )
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


def _build_bundle(frame: pd.DataFrame, results: list[ModelResult]) -> dict:
    cards = pd.DataFrame([scorecard(frame, r) for r in results])
    ranked = composite_rank(cards)
    agree, kappa = pairwise_matrices(results)
    preds = frame[["text", "source", "prior", "gold"]].copy()
    for r in results:
        if not r.error:
            preds[r.name] = r.labels
            preds[f"{r.name}_conf"] = np.round(r.scores, 4)
    return {
        "eval_frame": frame,
        "scorecard": cards,
        "ranked": ranked,
        "agreement": agree,
        "kappa": kappa,
        "predictions": preds,
        "results": results,
        "winner": None if ranked.empty else str(ranked.iloc[0]["model"]),
    }


def run_matrix(
    challenges_csv: Path | str = Path("output/processed/challenges.csv"),
    expectations_csv: Path | str = Path("output/processed/expectations.csv"),
    *,
    n_per_source: int | None = 80,
    models: list[tuple[str, str, str]] | None = None,
    seed: int = 42,
    isolated: bool = True,
    checkpoint_dir: Path | str | None = Path("output/model_matrix/checkpoints"),
    progress: Callable[[str], None] | None = print,
) -> dict:
    """Evaluate candidate models on the discovery dataset.

    Parameters
    ----------
    n_per_source:
        Sample size per source. ``None`` = full challenges + expectations.
    isolated:
        Run each model in a subprocess (recommended in Jupyter).
    """
    configure_runtime()
    challenges_csv = Path(challenges_csv)
    expectations_csv = Path(expectations_csv)
    models = models or MODEL_CANDIDATES
    ckpt = Path(checkpoint_dir) if checkpoint_dir else None
    if ckpt:
        ckpt.mkdir(parents=True, exist_ok=True)

    frame = load_eval_frame(
        challenges_csv,
        expectations_csv,
        n_per_source=n_per_source,
        seed=seed,
    )
    texts = frame["text"].tolist()
    if progress:
        progress(
            f"Evaluating {len(frame)} records "
            f"({int((frame['source'] == 'challenges').sum())} challenges, "
            f"{int((frame['source'] == 'expectations').sum())} expectations); "
            f"heuristic gold labels: {int(frame['gold'].notna().sum())}; "
            f"isolated={isolated}"
        )

    results: list[ModelResult] = []
    for name, model_id, kind in models:
        ckpt_path = ckpt / f"{name}.csv" if ckpt else None
        if ckpt_path and ckpt_path.exists():
            saved = pd.read_csv(ckpt_path)
            if len(saved) == len(texts) and "label" in saved.columns:
                if progress:
                    progress(f"Skipping {name} (checkpoint hit)")
                results.append(
                    ModelResult(
                        name=name,
                        model_id=model_id,
                        kind=kind,
                        labels=saved["label"].astype(str).tolist(),
                        scores=saved["score"].astype(float).tolist(),
                        margins=saved["margin"].astype(float).tolist(),
                        seconds=float(saved["seconds"].iloc[0])
                        if "seconds" in saved.columns
                        else 0.0,
                        error=None,
                    )
                )
                continue

        if progress:
            progress(f"Running {name} ({kind}) …")
        runner = run_model_isolated if isolated else run_model
        result = runner(name, model_id, kind, texts)
        results.append(result)

        if progress:
            if result.error:
                progress(f"  failed: {result.error[:160]}")
            else:
                progress(
                    f"  done in {result.seconds:.1f}s "
                    f"({result.seconds / max(len(texts), 1):.3f}s/text) "
                    f"prior_align={scorecard(frame, result)['prior_align']}"
                )

        if ckpt_path and not result.error:
            pd.DataFrame(
                {
                    "label": result.labels,
                    "score": result.scores,
                    "margin": result.margins,
                    "seconds": result.seconds,
                }
            ).to_csv(ckpt_path, index=False)

        gc.collect()

    return _build_bundle(frame, results)


def save_matrix_outputs(
    bundle: dict,
    out_dir: Path | str = Path("output/model_matrix"),
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle["scorecard"].to_csv(out_dir / "scorecard.csv", index=False)
    bundle["ranked"].to_csv(out_dir / "ranked.csv", index=False)
    bundle["agreement"].to_csv(out_dir / "agreement_matrix.csv")
    bundle["kappa"].to_csv(out_dir / "kappa_matrix.csv")
    bundle["predictions"].to_csv(out_dir / "predictions.csv", index=False)
    bundle["eval_frame"].to_csv(out_dir / "eval_frame.csv", index=False)
    return out_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="IPS model matrix bake-off")
    parser.add_argument("--worker", action="store_true", help="Score one model (subprocess)")
    parser.add_argument("--name")
    parser.add_argument("--model-id")
    parser.add_argument("--kind")
    parser.add_argument("--texts")
    parser.add_argument("--out")
    parser.add_argument("--n-per-source", type=int, default=40)
    args, _unknown = parser.parse_known_args()

    if args.worker:
        raise SystemExit(_worker_main(args))

    configure_runtime()
    bundle = run_matrix(n_per_source=args.n_per_source, isolated=True)
    out = save_matrix_outputs(bundle)
    print("\n=== Ranked ===")
    cols = [
        c
        for c in [
            "rank",
            "model",
            "prior_align",
            "gold_acc",
            "neutral_rate",
            "sec_per_text",
            "score",
        ]
        if c in bundle["ranked"].columns
    ]
    print(bundle["ranked"][cols].to_string(index=False))
    print(f"\nWinner: {bundle['winner']}")
    print(f"Saved → {out}")
