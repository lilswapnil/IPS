"""Sentiment scoring + realignment for IPS discovery records.

Uses DistilBERT SST-2 (best on the IPS gold bake-off in
``notebook/analyze_model.ipynb``):
  accuracy 0.767 vs 0.600 for Thi144/sentiment-distilbert-7class.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd
from transformers import pipeline

# Bake-off winner (see output/processed/sentiment_model_leaderboard.csv)
SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

# SST-2 is binary; we surface a soft "neutral" when confidence is low
SENTIMENT_LABELS = {
    "negative": {"scale": -1, "label": "negative", "name": "Negative"},
    "neutral": {"scale": 0, "label": "neutral", "name": "Neutral"},
    "positive": {"scale": 1, "label": "positive", "name": "Positive"},
}

# HF SST-2 pipeline labels → coarse polarity
_SST2_LOOKUP = {
    "LABEL_0": "negative",
    "LABEL_1": "positive",
    "NEGATIVE": "negative",
    "POSITIVE": "positive",
    "negative": "negative",
    "positive": "positive",
}

# Back-compat alias used by older notebook cells
SENTIMENT_LABEL_MAP = {
    "Negative": "negative",
    "Neutral": "neutral",
    "Positive": "positive",
    "LABEL_0": "negative",
    "LABEL_1": "positive",
    "NEGATIVE": "negative",
    "POSITIVE": "positive",
    "0": "negative",
    "1": "positive",
}

# Below this top-class score → treat as neutral (then prior / keywords decide)
CONFIDENCE_MARGIN = 0.65

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
    r"dashboard|notify|helpful if|ideal | future"
    r")\b",
    re.I,
)


def _load_hf_token() -> None:
    """Load HF_TOKEN from .env if present (higher rate limits)."""
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


def resolve_sentiment_label(raw_label) -> dict:
    """Map SST-2 pipeline label → SENTIMENT_LABELS meta."""
    key = str(raw_label).strip()
    polarity = _SST2_LOOKUP.get(key) or _SST2_LOOKUP.get(key.upper()) or _SST2_LOOKUP.get(
        key.lower()
    )
    if polarity is None:
        return SENTIMENT_LABELS["neutral"]
    return SENTIMENT_LABELS[polarity]


def build_sentiment_classifier():
    """Build the DistilBERT SST-2 sentiment-analysis pipeline."""
    _load_hf_token()
    return pipeline(
        "sentiment-analysis",
        model=SENTIMENT_MODEL,
        truncation=True,
        max_length=512,
        top_k=None,  # return both class scores
    )


def _score_one(classifier, text: str) -> tuple[str, float, float, str]:
    """Return (polarity, top_score, margin_over_second, fine_name)."""
    result = classifier(text)
    # pipeline may return list[dict] (top-1) or list[list[dict]] (all classes)
    if result and isinstance(result[0], list):
        ranked = sorted(result[0], key=lambda r: r["score"], reverse=True)
    else:
        ranked = result if isinstance(result, list) else [result]

    top_meta = resolve_sentiment_label(ranked[0]["label"])
    top_score = float(ranked[0]["score"])
    second = float(ranked[1]["score"]) if len(ranked) > 1 else 0.0
    margin = top_score - second

    # Soft neutral band for binary SST-2 when the model is unsure
    if top_score < CONFIDENCE_MARGIN:
        return "neutral", top_score, margin, "Neutral"
    return top_meta["label"], top_score, margin, top_meta["name"]


def predict_sentiment(
    texts: list,
    *,
    source: str,
    classifier=None,
    batch_size: int = 8,
) -> list[str]:
    """Score texts with SST-2 DistilBERT + source prior + keyword hints.

    source: \"challenges\" → prior negative; \"expectations\" → prior positive
    """
    if classifier is None:
        classifier = build_sentiment_classifier()

    prior = "negative" if source == "challenges" else "positive"
    out: list[str] = []

    clean = [("" if pd.isna(t) else str(t).strip()) for t in texts]
    for i in range(0, len(clean), batch_size):
        batch = clean[i : i + batch_size]
        for text in batch:
            if not text:
                out.append("neutral")
                continue

            label, score, margin, _name = _score_one(classifier, text)

            # Very flat binary scores → prefer dataset prior
            if margin < 0.05:
                label = prior

            # Keyword overrides when model is uncertain / conflicted
            pain_hit = bool(PAIN_HINTS.search(text))
            wish_hit = bool(WISH_HINTS.search(text))
            if label == "neutral":
                if pain_hit and not wish_hit:
                    label = "negative"
                elif wish_hit and not pain_hit:
                    label = "positive"
                else:
                    label = prior
            elif source == "challenges" and label == "positive" and pain_hit and not wish_hit:
                label = "negative"
            elif source == "expectations" and label == "negative" and wish_hit and not pain_hit:
                label = "positive"

            out.append(label)

    return out


def realign_by_sentiment(
    challenges: pd.DataFrame,
    expectations: pd.DataFrame,
    *,
    only_source: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Move positive challenges → expectations; negative expectations → challenges.

    If ``only_source`` is set (e.g. ``\"meeting_notes\"``), only that source is
    realigned; other rows stay in their original frames.
    """
    shared = ["focus_group", "processed_text", "sentiment"]
    if "department" in challenges.columns and "department" in expectations.columns:
        shared = ["department"] + shared
    if only_source is not None:
        if "source" not in challenges.columns or "source" not in expectations.columns:
            raise ValueError(
                "only_source requires a `source` column on both frames "
                f"(got challenges={list(challenges.columns)}, "
                f"expectations={list(expectations.columns)})"
            )
        ch_other = challenges.loc[challenges["source"] != only_source].copy()
        ex_other = expectations.loc[expectations["source"] != only_source].copy()
        ch_target = challenges.loc[challenges["source"] == only_source].copy()
        ex_target = expectations.loc[expectations["source"] == only_source].copy()
        ch_target, ex_target = realign_by_sentiment(ch_target, ex_target)
        challenges = pd.concat([ch_other, ch_target], ignore_index=True)
        expectations = pd.concat([ex_other, ex_target], ignore_index=True)
        return challenges, expectations

    if "source" in challenges.columns and "source" in expectations.columns:
        shared = shared + ["source"]

    to_exp = (
        challenges.loc[challenges["sentiment"] == "positive", shared + ["pain_points"]]
        .rename(columns={"pain_points": "expectations"})
    )
    to_pain = (
        expectations.loc[expectations["sentiment"] == "negative", shared + ["expectations"]]
        .rename(columns={"expectations": "pain_points"})
    )

    challenges = pd.concat(
        [
            challenges.loc[challenges["sentiment"] != "positive", shared + ["pain_points"]],
            to_pain,
        ],
        ignore_index=True,
    )
    expectations = pd.concat(
        [
            expectations.loc[expectations["sentiment"] != "negative", shared + ["expectations"]],
            to_exp,
        ],
        ignore_index=True,
    )
    return challenges, expectations


def run_sentiment(
    challenges: pd.DataFrame,
    expectations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    challenges = challenges.copy()
    expectations = expectations.copy()

    # Score on original text (full characters); fall back to processed_text
    if "processed_text" in challenges.columns:
        challenge_text = challenges["pain_points"].fillna("").astype(str)
        challenge_proc = challenges["processed_text"].fillna("").astype(str)
        challenge_text = challenge_text.where(challenge_text.str.strip() != "", challenge_proc)
    else:
        challenge_text = challenges["pain_points"].fillna("").astype(str)

    if "processed_text" in expectations.columns:
        expectation_text = expectations["expectations"].fillna("").astype(str)
        expectation_proc = expectations["processed_text"].fillna("").astype(str)
        expectation_text = expectation_text.where(
            expectation_text.str.strip() != "", expectation_proc
        )
    else:
        expectation_text = expectations["expectations"].fillna("").astype(str)

    classifier = build_sentiment_classifier()
    print(f"Scoring challenges with {SENTIMENT_MODEL}…")
    challenges["sentiment"] = predict_sentiment(
        challenge_text.tolist(), source="challenges", classifier=classifier
    )
    print("Scoring expectations…")
    expectations["sentiment"] = predict_sentiment(
        expectation_text.tolist(), source="expectations", classifier=classifier
    )
    return challenges, expectations
