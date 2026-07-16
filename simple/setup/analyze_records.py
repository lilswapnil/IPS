"""High-quality sentiment + categorization for IPS discovery records.

Designed for output/processed/challenges.csv and expectations.csv.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

# ---------------------------------------------------------------------------
# Sentiment — 7-class DistilBERT + source prior + keyword hints
# ---------------------------------------------------------------------------
SENTIMENT_MODEL = "Thi144/sentiment-distilbert-7class"
# Kept for notebook docs / backwards compatibility (not used by this classifier)
SENTIMENT_HYPOTHESIS = "This workplace discovery note indicates {}."

# Index / HF label → coarse polarity used downstream
SENTIMENT_LABELS = {
    0: {"scale": -3, "label": "negative", "name": "Very Negative"},
    1: {"scale": -2, "label": "negative", "name": "Negative"},
    2: {"scale": -1, "label": "negative", "name": "Slightly Negative"},
    3: {"scale": 0, "label": "neutral", "name": "Neutral"},
    4: {"scale": 1, "label": "positive", "name": "Slightly Positive"},
    5: {"scale": 2, "label": "positive", "name": "Positive"},
    6: {"scale": 3, "label": "positive", "name": "Very Positive"},
}

# Alternate keys the HF pipeline may return
_SENTIMENT_LOOKUP = {
    **{str(i): meta for i, meta in SENTIMENT_LABELS.items()},
    **{f"LABEL_{i}": meta for i, meta in SENTIMENT_LABELS.items()},
    **{meta["name"]: meta for meta in SENTIMENT_LABELS.values()},
    **{meta["name"].lower(): meta for meta in SENTIMENT_LABELS.values()},
    "negative": SENTIMENT_LABELS[1],
    "neutral": SENTIMENT_LABELS[3],
    "positive": SENTIMENT_LABELS[5],
}

# Back-compat alias used by older notebook cells
SENTIMENT_LABEL_MAP = {
    meta["name"]: meta["label"] for meta in SENTIMENT_LABELS.values()
} | {
    f"LABEL_{i}": meta["label"] for i, meta in SENTIMENT_LABELS.items()
} | {
    str(i): meta["label"] for i, meta in SENTIMENT_LABELS.items()
}

# If the top score is weak, trust the source dataset prior
CONFIDENCE_MARGIN = 0.45

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
    """Map pipeline label (LABEL_n / name / int) → SENTIMENT_LABELS meta."""
    key = raw_label if not isinstance(raw_label, str) else raw_label
    if isinstance(key, str) and key.isdigit():
        key = int(key)
    if isinstance(key, int) and key in SENTIMENT_LABELS:
        return SENTIMENT_LABELS[key]
    meta = _SENTIMENT_LOOKUP.get(str(raw_label)) or _SENTIMENT_LOOKUP.get(
        str(raw_label).lower()
    )
    if meta is None:
        return {"scale": 0, "label": "neutral", "name": str(raw_label)}
    return meta


def build_sentiment_classifier():
    """Build the 7-class sentiment-analysis pipeline."""
    _load_hf_token()
    return pipeline(
        "sentiment-analysis",
        model=SENTIMENT_MODEL,
        truncation=True,
        max_length=512,
        top_k=None,  # return all 7 class scores when supported
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
    return top_meta["label"], top_score, margin, top_meta["name"]


def predict_sentiment(
    texts: list,
    *,
    source: str,
    classifier=None,
    batch_size: int = 8,
) -> list[str]:
    """Score texts with 7-class sentiment + source prior + keyword hints.

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

            # Low confidence → prefer dataset prior (notes filed as pain vs wish)
            if score < CONFIDENCE_MARGIN or margin < 0.05:
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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Move positive challenges → expectations; negative expectations → challenges."""
    shared = ["focus_group", "processed_text", "sentiment"]

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


# ---------------------------------------------------------------------------
# Categorization — hybrid keywords + semantics
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "all-mpnet-base-v2"
TITLE_KEYWORD_MIN = 4
BODY_KEYWORD_MIN = 4
BODY_KEYWORD_LOW_MIN = 2
SEMANTIC_MIN = 0.32  # slightly lower to catch municipal phrasing

CATEGORY_CONFIG = {
    "Workflow & Business Processes": {
        "workflow": 3, "process": 2, "manual": 2, "duplicate entry": 4, "duplicate work": 4,
        "re enter": 4, "reenter": 4, "approval": 3, "routing": 3, "queue": 2, "task": 2,
        "automation": 4, "automate": 4, "streamline": 3, "paper": 2, "paperwork": 2,
        "bottleneck": 3, "delay": 2, "inefficient": 3, "spreadsheet": 4, "excel tracking": 5,
        "manual tracking": 5, "payment": 3, "reconciliation": 4, "action": 2,
        "too many actions": 5, "too many clicks": 5, "human error": 5, "returned": 3,
        "correction": 3, "rework": 4, "front counter": 4, "walk in": 3, "fee": 3,
        "bulk": 3, "status gap": 4, "overwhelming": 4, "traffic": 3, "ticketing": 4,
    },
    "Case Management": {
        "case": 3, "case management": 5, "inspection": 3, "inspector": 3, "violation": 3,
        "complaint": 3, "assignment": 2, "owner": 2, "priority": 2, "status": 2,
        "follow up": 3, "resolution": 2, "permit": 2, "citation": 2, "referral": 5,
        "referral tracking": 5, "ticket": 3, "hearing": 3, "appeal": 3, "evidence": 4,
        "applicant": 3, "constituent": 3, "code enforcement": 4,
    },
    "Data Management & Visibility": {
        "data": 2, "information": 2, "duplicate data": 4, "missing data": 4,
        "missing information": 4, "visibility": 4, "single source": 5, "history": 2,
        "property history": 4, "case history": 4, "cross department": 3,
        "shared information": 4, "information flow": 4, "inconsistent": 3, "accuracy": 3,
        "fragmented": 5, "fragmentation": 5, "scattered": 5, "spread across": 4,
        "multiple locations": 4, "historical": 3, "legacy": 3, "migration": 4,
        "synchronized": 4, "complete picture": 5, "linked": 4, "unlinked": 4,
    },
    "Records & Document Management": {
        "search": 4, "lookup": 4, "find": 3, "retrieve": 3, "locate": 3, "record": 2,
        "records": 2, "property research": 5, "parcel": 3, "address": 2, "document": 2,
        "attachment": 2, "photo": 2, "archive": 3, "filter": 2, "plan": 3, "survey": 3,
        "viewer": 4, "upload": 3, "pdf": 3, "notes": 3, "cheat sheet": 4, "letter": 3,
    },
    "System Integration": {
        "integration": 5, "integrate": 5, "api": 5, "gis": 4, "onbase": 4, "camino": 4,
        "as400": 4, "esri": 4, "sync": 4, "interface": 3, "external system": 5,
        "third party": 4, "multiple systems": 5, "system switching": 5,
        "too many systems": 5, "too many applications": 5, "aims": 4, "etax": 4,
        "invoice cloud": 4, "word": 2, "outlook": 3, "foxit": 3, "hamer": 4,
    },
    "Reporting & Decision Support": {
        "report": 4, "reporting": 4, "dashboard": 5, "analytics": 4, "metric": 3,
        "kpi": 4, "statistics": 3, "summary": 2, "export": 3, "excel": 2, "csv": 2,
        "stats": 4,
    },
    "Communication & Collaboration": {
        "communication": 4, "communicate": 4, "notification": 4, "notify": 3, "email": 3,
        "alert": 3, "coordination": 4, "coordinate": 4, "comment": 2, "discussion": 2,
        "collaboration": 4, "handoff": 3, "shared": 2, "call": 4, "phone": 4,
        "disconnect": 5, "call volume": 5, "repeat calls": 5, "customer calls": 4,
        "department": 3, "interdepartmental": 4, "mailing": 3,
    },
    "Scheduling & Resource Management": {
        "schedule": 4, "scheduling": 4, "calendar": 3, "appointment": 3, "availability": 3,
        "dispatch": 3, "route": 3, "deadline": 2, "timeline": 2, "workload": 3,
        "reschedule": 3, "tickler": 5, "tickle": 5, "reminder": 5, "due date": 4,
        "seasonal": 3, "60-day": 4, "60 day": 4,
    },
    "User Experience & Performance": {
        "slow": 4, "performance": 4, "lag": 4, "freeze": 4, "crash": 4, "timeout": 4,
        "loading": 3, "usability": 4, "user friendly": 5, "navigation": 3,
        "click": 2, "screen": 2, "confusing": 3, "easy": 2, "difficult": 2, "mobile": 4,
        "tablet": 4, "desktop": 3, "internet": 4, "offline": 5, "network": 4,
        "connectivity": 4, "shutdown": 5, "restart": 5, "reboot": 5, "exception": 5,
        "memory": 5, "bug": 4, "tab": 3, "large plans": 4, "zoom": 3,
        "display": 3, "module": 4, "dropdown": 3, "action button": 4,
        "cluttered": 4, "organized": 3,
    },
    "Training & Documentation": {
        "training": 5, "documentation": 5, "guide": 3, "instruction": 3,
        "support": 2, "knowledge": 4, "faq": 3, "onboarding": 4, "reference": 2, "sop": 5,
    },
}

# Natural-language theme blurbs → better semantic matches than keyword dumps
CATEGORY_BLURBS = {
    "Workflow & Business Processes": (
        "Inefficient or manual workflows, approvals, routing, duplicate data entry, "
        "paper processes, fees, and day-to-day operational bottlenecks in permitting "
        "and code enforcement."
    ),
    "Case Management": (
        "Managing cases, inspections, violations, complaints, tickets, hearings, "
        "appeals, evidence, referrals, and tracking ownership or status of work."
    ),
    "Data Management & Visibility": (
        "Missing, duplicate, fragmented, or hard-to-see data; need for a single "
        "source of truth, property or case history, and cross-department visibility."
    ),
    "Records & Document Management": (
        "Finding records, parcels, addresses, documents, photos, PDFs, attachments, "
        "plans, notes, letters, and searching or uploading files."
    ),
    "System Integration": (
        "Connecting IPS with other systems like Camino, GIS, OnBase, AS400, AIMS, "
        "Outlook, or reducing switching between too many applications."
    ),
    "Reporting & Decision Support": (
        "Reports, dashboards, analytics, KPIs, statistics, summaries, and exporting "
        "data for management decisions."
    ),
    "Communication & Collaboration": (
        "Email, notifications, phone calls, handoffs between departments, "
        "coordination with codes or constituents, and shared comments."
    ),
    "Scheduling & Resource Management": (
        "Calendars, appointments, inspection scheduling, reminders, ticklers, "
        "deadlines, workload, and due dates."
    ),
    "User Experience & Performance": (
        "Slow performance, crashes, freezes, confusing screens, too many clicks, "
        "mobile or tablet use, bugs, and usability of the software."
    ),
    "Training & Documentation": (
        "Training needs, SOPs, guides, documentation, onboarding, and knowledge "
        "for staff using the system."
    ),
}

CATEGORY_NAMES = list(CATEGORY_CONFIG.keys())
CATEGORY_DESCRIPTIONS = {
    name: f"{name}. {CATEGORY_BLURBS[name]} Key terms: " + ", ".join(
        sorted(CATEGORY_CONFIG[name], key=lambda k: -CATEGORY_CONFIG[name][k])[:12]
    )
    for name in CATEGORY_NAMES
}


def extract_title(text: str) -> str:
    text = str(text)
    return text.split(":")[0].strip() if ":" in text else text.strip()


def _keyword_hit(text_lower: str, keyword: str) -> bool:
    """Substring for multi-word phrases; word-boundary (+ optional plural) for short tokens."""
    if " " in keyword or len(keyword) >= 6:
        return keyword in text_lower
    # Match "case"/"cases", "fee"/"fees" but not "fee" inside "feedback"
    return bool(re.search(rf"\b{re.escape(keyword)}s?\b", text_lower))


def assign_keyword(text: str, category_config: dict) -> tuple[str, float]:
    text_lower = str(text).lower()
    best_category, best_score = "Other", 0.0
    for category, keywords in category_config.items():
        score = sum(w for kw, w in keywords.items() if _keyword_hit(text_lower, kw))
        if score > best_score:
            best_category, best_score = category, float(score)
    return best_category, best_score


def pick_category(row: pd.Series) -> tuple[str, str, float]:
    if row["title_score"] >= TITLE_KEYWORD_MIN:
        return row["title_category"], "title_keyword", row["title_score"]
    if row["keyword_score"] >= BODY_KEYWORD_MIN:
        return row["keyword_category"], "keyword", row["keyword_score"]
    if row["semantic_score"] >= SEMANTIC_MIN:
        return row["semantic_category"], "semantic", row["semantic_score"]
    if row["keyword_score"] >= BODY_KEYWORD_LOW_MIN:
        return row["keyword_category"], "keyword_low", row["keyword_score"]
    return row["semantic_category"], "semantic_fallback", row["semantic_score"]


def categorize_dataframe(
    frame: pd.DataFrame,
    text_column: str,
    model: SentenceTransformer,
    cat_embeddings: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    work = frame.copy()
    # Prefer original text for signal; fall back to processed_text
    body = work[text_column].fillna("").astype(str)
    if "processed_text" in work.columns:
        proc = work["processed_text"].fillna("").astype(str)
        body = pd.Series(
            np.where(body.str.len() >= proc.str.len(), body, proc),
            index=work.index,
        )

    work["title"] = body.map(extract_title)
    work["categorize_text"] = work["title"] + ". " + body.astype(str)

    text_embeddings = model.encode(
        work["categorize_text"].tolist(),
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=64,
    )
    sims = cosine_similarity(text_embeddings, cat_embeddings)
    work["semantic_category"] = [CATEGORY_NAMES[i] for i in sims.argmax(axis=1)]
    work["semantic_score"] = sims.max(axis=1)

    kw = body.map(lambda v: assign_keyword(v, CATEGORY_CONFIG))
    work["keyword_category"] = kw.map(lambda r: r[0])
    work["keyword_score"] = kw.map(lambda r: r[1])

    title_kw = work["title"].map(lambda v: assign_keyword(v, CATEGORY_CONFIG))
    work["title_category"] = title_kw.map(lambda r: r[0])
    work["title_score"] = title_kw.map(lambda r: r[1])

    # Soft agreement boost: when keyword and semantic agree, prefer that label
    agree = (
        (work["keyword_category"] == work["semantic_category"])
        & (work["keyword_category"] != "Other")
        & (work["keyword_score"] >= BODY_KEYWORD_LOW_MIN)
    )
    work.loc[agree, "Category"] = work.loc[agree, "keyword_category"]
    work.loc[agree, "Category_Method"] = "keyword_semantic_agree"
    work.loc[agree, "Category_Confidence"] = (
        work.loc[agree, "keyword_score"] + work.loc[agree, "semantic_score"]
    )

    disagree = ~agree
    if disagree.any():
        picked = work.loc[disagree].apply(pick_category, axis=1, result_type="expand")
        work.loc[disagree, "Category"] = picked[0].values
        work.loc[disagree, "Category_Method"] = picked[1].values
        work.loc[disagree, "Category_Confidence"] = picked[2].values

    return work, text_embeddings


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
    print("Scoring challenges…")
    challenges["sentiment"] = predict_sentiment(
        challenge_text.tolist(), source="challenges", classifier=classifier
    )
    print("Scoring expectations…")
    expectations["sentiment"] = predict_sentiment(
        expectation_text.tolist(), source="expectations", classifier=classifier
    )
    return challenges, expectations


def run_categorize(
    challenges: pd.DataFrame,
    expectations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, SentenceTransformer]:
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    cat_embeddings = embedder.encode(
        list(CATEGORY_DESCRIPTIONS.values()),
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    challenges, challenge_embeddings = categorize_dataframe(
        challenges, "pain_points", embedder, cat_embeddings
    )
    expectations, _ = categorize_dataframe(
        expectations, "expectations", embedder, cat_embeddings
    )
    return challenges, expectations, challenge_embeddings, embedder
