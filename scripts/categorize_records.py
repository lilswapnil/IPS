"""Hybrid keyword + semantic categorization for IPS discovery records."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

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
        "case": 2, "cases": 2, "case management": 5, "inspections": 3, "inspection": 3, "violation": 3, "violations": 3,
        "complaint": 3, "assignment": 2, "owner": 1, "priority": 1, "status": 2,
        "follow up": 3, "resolution": 2, "citation": 2, "referral": 5,
        "referral tracking": 5, "ticket": 3, "hearing": 3, "appeal": 3, "evidence": 4,
        "applicant": 3, "constituent": 2, "code enforcement": 2, "codes": 1, "permit": 1, "application": 3, 
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
        "slow": 5, "performance": 4, "lag": 5, "freeze": 6, "crash": 6, "timeout": 5,
        "loading": 3, "usability": 4, "user friendly": 5, "navigation": 3,
        "click": 2, "screen": 2, "confusing": 3, "easy": 2, "difficult": 2, "mobile": 4,
        "tablet": 4, "desktop": 3, "internet": 4, "offline": 5, "network": 4,
        "connectivity": 4, "shutdown": 5, "restart": 5, "reboot": 5, "exception": 5,
        "memory": 5, "bug": 4, "tab": 3, "large plans": 4, "zoom": 3,
        "display": 3, "module": 4, "dropdown": 3, "action button": 4,
        "cluttered": 4, "organized": 3,
    },
    "Training & Documentation": {
        "training": 5, "train" : 5, "documentation": 5, "document": 5, "guide": 3, "instruction": 3,
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
    """Substring for multi-word phrases; word-boundary (+ English inflection) for tokens."""
    if " " in keyword:
        return keyword in text_lower
    kw = re.escape(keyword)
    if len(keyword) >= 4:
        # crash→crashes/crashing; freeze→freezes/freezing (drop silent e)
        if re.search(rf"\b{kw}(?:ing|es|ed|s)?\b", text_lower):
            return True
        if keyword.endswith("e") and re.search(
            rf"\b{re.escape(keyword[:-1])}ing\b", text_lower
        ):
            return True
        return False
    return bool(re.search(rf"\b{kw}(?:es|s)?\b", text_lower))


# Symptom words that should beat contextual nouns (e.g. "scheduling crashes" → UX)
_UX_FAILURE_RE = re.compile(
    r"\b("
    r"crash|crashes|crashing|freeze|freezes|freezing|frozen|"
    r"slow|slows|lag|lags|lagging|timeout|timeouts|"
    r"bug|bugs|reboot|reboots|restart|restarts|shutdown"
    r")\b",
    re.I,
)


def assign_keyword(text: str, category_config: dict) -> tuple[str, float]:
    text_lower = str(text).lower()
    scores: dict[str, float] = {}
    for category, keywords in category_config.items():
        score = float(sum(w for kw, w in keywords.items() if _keyword_hit(text_lower, kw)))
        if score > 0:
            scores[category] = score

    if not scores:
        return "Other", 0.0

    # Reliability/performance failures outrank setting/context keywords
    ux = "User Experience & Performance"
    if _UX_FAILURE_RE.search(text_lower) and ux in scores:
        best_other = max((s for c, s in scores.items() if c != ux), default=0.0)
        if scores[ux] >= best_other - 2:
            scores[ux] = max(scores[ux], best_other) + 2

    best_category = max(scores, key=scores.get)
    return best_category, scores[best_category]


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
