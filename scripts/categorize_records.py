"""Hybrid keyword + semantic categorization for IPS discovery records."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

EMBEDDING_MODEL = "all-mpnet-base-v2"
BODY_KEYWORD_MIN = 4
BODY_KEYWORD_LOW_MIN = 2
SEMANTIC_MIN = 0.32  # slightly lower to catch municipal phrasing

CATEGORY_CONFIG = {
    "Workflow & Business Processes": {
        # "workflow": 3, "process": 2, "manual": 2, "duplicate entry": 4, "duplicate work": 4,
        # "re enter": 4, "reenter": 4, "approval": 3, "routing": 3, "queue": 2, "task": 2,
        # "automation": 4, "automate": 4, "streamline": 3, "paper": 2, "paperwork": 2,
        # "bottleneck": 3, "delay": 2, "inefficient": 3, "spreadsheet": 4, "excel tracking": 5,
        # "manual tracking": 5, "payment": 3, "reconciliation": 4, "action": 2,
        # "too many actions": 5, "too many clicks": 5, "human error": 5, "returned": 3,
        # "correction": 3, "rework": 4, "front counter": 4, "walk in": 3, "fee": 3,
        # "bulk": 3, "status gap": 4, "overwhelming": 4, "traffic": 3, "ticketing": 4,

        "workflow" : 5, "business" : 4, "process" : 5, "administration" : 4,
        "certificate" : 2, "access" : 2, "sprinkler report" : 4, "permit" : 2, "permit processing" : 4,
        "permit application" : 3, "permit renewal" : 4, "permit related" : 3, "permit request" : 3,
        "request" : 2, "referral" : 3, "HVAC" : 4, "zoning approval" : 4, "department" : 3,
        "work order" : 4, "license" : 2, "license related" : 3, "license request" : 3, "license processing" : 4,
        "license application" : 3, "license renewal" : 4, "license renewal request" : 4, "license related" : 3,
        "license request" : 3, "license renewal" : 4, "license renewal request" : 4, "license related" : 3,
        "BAA process" : 2, "BAA" : 2, "BAA related" : 3,  "BAA processing" : 4, "baa" : 2,
        "LAW process" : 2, "LAW" : 2, "LAW related" : 3,  "LAW processing" : 4, "law" : 2,
        "payment" : 4, "payment processing" : 4, "payment request" : 3, "payment renewal" : 4, "payment renewal request" : 4, "payment related" : 3, 
        "walkin" : 3, "walk in" : 3, "walk in application" : 3, "walk in related" : 3, "traffic" : 3, "customer": 2, "service" :4, 
        "fee" : 3, "fees" : 3, "fee related" : 3, 
    },

    "Case Management": {
        # "case": 2, "cases": 2, "case management": 5, "inspections": 3, "inspection": 3, "violation": 3, "violations": 3,
        # "complaint": 3, "assignment": 2, "owner": 1, "priority": 1, "status": 2,
        # "follow up": 3, "resolution": 2, "citation": 2, "referral": 5,
        # "referral tracking": 5, "ticket": 3, "hearing": 3, "appeal": 3, "evidence": 4,
        # "applicant": 3, "constituent": 2, "code enforcement": 2, "codes": 1, "permit": 1, "application": 3, 


        "case" : 2, "cases" : 2, "case management" : 5, "inspections" : 3, "inspection" : 3, "violation" : 3, "violations" : 3,
        "complaint" : 4, "complaints" : 4, "complaint management" : 5, "complaint tracking" : 5, "complaint resolution" : 5, "complaint resolution tracking" : 5,
        "BAA case" : 4, "BAA cases" : 4, "BAA case management" : 5, "BAA case tracking" : 5, "BAA case resolution" : 5, "BAA case resolution tracking" : 5, "BAA process" : 2, "BAA related" : 3, "BAA processing" : 4, "baa" : 2,
        "LAW case" : 4, "LAW cases" : 4, "LAW case management" : 5, "LAW case tracking" : 5, "LAW case resolution" : 5, "LAW case resolution tracking" : 5, "LAW process" : 2, "LAW related" : 3, "LAW processing" : 4, "law" : 2,
        "permit" : 3, "permits" : 3, "permit management" : 5, "permit tracking" : 5, "permit resolution" : 5, "permit resolution tracking" : 5, "permit process" : 4, "permit related" : 3, "permit processing" : 4, "permit application" : 3, 
        "permit renewal" : 4, "permit renewal processing" : 3, "permit renewal application" : 4, 
        "license" : 3, "licenses" : 3, "license management" : 5, "license tracking" : 5, "license resolution" : 5, "license resolution tracking" : 5, "license process" : 4, "license related" : 3, "license processing" : 4, "license application" : 3, 
        "inspection" : 2, "inspections" : 2, "inspection management" : 5, "inspection tracking" : 5, "inspection resolution" : 5, "inspection resolution tracking" : 5, "inspection process" : 4, "inspection related" : 3, "inspection processing" : 4, "inspection application" : 3,
        "hearing" : 3, "hearings" : 3, "hearing management" : 5, "hearing tracking" : 5, "hearing resolution" : 5, "hearing resolution tracking" : 5, "hearing process" : 4, "hearing related" : 3, "hearing processing" : 4, "hearing application" : 3,
        "appeal" : 3, "appeals" : 3, "appeal management" : 5, "appeal tracking" : 5, "appeal resolution" : 5, "appeal resolution tracking" : 5, "appeal process" : 4, "appeal related" : 3, "appeal processing" : 4, "appeal application" : 3,
        "applicant" : 2, "applicants" : 2, "application management" : 3, "application tracking" : 3, "application resolution" : 3, "application related" : 3, "application processing" : 3,
        "constituent" : 2, "constituents" : 2, "constituent management" : 5, "constituent tracking" : 5, "constituent resolution" : 5, "constituent resolution tracking" : 5, "constituent process" : 4, "constituent related" : 3, "constituent processing" : 4, "constituent application" : 3,
        "code enforcement" : 2, "code enforcement related" : 3, "code enforcement processing" : 4, "code enforcement application" : 3,
        "codes" : 1, "code" : 1, "code management" : 5, "code tracking" : 5, "code resolution" : 5, "code resolution tracking" : 5, "code process" : 4, "code related" : 3, "code processing" : 4, "code application" : 3,
        "referral" : 5, "referral tracking" : 5, "referral resolution" : 5, "referral resolution tracking" : 5, "referral process" : 4, "referral related" : 3, "referral processing" : 4, "referral application" : 3,
        "records" : 3, "record management" : 3, "record tracking" : 3, "record related" : 3, "record application" : 3,
        "ticket" : 4, "tickets" : 4, "ticket management" : 5, "ticket tracking" : 5, "ticket resolution" : 5, "ticket resolution tracking" : 5, "ticket process" : 4, "ticket related" : 3, "ticket processing" : 4, 
        "follow up" : 3, "follow up tracking" : 3, "follow up resolution" : 3, "follow up resolution tracking" : 3, "follow up process" : 4, "follow up related" : 3, "follow up application" : 3,
        "fee" : 2, "fee processing" : 4, "Rental Registry" : 4, "rental registry" : 4
    },

    "Data Management & Visibility": {
        # "data": 2, "information": 2, "duplicate data": 4, "missing data": 4,
        # "missing information": 4, "visibility": 4, "single source": 5, "history": 2,
        # "property history": 4, "case history": 4, "cross department": 3,
        # "shared information": 4, "information flow": 4, "inconsistent": 3, "accuracy": 3,
        # "fragmented": 5, "fragmentation": 5, "scattered": 5, "spread across": 4,
        # "multiple locations": 4, "historical": 3, "legacy": 3, "migration": 4,
        # "synchronized": 4, "complete picture": 5, "linked": 4, "unlinked": 4,

        "database" : 4, "databases" : 4, "database management" : 5, "data" : 2, "information" : 3, "information management" : 3,
        "hidden data" : 4, "hidden information" : 4, "hidden visibility" : 4, "hidden" : 3, "hidden history" : 2,
        "missing data" : 4, "missing information" : 4, "visibility" : 4, "single source" : 3, "history" : 2,
        "duplicate data" : 5, "duplicate" : 2, "data entry" : 3, "data accuracy" : 5,
        "multiple" : 3, "multiple data" : 4, "multiple source" : 4, "synchronize" : 4, "synchronization" : 4,
        "migration" : 4, "data migration" : 4, "data synchronization" : 4, "migrate": 4,
        "transfer" : 4, "transfer data" : 4, "transfer information" : 4, "cross department" : 3, "department" : 2,
    },

    "Records & Document Management": {
        # "search": 4, "lookup": 4, "find": 3, "retrieve": 3, "locate": 3, "record": 2,
        # "records": 2, "property research": 5, "parcel": 3, "address": 2, "document": 2,
        # "attachment": 2, "photo": 2, "archive": 3, "filter": 2, "plan": 3, "survey": 3,
        # "viewer": 4, "upload": 3, "pdf": 3, "notes": 3, "cheat sheet": 4, "letter": 3,

        "attachment" : 2, "attachments" : 2, "attachment management" : 5, "attachment tracking" : 5, "attachment related" : 3, "attachment processing" : 4,
        "document" : 2, "documents" : 2, "document management" : 5, "document tracking" : 5, "document related" : 3, "document processing" : 4,
        "photo" : 2, "photos" : 2, "photo management" : 5, "photo tracking" : 5, "photo related" : 3, "photo processing" : 4,
        "image" : 2, "images" : 2, "image management" : 5, "image tracking" : 5, "image related" : 3, "image processing" : 4,
        "plan" : 3, "plans" : 3, "plan management" : 5, "plan tracking" : 5, "plan related" : 3, "plan processing" : 4,
        "survey" : 3, "surveys" : 3, "survey management" : 5, "survey tracking" : 5, "survey related" : 3, "survey processing" : 4,
        "letter" : 3, "letters" : 3, "letter management" : 5, "letter tracking" : 5, "letter related" : 3, "letter processing" : 4,
        "notes" : 3, "note" : 3, "note management" : 5, "note tracking" : 5, "note related" : 3, "note processing" : 4,
        "cheat sheet" : 4, "cheat sheets" : 4, "cheat sheet management" : 5, "cheat sheet tracking" : 5, "cheat sheet related" : 3, "cheat sheet processing" : 4,
        "pdf" : 3, "pdfs" : 3, "pdf management" : 5, "pdf tracking" : 5, "pdf related" : 3, "pdf processing" : 4,
        "evidence" : 4, "evidence management" : 5, "evidence tracking" : 5, "evidence related" : 3, "evidence processing" : 4,
        "file" : 2, "files" : 2, "file management" : 5, "file tracking" : 5, "file related" : 3, "file processing" : 4,
        "record" : 3, "records" : 4, "record management" : 5, "record tracking" : 5, "record related" : 3, "record processing" : 4,
        "lookup" : 4, "find" : 3, "retrieve" : 3, "locate" : 3, "property research" : 5, "parcel" : 3, "address" : 2, "filter" : 2,
    },

    "System Integration": {
        # "integration": 5, "integrate": 5, "api": 5, "gis": 4, "onbase": 4, "camino": 4,
        # "as400": 4, "esri": 4, "sync": 4, "interface": 3, "external system": 5,
        # "third party": 4, "multiple systems": 5, "system switching": 5,
        # "too many systems": 5, "too many applications": 5, "aims": 4, "etax": 4,
        # "invoice cloud": 4, "word": 2, "outlook": 3, "foxit": 3, "hamer": 4,

        "Camino" : 4, "Camino integration" : 5, "Camino related" : 3, "Camino processing" : 4, "camino" : 2,
        "GIS" : 4, "GIS integration" : 5, "GIS related" : 3, "GIS processing" : 4, "gis" : 2,
        "OnBase" : 4, "OnBase integration" : 5, "OnBase related" : 3, "OnBase processing" : 4, "onbase" : 2,
        "AS400" : 4, "AS400 integration" : 5, "AS400 related" : 3, "AS400 processing" : 4, "as400" : 2,
        "ESRI" : 4, "ESRI integration" : 5, "ESRI related" : 3, "ESRI processing" : 4, "esri" : 2,
        "AIMS" : 4, "AIMS integration" : 5, "AIMS related" : 3, "AIMS processing" : 4, "aims" : 2,
        "ETAX" : 4, "ETAX integration" : 5, "ETAX related" : 3, "ETAX processing" : 4, "etax" : 2,
        "Invoice Cloud" : 4, "Invoice Cloud integration" : 5, "Invoice Cloud related" : 3, "Invoice Cloud processing" : 4, "invoice cloud" : 2,
        "Sync" : 4, "Sync integration" : 5, "Sync related" : 3, "Sync processing" : 4, "sync" : 2, "Synchronization" : 4, "synchronization" : 4,
        "Interface" : 2, "Interface integration" : 4, "Interface related" : 3, "Interface processing" : 4, "interface" : 2,
        "External system" : 5, "External system integration" : 5, "External system related" : 3, "External system processing" : 4, "external system" : 2,
        "Third party" : 4, "Third party integration" : 5, "Third party related" : 3, "Third party processing" : 4, "third party" : 2,
        "Multiple systems" : 5, "Multiple systems integration" : 5, "Multiple systems related" : 3, "Multiple systems processing" : 4, "multiple systems" : 2,
        "System switching" : 5, "System switching integration" : 5, "System switching related" : 3, "System switching processing" : 4, "system switching" : 2,
        "Too many systems" : 5, "Too many systems integration" : 5, "Too many systems related" : 3, "Too many systems processing" : 4, "too many systems" : 2,
        "Too many applications" : 5, "Too many applications integration" : 5, "Too many applications related" : 3, "Too many applications processing" : 4, "too many applications" : 2,
    },

    "Reporting & Decision Support": {
        # "report": 4, "reporting": 4, "dashboard": 5, "analytics": 4, "metric": 3,
        # "kpi": 4, "statistics": 3, "summary": 2, "export": 3, "excel": 2, "csv": 2,
        # "stats": 4,

        "report" : 4, "reports" : 4, "report management" : 5, "report tracking" : 5, "report related" : 3, "report processing" : 4,
        "dashboard" : 5, "dashboards" : 5, "dashboard management" : 5, "dashboard tracking" : 5, "dashboard related" : 3, "dashboard processing" : 4,
        "analytics" : 4, "analytics management" : 5, "analytics tracking" : 5, "analytics related" : 3, "analytics processing" : 4,
        "metric" : 3, "metrics" : 3, "metric management" : 5, "metric tracking" : 5, "metric related" : 3, "metric processing" : 4,
        "kpi" : 4, "kpis" : 4, "kpi management" : 5, "kpi tracking" : 5, "kpi related" : 3, "kpi processing" : 4,
        "summary" : 2, "summaries" : 2, "summary management" : 5, "summary tracking" : 5, "summary related" : 3, "summary processing" : 4,
        "export" : 3, "exports" : 3, "export management" : 5, "export tracking" : 5, "export related" : 3, "export processing" : 4,
        "excel" : 2, "excel management" : 5, "excel tracking" : 5, "excel related" : 3, "excel processing" : 4,
        "csv" : 2, "csv management" : 5, "csv tracking" : 5, "csv related" : 3, "csv processing" : 4,
        "decision" : 2, "decision making" : 4, "decision support" : 5,
    },

    "Communication & Collaboration": {
        # "communication": 4, "communicate": 4, "notification": 4, "notify": 3, "email": 3,
        # "alert": 3, "coordination": 4, "coordinate": 4, "comment": 2, "discussion": 2,
        # "collaboration": 4, "handoff": 3, "shared": 2, "call": 4, "phone": 4,
        # "disconnect": 5, "call volume": 5, "repeat calls": 5, "customer calls": 4,
        # "department": 3, "interdepartmental": 4, "mailing": 3,

        "phone" : 4, "phone calls" : 4, "calls" : 2, "communication" : 4, "coomunicate" : 4, "coordinate": 4, "coordination" : 3,
        "notification" : 4, "notes" : 2, "comment" : 3, "mail" : 3, "disconnect": 4, "connect" : 4,
    },

    "Scheduling & Resource Management": {
        # "schedule": 5, "scheduling": 5, "calendar": 3, "appointment": 3, "availability": 3,
        # "dispatch": 3, "route": 3, "deadline": 2, "timeline": 2, "workload": 3,
        # "reschedule": 3, "tickler": 5, "tickle": 5, "reminder": 5, "due date": 4,
        # "seasonal": 3, "60-day": 4, "60 day": 4,

        "schedule": 5, "scheduling" : 5, "calendar" : 4, "appointment": 4, "inspection" : 3, "availability" : 3,
        "reschedule": 3, "tickler": 5, "tickle": 5, "reminder": 5, "date": 2, "workload" : 2, "outlook": 3,
        "complaint inspection": 5, "60-day": 4, "60 day": 4, 

    },
    "User Experience & Performance": {
        "slow": 5, "performance": 4, "lag": 5, "freeze": 6, "crash": 6, "timeout": 5,
        "loading": 3, "usability": 4, "user friendly": 5, "navigation": 3,
        "click": 2, "screen": 2, "confusing": 3, "easy": 2, "difficult": 2, "mobile": 4,
        "tablet": 4, "phone" : 4, "desktop": 3, "internet": 4, "offline": 5, "network": 4,
        "connectivity": 4, "shutdown": 5, "restart": 5, "reboot": 5, "exception": 5,
        "memory": 5, "bug": 4, "tab": 3, "large plans": 4, "zoom": 3,
        "display": 3, "module": 4, "dropdown": 3, "action button": 4,
        "cluttered": 4, "organized": 3, "intuitive" : 5, "nonintuitive": 5, "in field" : 4, "UI":5, "ui":5, "user experience": 4,
        "manual process": 5, "manual": 2, "archaic" : 4, "action" : 4, "actions" : 4, "log" : 3, "button" : 5,
        "user" : 2, "account" : 2, "search" : 4, "symbol": 3
        
    },
    "Training & Documentation": {
        "training": 5, "train" : 5, "documentation": 5, "document": 5, "guide": 3, "instruction": 3,
        "support": 2, "knowledge": 4, "faq": 3, "onboarding": 4, "reference": 2, "sop": 5,
        "cheatsheet": 4,
    }
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

    work["categorize_text"] = body.astype(str)

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
