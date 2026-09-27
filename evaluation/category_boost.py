#!/usr/bin/env python3
"""Query-time authority-class classifier for the 'other' lane, multi-label, keyword-based.
Only classifies the MINORITY classes with reliable lexical signal (OFFICIAL_REGULATOR_GUIDANCE,
OFFICIAL_TECHNICAL_GUIDANCE, OFFICIAL_WORKFLOW) - deliberately does NOT try to predict
OFFICIAL_GOVERNMENT_GUIDANCE since it's already the dominant class in the pool (62%) and
boosting an already-dominant class doesn't sharpen ranking. UNKNOWN (no trigger fires) means
no boost applied - never forces a guess.
"""
REGULATOR_KEYWORDS = [
    "foi", "freedom of information", "judicial review", "administrative court",
    "technology and construction court", " tcc ", "criminal offence", "prosecutor",
    "misconduct in public office", "prosecution", "eir", "environmental information",
]
TECHNICAL_KEYWORDS = ["wto", "government procurement agreement", " gpa ", "treaty state"]
WORKFLOW_KEYWORDS = [
    "practical process", "what should we actually do", "day to day", "day-to-day",
    "operationally", "quick quote", "procurement journey", "scotland", "scottish",
    "checking off", "should be checking", "what's the practical", "close things out",
    "from before", "sign-off", "what should our team",
]

def classify_query(query: str) -> set[str]:
    q = (query or "").lower()
    out = set()
    if any(k in q for k in REGULATOR_KEYWORDS):
        out.add("OFFICIAL_REGULATOR_GUIDANCE")
    if any(k in q for k in TECHNICAL_KEYWORDS):
        out.add("OFFICIAL_TECHNICAL_GUIDANCE")
    if any(k in q for k in WORKFLOW_KEYWORDS):
        out.add("OFFICIAL_WORKFLOW")
    return out
