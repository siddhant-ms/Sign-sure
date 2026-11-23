from __future__ import annotations

import re

LEGAL_EXPANSIONS: dict[str, list[str]] = {
    "fir": ["Section 154 CrPC", "Section 173 BNSS", "First Information Report"],
    "anticipatory bail": ["Section 438 CrPC", "Section 482 BNSS", "pre-arrest bail"],
    "regular bail": ["Section 437 CrPC", "Section 439 CrPC", "Section 480 BNSS", "Section 483 BNSS"],
    "cheque bounce": ["Section 138 Negotiable Instruments Act", "dishonour of cheque", "Section 142 NI Act"],
    "khata": ["Karnataka Land Revenue Act", "mutation", "RTC", "pahani", "BBMP"],
    "mutation": ["Section 128 Karnataka Land Revenue Act", "record of rights", "khata"],
    "injunction": ["Order XXXIX Rule 1 2 CPC", "temporary injunction", "Specific Relief Act"],
    "writ": ["Article 226 Constitution of India", "Article 227", "High Court writ petition"],
    "limitation": ["Limitation Act", "condonation of delay", "Section 5 Limitation Act"],
    "evidence": ["Indian Evidence Act", "Bharatiya Sakshya Adhiniyam", "BSA", "burden of proof"],
    "murder": ["Section 302 IPC", "Section 103 BNS", "culpable homicide"],
    "cheating": ["Section 420 IPC", "Section 318 BNS", "fraudulent inducement"],
    "property transfer": ["Transfer of Property Act", "sale deed", "registration"],
}


def expand_legal_query(query: str) -> str:
    lower = query.lower()
    additions: list[str] = []

    for term, expansions in LEGAL_EXPANSIONS.items():
        if re.search(r"\b" + re.escape(term) + r"\b", lower):
            for exp in expansions:
                if exp.lower() not in lower:
                    additions.append(exp)

    if not additions:
        return query
    return f"{query} | Related statutory terms: {', '.join(additions)}"