from __future__ import annotations

import re
from pathlib import Path

from app.models import Bench, DocMetadata, DocType, ExtractedDocument, Jurisdiction, LegalDomain

DOMAIN_KEYWORDS: dict[LegalDomain, list[str]] = {
    "criminal": [
        "bharatiya nyaya sanhita",
        "bns",
        "penal code",
        "ipc",
        "crpc",
        "bnss",
        "bail",
        "cognizable",
        "fir",
        "offence",
        "offense",
        "imprisonment",
        "accused",
        "cheque",
        "negotiable instruments",
        "section 138",
    ],
    "property": [
        "transfer of property",
        "land revenue",
        "khata",
        "mutation",
        "registration act",
        "stamp act",
        "tenant",
        "landlord",
        "eviction",
        "lease",
        "mortgage",
        "immovable property",
        "bbmp",
        "acquisition",
    ],
    "procedural": [
        "civil procedure",
        "cpc",
        "order xxxix",
        "limitation act",
        "evidence act",
        "bharatiya sakshya",
        "bsa",
        "injunction",
        "decree",
        "appeal",
        "writ",
        "jurisdiction",
        "plaint",
    ],
    "constitutional": [
        "constitution",
        "article 226",
        "article 227",
        "article 14",
        "article 19",
        "article 21",
        "fundamental rights",
        "mandamus",
        "certiorari",
        "habeas corpus",
    ],
    "consumer": [
        "consumer protection",
        "deficiency in service",
        "unfair trade practice",
        "consumer forum",
        "consumer commission",
    ],
    "administrative": [
        "lokayukta",
        "tribunal",
        "municipal",
        "panchayat",
        "service rules",
        "disciplinary",
        "government servant",
        "notification",
    ],
    "civil": [
        "contract",
        "damages",
        "specific relief",
        "partnership",
        "company",
        "tort",
        "recovery of money",
        "partition",
        "succession",
    ],
}


def parse_folder_hierarchy(file_path: Path) -> tuple[Jurisdiction, DocType, Bench | None, str | None]:
    parts_lower = [p.lower() for p in file_path.parts]
    joined = "/".join(parts_lower)

    bench: Bench | None = None
    court: str | None = None

    if "category 1" in joined or "central acts" in joined:
        return "central", "act", None, None

    if "category 3" in joined or "high court judgments" in joined or "karnataka high court" in joined:
        court = "High Court of Karnataka"
        if any("bengaluru" in p or "bangalore" in p for p in parts_lower):
            bench = "bengaluru"
        elif any("dharwad" in p for p in parts_lower):
            bench = "dharwad"
        elif any("kalaburagi" in p or "gulbarga" in p for p in parts_lower):
            bench = "kalaburagi"
        return "karnataka", "judgment", bench, court

    if "category 2" in joined or "karnataka state acts" in joined or "karnataka acts" in joined:
        return "karnataka", "act", None, None

    return "central", "act", None, None


def _clean_title_candidate(line: str) -> str:
    line = re.sub(r"\s+", " ", line).strip(" -–—:;,.")
    return line


def extract_act_title_and_year(cover_text: str, file_name: str) -> tuple[str, int | None]:
    # IndiaCode filenames often look like a1908-05.pdf or a1881-26.pdf
    stem_match = re.match(r"^a(\d{4})-\d+", file_name.lower())
    file_year = int(stem_match.group(1)) if stem_match else None

    lines = [_clean_title_candidate(ln) for ln in cover_text.splitlines() if _clean_title_candidate(ln)]
    title_candidates: list[str] = []

    for i, ln in enumerate(lines[:35]):
        if re.match(r"^(THE\s+[A-Z][A-Za-z0-9\s(),\-–]+ACT,?\s*\d{4})", ln):
            title_candidates.append(ln)
        elif re.match(r"^THE\s+[A-Z][A-Z\s(),\-–]+ACT", ln):
            combined = ln
            if i + 1 < len(lines) and re.match(r"^(\d{4}|OF\s+\d{4}|,?\s*\d{4})", lines[i + 1]):
                combined = f"{combined}, {lines[i + 1]}".replace(",,", ",")
            title_candidates.append(combined)

    chosen_title = title_candidates[0] if title_candidates else ""
    if not chosen_title:
        # Fallback to first substantial non-header line
        for ln in lines[:15]:
            if len(ln) >= 8 and not re.search(r"indiacode|ministry|gazette|page\s+\d+", ln, re.I):
                chosen_title = ln
                break

    if not chosen_title:
        chosen_title = Path(file_name).stem.replace("_", " ").replace("-", " ").title()

    year_match = re.search(r"\b(18\d{2}|19\d{2}|20\d{2})\b", chosen_title)
    if not year_match:
        year_match = re.search(r"\b(18\d{2}|19\d{2}|20\d{2})\b", cover_text[:1500])

    year = int(year_match.group(1)) if year_match else file_year
    return chosen_title, year


def extract_judgment_details(cover_text: str, file_name: str) -> tuple[str, str | None, str | None, int | None]:
    case_patterns = [
        r"\b(W\.?\s*P\.?\s*(?:No\.?|Nos\.?)?\s*[\d/]+(?:\s*\([A-Z\-]+\))?(?:\s+of\s+\d{4})?)",
        r"\b(Crl\.?\s*P\.?\s*(?:No\.?)?\s*[\d/]+(?:\s+of\s+\d{4})?)",
        r"\b(R\.?\s*F\.?\s*A\.?\s*(?:No\.?)?\s*[\d/]+(?:\s+of\s+\d{4})?)",
        r"\b(M\.?\s*F\.?\s*A\.?\s*(?:No\.?)?\s*[\d/]+(?:\s+of\s+\d{4})?)",
        r"\b(Crl\.?\s*A\.?\s*(?:No\.?)?\s*[\d/]+(?:\s+of\s+\d{4})?)",
        r"\b(W\.?\s*A\.?\s*(?:No\.?)?\s*[\d/]+(?:\s+of\s+\d{4})?)",
        r"\b(R\.?\s*S\.?\s*A\.?\s*(?:No\.?)?\s*[\d/]+(?:\s+of\s+\d{4})?)",
    ]

    case_number: str | None = None
    for pat in case_patterns:
        m = re.search(pat, cover_text[:2500], re.IGNORECASE)
        if m:
            case_number = re.sub(r"\s+", " ", m.group(1)).strip()
            break

    date_match = re.search(
        r"\b(\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December),?\s+\d{4}|\d{2}[/-]\d{2}[/-]\d{4})\b",
        cover_text[:2500],
        re.IGNORECASE,
    )
    judgment_date = date_match.group(1) if date_match else None

    year: int | None = None
    if judgment_date:
        ym = re.search(r"\b(19\d{2}|20\d{2})\b", judgment_date)
        if ym:
            year = int(ym.group(1))
    if year is None and case_number:
        ym = re.search(r"\b(19\d{2}|20\d{2})\b", case_number)
        if ym:
            year = int(ym.group(1))

    # Attempt party names ("X VS Y" or "X v. Y")
    party_match = re.search(
        r"([A-Z][A-Za-z0-9\s.&(),\-]{2,70}?)\s+(?:V/?S\.?|VERSUS|V\.)\s+([A-Z][A-Za-z0-9\s.&(),\-]{2,70}?)(?:\n|,|\s{2,})",
        cover_text[:2500],
        re.IGNORECASE,
    )
    if case_number and party_match:
        p1 = _clean_title_candidate(party_match.group(1))
        p2 = _clean_title_candidate(party_match.group(2))
        doc_title = f"{case_number} — {p1} v. {p2}"
    elif case_number:
        doc_title = f"Karnataka HC Judgment — {case_number}"
    else:
        doc_title = Path(file_name).stem.replace("_", " ").replace("-", " ").title()

    return doc_title, case_number, judgment_date, year


def infer_legal_domain(sample_text: str) -> LegalDomain:
    lower = sample_text.lower()
    scores: dict[LegalDomain, int] = {k: 0 for k in DOMAIN_KEYWORDS}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        for kw in keywords:
            count = len(re.findall(r"\b" + re.escape(kw) + r"\b", lower))
            if count > 0:
                scores[domain] += count

    best_domain = max(scores, key=lambda d: scores[d])
    return best_domain if scores[best_domain] > 0 else "general"


def build_doc_metadata(doc: ExtractedDocument) -> DocMetadata:
    file_path = Path(doc.file_path)
    jurisdiction, doc_type, bench, court = parse_folder_hierarchy(file_path)

    first_two_pages = "\n\n".join(p.text for p in doc.pages[:2] if p.text)
    sample_for_domain = "\n\n".join(p.text for p in doc.pages[:6] if p.text)

    case_number: str | None = None
    judgment_date: str | None = None

    if doc_type == "judgment":
        doc_title, case_number, judgment_date, year = extract_judgment_details(
            first_two_pages, doc.file_name
        )
    else:
        doc_title, year = extract_act_title_and_year(first_two_pages, doc.file_name)

    domain = infer_legal_domain(sample_for_domain)
    doc_id = doc.sha256[:16]

    return DocMetadata(
        doc_id=doc_id,
        file_name=doc.file_name,
        sha256=doc.sha256,
        jurisdiction=jurisdiction,
        doc_type=doc_type,
        bench=bench,
        court=court,
        doc_title=doc_title,
        year=year,
        case_number=case_number,
        judgment_date=judgment_date,
        domain=domain,
    )