from __future__ import annotations

import re
import unicodedata

_ABBREVIATION_MAP: dict[str, str] = {
    "tp hcm": "thanh pho ho chi minh",
    "tphcm": "thanh pho ho chi minh",
    "tp. hcm": "thanh pho ho chi minh",
    "hcm": "ho chi minh",
    "hn": "ha noi",
    "dn": "da nang",
    "sg": "sai gon",
    "q1": "quan 1",
    "q2": "quan 2",
    "q3": "quan 3",
    "q4": "quan 4",
    "q5": "quan 5",
    "q6": "quan 6",
    "q7": "quan 7",
    "q8": "quan 8",
    "q9": "quan 9",
    "q10": "quan 10",
    "q11": "quan 11",
    "q12": "quan 12",
    "p1": "phuong 1",
    "p2": "phuong 2",
    "p3": "phuong 3",
    "p4": "phuong 4",
    "p5": "phuong 5",
    "p6": "phuong 6",
    "p7": "phuong 7",
    "p8": "phuong 8",
    "p9": "phuong 9",
    "pn": "phong ngu",
    "wc": "phong tam",
    "vs": "ve sinh",
}


def normalize_text(text: str) -> str:
    if not text:
        return ""
    output = str(text).lower().replace("đ", "d")
    output = unicodedata.normalize("NFKD", output)
    output = "".join(ch for ch in output if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", output).strip()


def expand_abbreviations(text: str) -> str:
    normalized = normalize_text(text)
    if not normalized:
        return ""

    expanded = normalized
    # Replace longer keys first to avoid partial overwrite.
    for source in sorted(_ABBREVIATION_MAP, key=len, reverse=True):
        pattern = rf"\b{re.escape(source)}\b"
        expanded = re.sub(pattern, _ABBREVIATION_MAP[source], expanded)
    return re.sub(r"\s+", " ", expanded).strip()


def normalize_query_pipeline(text: str) -> str:
    expanded_text = expand_abbreviations(text)
    normalized_ascii_text = normalize_text(expanded_text)
    if not expanded_text:
        return ""
    if expanded_text == normalized_ascii_text:
        return expanded_text
    return f"{expanded_text} {normalized_ascii_text}".strip()
