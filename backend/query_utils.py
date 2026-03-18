from __future__ import annotations

import re

from .search_index import tokenize


def normalize_query(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def is_out_of_scope(query: str) -> tuple[bool, str]:
    q = normalize_query(query)
    q_l = q.lower()

    if len(q_l) < 2:
        return True, "Query too short."

    en_patterns = [
        r"^(hi|hello|hey|yo)\b",
        r"^(good\s+(morning|afternoon|evening|night))\b",
        r"^how\s+are\s+you\b",
        r"^what'?s\s+up\b",
        r"^thanks\b|^thank\s+you\b",
    ]

    ar_patterns = [
        r"^(مرحبا|أهلا|اهلا|هلا)\b",
        r"^(السلام\s+عليكم|سلام)\b",
        r"^(صباح\s+الخير|مساء\s+الخير)\b",
        r"^(كيف\s+حال(ك|كم)|شلونك)\b",
        r"^(شكرا|شكرًا|يسلمو)\b",
    ]

    for pat in en_patterns:
        if re.search(pat, q_l):
            return True, "Casual greeting/small-talk (English)."
    for pat in ar_patterns:
        if re.search(pat, q):
            return True, "Casual greeting/small-talk (Arabic)."

    if len(tokenize(q)) <= 1 and len(q) <= 4:
        return True, "Query too generic/low-signal."

    return False, ""


def build_query_variants(original: str, translated: str | None, rephrase: bool) -> list[str]:
    variants: list[str] = []
    o = normalize_query(original)
    if o:
        variants.append(o)

    if translated:
        t = normalize_query(translated)
        if t and t not in variants:
            variants.append(t)

    if rephrase:
        low = o.lower()
        if low and low not in variants:
            variants.append(low)

        if translated:
            toks = [tok for tok in tokenize(translated) if len(tok) >= 3]
            kw = " ".join(toks)
            if kw and kw not in variants:
                variants.append(kw)

    return variants[:6]

