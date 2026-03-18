from __future__ import annotations

import re

from .config import env


def contains_arabic(text: str) -> bool:
    # Arabic blocks: \u0600-\u06FF, \u0750-\u077F, \u08A0-\u08FF
    return bool(re.search(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]", text or ""))


def maybe_translate_ar_to_en(text: str) -> str | None:
    enabled = (env("ARGOS_ENABLED", "false") or "false").lower() in {"1", "true", "yes", "y"}
    if not enabled:
        return None
    try:
        from argostranslate import translate as argo_translate
    except Exception:
        return None
    try:
        return argo_translate.translate(text, "ar", "en")
    except Exception:
        return None

