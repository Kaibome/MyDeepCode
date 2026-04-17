"""Multilingual keyword extraction for food delivery reviews."""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable

import yake
from langdetect import DetectorFactory, LangDetectException, detect_langs

from .config import BUSINESS_KEYWORDS, KEYWORD_RULES, LANG_STOPWORDS, SUPPORTED_LANGS

DetectorFactory.seed = 0

DEFAULT_TOP_K = 5
MIN_TOP_K = 3
MAX_TOP_K = 5

URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
SPACE_RE = re.compile(r"\s+")


def extract_keywords(
    text: str,
    top_k: int = DEFAULT_TOP_K,
    *,
    enable_online_fallback: bool = False,
) -> list[str]:
    """Extract 3-5 business keywords from a multilingual review."""
    normalized_top_k = _normalize_top_k(top_k)
    cleaned = _preprocess_text(text)
    if not cleaned:
        return []

    lang, lang_score = _detect_language(cleaned)
    initial_keywords = _extract_offline(cleaned, lang=lang, top_k=normalized_top_k)

    if len(initial_keywords) >= MIN_TOP_K or not enable_online_fallback:
        return initial_keywords[:normalized_top_k]

    translated = _translate_for_fallback(cleaned, source_lang=lang)
    if not translated:
        return initial_keywords[:normalized_top_k]

    fallback_keywords = _extract_offline(translated, lang="en", top_k=normalized_top_k)
    merged = _merge_keywords(initial_keywords, fallback_keywords, normalized_top_k)
    if lang_score < 0.60:
        return merged
    return merged if len(merged) >= len(initial_keywords) else initial_keywords


def _normalize_top_k(top_k: int) -> int:
    return max(MIN_TOP_K, min(MAX_TOP_K, int(top_k)))


def _preprocess_text(text: str) -> str:
    if not text:
        return ""
    cleaned = URL_RE.sub(" ", text)
    cleaned = unicodedata.normalize("NFKC", cleaned)
    cleaned = "".join(ch for ch in cleaned if unicodedata.category(ch) != "So")
    cleaned = SPACE_RE.sub(" ", cleaned).strip()
    return cleaned


def _detect_language(text: str) -> tuple[str, float]:
    try:
        lang_scores = detect_langs(text)
    except LangDetectException:
        return "en", 0.0

    if not lang_scores:
        return "en", 0.0

    best = lang_scores[0]
    lang = best.lang if best.lang in SUPPORTED_LANGS else "en"
    return lang, float(best.prob)


def _extract_offline(text: str, lang: str, top_k: int) -> list[str]:
    candidates = _extract_candidates_with_yake(text, lang=lang, max_candidates=12)
    mapped_scores = _map_candidates_to_business_keywords(candidates, lang=lang)
    for keyword, score in _rule_based_scores(text).items():
        mapped_scores[keyword] = mapped_scores.get(keyword, 0.0) + score
    ranked = sorted(mapped_scores.items(), key=lambda pair: pair[1], reverse=True)
    return [keyword for keyword, _ in ranked][:top_k]


def _extract_candidates_with_yake(
    text: str, *, lang: str, max_candidates: int
) -> list[tuple[str, float]]:
    keyword_extractor = yake.KeywordExtractor(
        lan=lang if lang in (SUPPORTED_LANGS | {"en"}) else "en",
        n=2,
        dedupLim=0.85,
        top=max_candidates,
    )
    return keyword_extractor.extract_keywords(text)


def _map_candidates_to_business_keywords(
    candidates: Iterable[tuple[str, float]], *, lang: str
) -> dict[str, float]:
    stopwords = LANG_STOPWORDS.get(lang, set()) | LANG_STOPWORDS.get("en", set())
    normalized_scores = {keyword: 0.0 for keyword in BUSINESS_KEYWORDS}

    for phrase, yake_score in candidates:
        phrase_norm = _fold_text(phrase)
        if not phrase_norm or phrase_norm in stopwords:
            continue
        match = _keyword_match_score(phrase_norm)
        if not match:
            continue
        keyword, rule_score = match
        extraction_score = 1.0 / (1.0 + max(yake_score, 0.0))
        normalized_scores[keyword] += (0.65 * rule_score) + (0.35 * extraction_score)

    return {k: v for k, v in normalized_scores.items() if v > 0}


def _keyword_match_score(phrase_norm: str) -> tuple[str, float] | None:
    best_keyword = ""
    best_score = 0.0
    for business_keyword, synonyms in KEYWORD_RULES.items():
        for synonym in synonyms:
            synonym_norm = _fold_text(synonym)
            if phrase_norm == synonym_norm:
                score = 1.0
            elif synonym_norm in phrase_norm:
                score = max(0.6, len(synonym_norm) / max(len(phrase_norm), 1))
            elif phrase_norm in synonym_norm:
                score = max(0.5, len(phrase_norm) / max(len(synonym_norm), 1))
            else:
                score = 0.0

            if score > best_score:
                best_keyword = business_keyword
                best_score = score

    if best_keyword:
        return best_keyword, best_score
    return None


def _rule_based_scores(text: str) -> dict[str, float]:
    folded = _fold_text(text)
    scores: dict[str, float] = {}
    for business_keyword, synonyms in KEYWORD_RULES.items():
        hit_count = 0
        for synonym in synonyms:
            synonym_norm = _fold_text(synonym)
            if synonym_norm and synonym_norm in folded:
                hit_count += 1
        if hit_count:
            scores[business_keyword] = 0.2 * hit_count
    return scores


def _fold_text(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode(
        "ascii"
    ).lower() or text.lower()


def _translate_for_fallback(text: str, source_lang: str) -> str:
    try:
        from deep_translator import GoogleTranslator
    except Exception:
        return ""

    try:
        translated = GoogleTranslator(source=source_lang, target="en").translate(text)
        return translated or ""
    except Exception:
        return ""


def _merge_keywords(primary: list[str], secondary: list[str], top_k: int) -> list[str]:
    merged: list[str] = []
    for item in primary + secondary:
        if item not in merged:
            merged.append(item)
        if len(merged) >= top_k:
            break
    return merged

