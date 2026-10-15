"""Evaluation metrics for fast tokenizers.

All metrics operate on a list of raw UTF-8 text samples and are returned as
plain JSON-serialisable dicts.

Metric definitions (also documented in README.md)
-------------------------------------------------
base layer
    avg_tokens              mean number of tokens produced per sample
    fertility               tokens / word  (1.0 is ideal, >= 4.0 is considered bad)
    compression_ratio       tokens / character
    byte_compression        tokens / UTF-8 byte
    fidelity_pass_rate      share of samples where decode(encode(text)) == text
    samples_evaluated       number of non-empty samples that were evaluated
lexical layer
    avg_token_bytes         mean length (UTF-8 bytes) of the emitted token strings
    avg_token_chars         mean length (unicode codepoints) of the emitted tokens
    type_token_ratio        unique tokens / total tokens
    vocabulary_utilization  unique tokens / vocab size of the tokenizer
    unique_tokens           number of distinct token ids observed
    subword_fertility       continuation sub-tokens / word
    continued_word_rate     share of words that were split into > 1 token
advanced layer
    shannon_entropy         Shannon entropy (bits) of the token distribution
    renyi_entropy_alpha_*   Renyi entropy for alpha = 0.5 / 1 / 2 (bits)
    fairness_gini           Gini coefficient of per-language token cost
                            (cost = tokens / raw UTF-8 bytes, languages equally weighted)
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, List, Sequence, Tuple

# --------------------------------------------------------------------------
# word segmentation
# --------------------------------------------------------------------------

#: Scripts that do not use whitespace to separate words. Every character of
#: these scripts is counted as one "word" so that fertility stays comparable
#: across languages (this is the same convention used by the reference
#: tokenizer benchmarks for CJK text).
_CJK_WORD_RANGES: Tuple[Tuple[int, int], ...] = (
    (0x1100, 0x11FF),  # Hangul Jamo
    (0x3040, 0x30FF),  # Hiragana + Katakana
    (0x3400, 0x4DBF),  # CJK ext A
    (0x4E00, 0x9FFF),  # CJK unified ideographs
    (0xAC00, 0xD7AF),  # Hangul syllables
    (0xF900, 0xFAFF),  # CJK compatibility ideographs
    (0x20000, 0x2FA1F),  # CJK ext B-F
)

#: Unicode-aware word character test: ``\w`` matches letters / digits /
#: underscore in *every* script (Latin, Cyrillic, Arabic, Devanagari, ...),
#: which keeps ``fertility`` meaningful for non-Latin languages.
_HAS_WORD_CHAR = re.compile(r"\w", re.UNICODE)


def is_cjk_word_char(ch: str) -> bool:
    cp = ord(ch)
    for lo, hi in _CJK_WORD_RANGES:
        if lo <= cp <= hi:
            return True
    return False


def word_spans(text: str) -> List[Tuple[int, int]]:
    """Return ``(start, end)`` character spans of every "word" in ``text``.

    - CJK / Hangul / Kana characters are one word each.
    - Everything else is split on whitespace.
    - Spans that contain no word character at all (pure punctuation) are dropped.
    """
    spans: List[Tuple[int, int]] = []
    buf_start = None
    n = len(text)
    i = 0
    while i < n:
        ch = text[i]
        if ch.isspace():
            if buf_start is not None:
                spans.append((buf_start, i))
                buf_start = None
            i += 1
            continue
        if is_cjk_word_char(ch):
            if buf_start is not None:
                spans.append((buf_start, i))
                buf_start = None
            spans.append((i, i + 1))
            i += 1
            continue
        if buf_start is None:
            buf_start = i
        i += 1
    if buf_start is not None:
        spans.append((buf_start, n))

    out: List[Tuple[int, int]] = []
    for start, end in spans:
        frag = text[start:end]
        if any(is_cjk_word_char(c) for c in frag) or _HAS_WORD_CHAR.search(frag):
            out.append((start, end))
    return out


# --------------------------------------------------------------------------
# entropy helpers
# --------------------------------------------------------------------------

def shannon_entropy(counts: Sequence[int]) -> float:
    """Shannon entropy in bits of a count distribution."""
    total = float(sum(counts))
    if total <= 0:
        return 0.0
    acc = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = c / total
        acc -= p * math.log2(p)
    return acc


def renyi_entropy(counts: Sequence[int], alpha: float) -> float:
    """Renyi entropy of order ``alpha`` in bits (alpha == 1 -> Shannon)."""
    if abs(alpha - 1.0) < 1e-9:
        return shannon_entropy(counts)
    total = float(sum(counts))
    if total <= 0:
        return 0.0
    acc = 0.0
    for c in counts:
        if c <= 0:
            continue
        acc += (c / total) ** alpha
    if acc <= 0:
        return 0.0
    return math.log2(acc) / (1.0 - alpha)


def gini_coefficient(values: Sequence[float]) -> float:
    """Gini coefficient of a list of non-negative values (0 = perfectly even)."""
    vals = sorted(float(v) for v in values)
    n = len(vals)
    if n == 0:
        return 0.0
    total = sum(vals)
    if total <= 0:
        return 0.0
    weighted = sum((i + 1) * v for i, v in enumerate(vals))
    return (2.0 * weighted) / (n * total) - (n + 1.0) / n


def normalised_entropy(counts: Sequence[int]) -> float:
    """Shannon entropy divided by log2(vocab) -> 0..1."""
    k = sum(1 for c in counts if c > 0)
    if k <= 1:
        return 0.0
    h = shannon_entropy(counts)
    return h / math.log2(k)


# --------------------------------------------------------------------------
# subword marker helpers
# --------------------------------------------------------------------------

_CONTINUATION_MARKERS = ("##", "\u2581", "\u0120", "\u010a")  # ##, ▁, Ġ, Ċ


def is_marked_continuation(token_str: str) -> bool:
    return token_str.startswith(_CONTINUATION_MARKERS)


def strip_markers(token_str: str) -> str:
    for marker in _CONTINUATION_MARKERS:
        if token_str.startswith(marker):
            return token_str[len(marker):]
    return token_str


# --------------------------------------------------------------------------
# main evaluation entry points
# --------------------------------------------------------------------------

def _safe_ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def evaluate_samples(handle: Any, samples: Sequence[str]) -> Dict[str, Any]:
    """Evaluate a single tokenizer over ``samples`` (list of raw texts)."""
    vocab_size = int(handle.vocab_size)

    total_tokens = 0
    total_words = 0
    total_chars = 0
    total_bytes = 0
    fidelity_pass = 0
    samples_evaluated = 0
    failed_samples: List[Dict[str, Any]] = []

    token_counter: Counter = Counter()
    token_char_len_sum = 0
    token_byte_len_sum = 0
    continuation_tokens = 0
    continued_words = 0

    for text in samples:
        if text is None or not text.strip():
            continue
        samples_evaluated += 1
        total_chars += len(text)
        total_bytes += len(text.encode("utf-8"))

        enc = handle.encode_with_offsets(text, add_special_tokens=False)
        ids: List[int] = enc["ids"]
        offsets: List[Tuple[int, int]] = enc["offsets"]
        token_strings: List[str] = enc["tokens"]

        total_tokens += len(ids)
        token_counter.update(ids)

        for token_str in token_strings:
            token_char_len_sum += len(strip_markers(token_str))
            token_byte_len_sum += len(strip_markers(token_str).encode("utf-8"))

        spans = word_spans(text)
        total_words += len(spans)

        covered: Dict[int, int] = {}
        for (ts, te), token_str in zip(offsets, token_strings):
            if te <= ts:
                # zero-width token (special token / added token)
                if is_marked_continuation(token_str):
                    continuation_tokens += 1
                continue
            for wi, (ws, we) in enumerate(spans):
                if ws <= ts < we:
                    covered[wi] = covered.get(wi, 0) + 1
                    if ts != ws or is_marked_continuation(token_str):
                        continuation_tokens += 1
                    break
        continued_words += sum(1 for c in covered.values() if c > 1)

        decoded = handle.decode(ids)
        ok = _fidelity_ok(text, decoded, handle, ids)
        if ok:
            fidelity_pass += 1
        elif len(failed_samples) < 20:
            failed_samples.append({"text": text[:200], "decoded": decoded[:200]})

    counts = list(token_counter.values())
    unique_tokens = len(counts)

    metrics: Dict[str, Any] = {
        "samples_evaluated": samples_evaluated,
        "avg_tokens": _safe_ratio(total_tokens, samples_evaluated),
        "fertility": _safe_ratio(total_tokens, total_words),
        "compression_ratio": _safe_ratio(total_tokens, total_chars),
        "byte_compression": _safe_ratio(total_tokens, total_bytes),
        "fidelity_pass_rate": _safe_ratio(fidelity_pass, samples_evaluated),
        "fidelity_fails": samples_evaluated - fidelity_pass,
        "avg_token_bytes": _safe_ratio(token_byte_len_sum, total_tokens),
        "avg_token_chars": _safe_ratio(token_char_len_sum, total_tokens),
        "type_token_ratio": _safe_ratio(unique_tokens, total_tokens),
        "vocabulary_utilization": _safe_ratio(unique_tokens, vocab_size),
        "unique_tokens": unique_tokens,
        "vocab_size": vocab_size,
        "subword_fertility": _safe_ratio(continuation_tokens, total_words),
        "continued_word_rate": _safe_ratio(continued_words, total_words),
        "shannon_entropy": shannon_entropy(counts),
        "renyi_entropy_alpha_0_5": renyi_entropy(counts, 0.5),
        "renyi_entropy_alpha_1": shannon_entropy(counts),
        "renyi_entropy_alpha_2": renyi_entropy(counts, 2.0),
        "total_tokens": total_tokens,
        "total_words": total_words,
        "total_chars": total_chars,
        "total_bytes": total_bytes,
    }
    metrics["fertility_grade"] = grade_fertility(metrics["fertility"])
    metrics["failed_samples"] = failed_samples
    return metrics


def _fidelity_ok(text: str, decoded: str, handle: Any, ids: Sequence[int]) -> bool:
    """encode -> decode round-trip check.

    Exact match is required after stripping the leading / trailing whitespace
    that some (WordPiece style) tokenizers legitimately drop, and after trying
    the raw backend decoder which never applies ``clean_up_tokenization_spaces``.
    """
    if decoded == text:
        return True
    if decoded.strip() == text.strip():
        return True
    try:
        raw = handle.raw_decode(ids)
    except Exception:
        raw = None
    if raw is not None and (raw == text or raw.strip() == text.strip()):
        return True
    return False


def grade_fertility(fertility: float) -> str:
    if fertility <= 0:
        return "n/a"
    if fertility < 1.5:
        return "excellent"
    if fertility < 2.0:
        return "good"
    if fertility < 3.0:
        return "fair"
    if fertility < 4.0:
        return "poor"
    return "bad"


# --------------------------------------------------------------------------
# language fairness
# --------------------------------------------------------------------------

def fairness_gini(per_language_cost: Dict[str, float]) -> float:
    """Gini coefficient over per-language token costs (languages equally weighted)."""
    if not per_language_cost:
        return 0.0
    return gini_coefficient(list(per_language_cost.values()))


DEFAULT_WEIGHTS: Dict[str, float] = {
    "fertility": 0.35,
    "byte_compression": 0.2,
    "fidelity_pass_rate": 0.25,
    "vocabulary_utilization": 0.1,
    "fairness": 0.1,
}


def composite_score(summary: Dict[str, Any]) -> float:
    """Weighted 0-100 score used for the compare leaderboard (higher is better)."""
    def norm_fertility(f: float) -> float:
        if f <= 0:
            return 0.0
        # 1.0 -> 1.0, 4.0 -> 0.0
        return max(0.0, min(1.0, (4.0 - f) / 3.0))

    def norm_byte(b: float) -> float:
        if b <= 0:
            return 0.0
        # lower is better; 0.10 tokens/byte -> 1.0, 0.40 -> 0.0
        return max(0.0, min(1.0, (0.40 - b) / 0.30))

    parts = {
        "fertility": norm_fertility(float(summary.get("fertility", 0) or 0)),
        "byte_compression": norm_byte(float(summary.get("byte_compression", 0) or 0)),
        "fidelity_pass_rate": max(0.0, min(1.0, float(summary.get("fidelity_pass_rate", 0) or 0))),
        "vocabulary_utilization": max(0.0, min(1.0, float(summary.get("vocabulary_utilization", 0) or 0))),
        "fairness": 1.0 - max(0.0, min(1.0, float(summary.get("fairness_gini", 0) or 0))),
    }
    return round(100.0 * sum(parts[k] * DEFAULT_WEIGHTS[k] for k in parts), 2)
