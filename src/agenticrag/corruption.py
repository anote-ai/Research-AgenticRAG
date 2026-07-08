"""Certified content-level corruption of retrieved evidence.

The structural live interventions (empty retrieval, off-topic noise docs,
prepended false-premise strings) are detectable *by construction*: a
coverage-gated diagnoser sees zero support and localizes the hop trivially,
which inflates hop-1 attribution accuracy (AUDIT A6).  Content corruption is
the complementary — and more deployment-realistic — fault family: the
documents stay topically intact, but the answer-bearing fact (or the bridge
entity that carries the reasoning chain to the next hop) is flipped to a
plausible wrong value.  This models a stale index entry, a poisoned or
mis-OCR'd document, or an upstream data bug.

Every corruption is *certified*: we record exactly which span changed and what
it changed to (:class:`CorruptionRecord`), which enables a fully deterministic
generation-level evaluation with no LLM judge — the final answer is classified
by string matching against the known spans:

- ``absorbed``  — the corrupted value shows up in the final answer
  (the fault propagated verbatim into generation);
- ``resisted``  — the answer is still correct (the agent recovered);
- ``derailed``  — neither (the corruption knocked the agent into some other
  failure mode).

Span selection strategies, tried in order at the injected hop:

1. ``answer_fact``    — the gold answer appears in this hop's docs: corrupt it.
   The sharpest certified intervention at the answer-bearing hop.
2. ``bridge_entity``  — an entity that appears in this hop's docs *and* in a
   later hop's sub-query but *not* in the original question: the fact the
   agent actually carried forward.  Corrupting it breaks the chain link.
3. ``salient_entity`` — fallback: the most salient entity (or number) in this
   hop's docs.

All randomness is seeded per (seed, trace, hop) via CRC32, so corruption is
reproducible across processes without ``PYTHONHASHSEED`` pinning.
"""

from __future__ import annotations

import random
import re
import zlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .core import _answer_correct, _normalize_answer, _token_recall


# --------------------------------------------------------------------------- #
# Corruption record (the certified ground truth)                               #
# --------------------------------------------------------------------------- #

@dataclass
class CorruptionRecord:
    """Exactly what was changed by a content corruption — the certified label.

    ``original_span`` -> ``corrupted_span`` is the key fact this corruption
    flipped; downstream evaluation matches generated answers against these
    spans, so the record doubles as the deterministic evaluation key.
    """

    hop: int
    original_span: str
    corrupted_span: str
    strategy: str          # "answer_fact" | "bridge_entity" | "salient_entity"
    replacement_kind: str  # "numeric" | "entity"
    n_docs_corrupted: int
    n_replacements: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "hop": self.hop,
            "original_span": self.original_span,
            "corrupted_span": self.corrupted_span,
            "strategy": self.strategy,
            "replacement_kind": self.replacement_kind,
            "n_docs_corrupted": self.n_docs_corrupted,
            "n_replacements": self.n_replacements,
        }


# --------------------------------------------------------------------------- #
# Candidate-span extraction                                                    #
# --------------------------------------------------------------------------- #

# Lowercase connectors allowed *inside* a multi-token proper-noun span
# ("Duke of Wellington", "Rio de Janeiro").
_CONNECTORS = r"(?:of|the|de|la|le|van|von|da|di|del|der|al|bin|ibn)"
_CAP = r"[A-Z][A-Za-z0-9'’\-]*"
_ENTITY_RE = re.compile(
    rf"\b{_CAP}(?:\s+(?:{_CONNECTORS}\s+)?{_CAP})*\b"
)
_NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)*\b")

# Single capitalized tokens that are almost always sentence starters or
# function words, never corruptible facts.
_STOP_SINGLES = {
    "the", "a", "an", "in", "it", "he", "she", "they", "we", "i", "you",
    "this", "that", "these", "those", "there", "however", "according",
    "as", "on", "at", "for", "with", "after", "before", "by", "during",
    "its", "his", "her", "their", "when", "while", "although", "but",
    "and", "or", "if", "then", "also", "some", "many", "most", "one",
    "two", "both", "each", "since", "later", "today", "yes", "no",
}


def _stable_rng(seed: int, *parts: object) -> random.Random:
    """Deterministic RNG keyed on (seed, *parts) — stable across processes."""
    key = ":".join([str(seed), *[str(p) for p in parts]])
    return random.Random(zlib.crc32(key.encode("utf-8")))


def _span_in_text(span: str, text: str) -> bool:
    """Case-insensitive whole-word containment of *span* in *text*."""
    if not span.strip():
        return False
    pattern = re.compile(rf"(?<!\w){re.escape(span.strip())}(?!\w)", re.IGNORECASE)
    return bool(pattern.search(text))


def _candidate_spans(docs: Sequence[str]) -> Dict[str, int]:
    """Extract candidate fact spans (entities + numbers) with frequencies.

    Keys are surface forms as they appear in the docs; values are occurrence
    counts across all docs.  Multi-token entities are preferred downstream, so
    frequency alone does not decide the winner.
    """
    counts: Dict[str, int] = {}
    joined = "\n".join(docs)
    for m in _ENTITY_RE.finditer(joined):
        span = m.group(0).strip()
        toks = span.split()
        if len(toks) == 1 and toks[0].lower() in _STOP_SINGLES:
            continue
        counts[span] = counts.get(span, 0) + 1
    for m in _NUMBER_RE.finditer(joined):
        span = m.group(0)
        counts[span] = counts.get(span, 0) + 1
    return counts


def select_target_span(
    docs: Sequence[str],
    later_queries: Sequence[str] = (),
    question: str = "",
    reference_answer: str = "",
) -> Optional[Tuple[str, str]]:
    """Choose the fact span to corrupt in *docs* — ``(span, strategy)`` or None.

    Strategy order: gold answer present in the docs; else a bridge entity
    (in the docs and in a later sub-query but not in the original question);
    else the most salient entity/number.  Returns None when the docs contain
    no corruptible span at all (caller should skip the sample — an uncertified
    corruption is worse than none).
    """
    joined = "\n".join(docs)
    if not joined.strip():
        return None

    # 1. Answer fact.  Skip degenerate golds ("yes"/"no", 1–2 chars) whose
    #    corruption would be meaningless or unmatchable.
    gold = (reference_answer or "").strip()
    gold_n = _normalize_answer(gold)
    if len(gold_n) >= 3 and gold_n not in ("yes", "no") and _span_in_text(gold, joined):
        return gold, "answer_fact"

    candidates = _candidate_spans(docs)
    if not candidates:
        return None

    def _rank(item: Tuple[str, int]) -> Tuple[int, int, int, str]:
        span, freq = item
        # More tokens > more occurrences > longer string; span text breaks ties
        # deterministically.
        return (len(span.split()), freq, len(span), span)

    # 2. Bridge entity: carried into a later sub-query, but not simply copied
    #    from the original question.
    bridges = [
        (span, freq)
        for span, freq in candidates.items()
        if any(_token_recall(q, span) >= 1.0 for q in later_queries)
        and _token_recall(question, span) < 1.0
    ]
    if bridges:
        span, _ = max(bridges, key=_rank)
        return span, "bridge_entity"

    # 3. Salient entity fallback.
    span, _ = max(candidates.items(), key=_rank)
    return span, "salient_entity"


# --------------------------------------------------------------------------- #
# Replacement generation                                                        #
# --------------------------------------------------------------------------- #

# Fallback distractors when no in-domain pool is supplied.  Deliberately
# plausible-looking proper nouns with zero overlap with real QA corpora.
_DEFAULT_DISTRACTORS: List[str] = [
    "Meridian Vale",
    "Corvin Aldermann",
    "Ostrava Heights",
    "Talwyn Marsh",
    "the Halcyon Institute",
    "Sable Point",
    "Ilsa Verhoeven",
    "Branmore County",
]


def _is_numeric(span: str) -> bool:
    return bool(re.fullmatch(r"\d+(?:[.,]\d+)*", span.strip()))


def _perturb_number(span: str, rng: random.Random) -> str:
    """Deterministically perturb a numeric span into a nearby-but-wrong value."""
    s = span.strip()
    if re.fullmatch(r"[12]\d{3}", s):  # a year: shift by 2–5
        delta = rng.choice([-5, -4, -3, -2, 2, 3, 4, 5])
        return str(int(s) + delta)
    digits = s.replace(",", "")
    if re.fullmatch(r"\d+", digits):
        v = int(digits)
        if v < 10:
            return str(v + rng.choice([1, 2, 3]))
        factor = rng.choice([0.5, 0.7, 1.3, 1.5, 2.0])
        new_v = max(1, int(round(v * factor)))
        return str(new_v if new_v != v else v + 1)
    if re.fullmatch(r"\d+\.\d+", digits):
        v = float(digits)
        factor = rng.choice([0.5, 0.7, 1.3, 1.5, 2.0])
        decimals = len(digits.split(".")[1])
        return f"{v * factor:.{decimals}f}"
    # Unparseable (e.g. "1,2.3") — swap the first digit.
    first = s[0]
    swapped = str((int(first) + rng.choice([1, 2, 3])) % 10)
    return swapped + s[1:]


def make_replacement(
    span: str,
    rng: random.Random,
    distractor_pool: Optional[Sequence[str]] = None,
    avoid_texts: Sequence[str] = (),
) -> Tuple[str, str]:
    """Produce a certified-wrong replacement for *span* — ``(replacement, kind)``.

    Numeric spans get a deterministic perturbation.  Entity spans draw from
    ``distractor_pool`` (in practice: other samples' gold answers — natural,
    in-domain, wrong), filtered so the replacement shares no tokens with the
    original and does not already occur in ``avoid_texts`` (which would muddy
    the certified absorbed/resisted classification).  Falls back to a built-in
    distractor list when the pool filters to nothing.
    """
    if _is_numeric(span):
        return _perturb_number(span, rng), "numeric"

    joined_avoid = "\n".join(avoid_texts)

    def _usable(cand: str) -> bool:
        c = cand.strip()
        return (
            len(_normalize_answer(c)) >= 3
            and not _is_numeric(c)
            and _token_recall(c, span) == 0.0  # no token shared with original
            and _token_recall(span, c) == 0.0
            and not _span_in_text(c, joined_avoid)
        )

    pool = [c for c in (distractor_pool or []) if _usable(c)]
    if not pool:
        pool = [c for c in _DEFAULT_DISTRACTORS if _usable(c)]
    if not pool:  # pathological: original overlaps every fallback too
        pool = list(_DEFAULT_DISTRACTORS)
    # Sort before choice so the draw depends only on the seeded RNG, not on
    # pool ordering quirks upstream.
    return rng.choice(sorted(set(pool))), "entity"


# --------------------------------------------------------------------------- #
# Document rewriting                                                            #
# --------------------------------------------------------------------------- #

def corrupt_docs(
    docs: Sequence[str], span: str, replacement: str
) -> Tuple[List[str], int, int]:
    """Replace every whole-word occurrence of *span* in *docs* (case-insensitive).

    Returns ``(new_docs, n_docs_touched, n_replacements)``.  Documents are
    never mutated in place.
    """
    pattern = re.compile(rf"(?<!\w){re.escape(span.strip())}(?!\w)", re.IGNORECASE)
    new_docs: List[str] = []
    n_docs = 0
    n_repl = 0
    for d in docs:
        new_d, k = pattern.subn(replacement, d)
        if k:
            n_docs += 1
            n_repl += k
        new_docs.append(new_d)
    return new_docs, n_docs, n_repl


# --------------------------------------------------------------------------- #
# Deterministic generation-level evaluation                                     #
# --------------------------------------------------------------------------- #

def classify_absorption(
    final_answer: str, reference_answer: str, corrupted_span: str
) -> str:
    """Classify a generated answer against the certified corruption spans.

    ``resisted`` — the answer is still correct (checked first: correctness
    dominates).  ``absorbed`` — the corrupted value's tokens all appear in the
    answer (the fault propagated verbatim into generation).  ``derailed`` —
    neither: the corruption caused some other failure.
    """
    if _answer_correct(final_answer, reference_answer):
        return "resisted"
    span_n = _normalize_answer(corrupted_span or "")
    if span_n and _token_recall(final_answer, corrupted_span) >= 0.8:
        return "absorbed"
    return "derailed"
