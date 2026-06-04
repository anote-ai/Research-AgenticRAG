from __future__ import annotations

"""Dataset adapters for real multi-hop QA benchmarks.

HotpotQA (distractor setting) and MuSiQue are the primary targets.
Both are loaded via the HuggingFace `datasets` library when available;
otherwise a small built-in fallback is used so the package stays importable
without the optional dependency.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional


@dataclass
class QASample:
    """Normalised representation of a multi-hop QA example."""

    question: str
    answer: str
    supporting_docs: List[str] = field(default_factory=list)
    hop_count: int = 1
    dataset: str = ""
    sample_id: str = ""


def _hf_available() -> bool:
    try:
        import datasets  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# HotpotQA adapter
# ---------------------------------------------------------------------------

def load_hotpotqa(
    split: str = "validation",
    max_samples: Optional[int] = None,
    setting: str = "distractor",
) -> List[QASample]:
    """Load HotpotQA samples.

    Parameters
    ----------
    split:
        "train", "validation", or "test" (test has no gold answers).
    max_samples:
        Cap the number of returned samples. None = no cap.
    setting:
        HotpotQA config name passed to `datasets.load_dataset`.

    Returns a list of QASample; falls back to built-in examples if the
    `datasets` package is not installed.
    """
    if not _hf_available():
        return _hotpotqa_fallback()

    import datasets as hf_datasets  # type: ignore

    ds = hf_datasets.load_dataset("hotpot_qa", setting, split=split, trust_remote_code=True)
    samples: List[QASample] = []
    for i, row in enumerate(ds):
        if max_samples is not None and i >= max_samples:
            break
        supporting = _hotpotqa_supporting_docs(row)
        samples.append(
            QASample(
                question=row["question"],
                answer=row.get("answer", ""),
                supporting_docs=supporting,
                hop_count=2,
                dataset="hotpotqa",
                sample_id=str(row.get("id", i)),
            )
        )
    return samples


def _hotpotqa_supporting_docs(row: Dict[str, Any]) -> List[str]:
    """Flatten HotpotQA context into a list of document strings."""
    docs: List[str] = []
    context = row.get("context", {})
    titles = context.get("title", [])
    sentences_list = context.get("sentences", [])
    for title, sentences in zip(titles, sentences_list):
        text = " ".join(sentences)
        docs.append(f"{title}: {text}")
    return docs


def _hotpotqa_fallback() -> List[QASample]:
    """Tiny built-in HotpotQA-style examples when `datasets` is unavailable."""
    return [
        QASample(
            question="Who is the CEO of the company that produces the iPhone?",
            answer="Tim Cook",
            supporting_docs=[
                "Apple Inc. produces the iPhone.",
                "Tim Cook is the CEO of Apple Inc.",
            ],
            hop_count=2,
            dataset="hotpotqa_fallback",
            sample_id="fallback-0",
        ),
        QASample(
            question="In what country is the headquarters of the company that makes Python?",
            answer="United States",
            supporting_docs=[
                "Python is maintained by the Python Software Foundation.",
                "The Python Software Foundation is headquartered in the United States.",
            ],
            hop_count=2,
            dataset="hotpotqa_fallback",
            sample_id="fallback-1",
        ),
    ]


# ---------------------------------------------------------------------------
# MuSiQue adapter
# ---------------------------------------------------------------------------

def load_musique(
    split: str = "validation",
    max_samples: Optional[int] = None,
) -> List[QASample]:
    """Load MuSiQue samples.

    Falls back to built-in examples if `datasets` is unavailable.
    """
    if not _hf_available():
        return _musique_fallback()

    import datasets as hf_datasets  # type: ignore

    ds = hf_datasets.load_dataset("microsoft/musique", split=split, trust_remote_code=True)
    samples: List[QASample] = []
    for i, row in enumerate(ds):
        if max_samples is not None and i >= max_samples:
            break
        paragraphs = row.get("paragraphs", [])
        docs = [f"{p.get('title', '')}: {p.get('paragraph_text', '')}" for p in paragraphs]
        decomposition = row.get("question_decomposition", [])
        hop_count = len(decomposition) if decomposition else 2
        samples.append(
            QASample(
                question=row["question"],
                answer=row.get("answer", ""),
                supporting_docs=docs,
                hop_count=hop_count,
                dataset="musique",
                sample_id=str(row.get("id", i)),
            )
        )
    return samples


def _musique_fallback() -> List[QASample]:
    return [
        QASample(
            question="What is the nationality of the person who invented the telephone?",
            answer="Scottish",
            supporting_docs=[
                "Alexander Graham Bell invented the telephone.",
                "Alexander Graham Bell was born in Edinburgh, Scotland.",
            ],
            hop_count=2,
            dataset="musique_fallback",
            sample_id="fallback-0",
        ),
    ]


# ---------------------------------------------------------------------------
# Unified loader
# ---------------------------------------------------------------------------

def load_dataset(
    name: str,
    split: str = "validation",
    max_samples: Optional[int] = None,
) -> List[QASample]:
    """Load a named dataset by string key.

    Parameters
    ----------
    name:
        "hotpotqa" or "musique".
    split:
        Dataset split to load.
    max_samples:
        Optional cap.
    """
    name = name.lower()
    if name == "hotpotqa":
        return load_hotpotqa(split=split, max_samples=max_samples)
    if name == "musique":
        return load_musique(split=split, max_samples=max_samples)
    raise ValueError(f"Unknown dataset '{name}'. Choose 'hotpotqa' or 'musique'.")


def iter_batches(samples: List[QASample], batch_size: int = 32) -> Iterator[List[QASample]]:
    """Yield successive batches from a sample list."""
    for i in range(0, len(samples), batch_size):
        yield samples[i : i + batch_size]
