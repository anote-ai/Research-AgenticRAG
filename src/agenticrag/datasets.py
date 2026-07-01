from __future__ import annotations

"""Dataset adapters for real multi-hop / retrieval QA benchmarks.

Anchor sets (comparability with Doctor-RAG / AgenticRAGTracer): HotpotQA
(distractor) and MuSiQue. Richness sets (the rigor differentiator): FRAMES
(variable 2–15-hop — the depth substrate for the identifiability curve) and
CRAG (multi-domain; false-premise / long-tail / temporal question types — the
natural triggers for the live false-premise / stale-evidence injections).

All are loaded via the HuggingFace `datasets` library when available; otherwise
a small built-in fallback is used so the package stays importable and tests run
without the optional dependency or network/HF credentials.
"""

import ast
import json
import os
import re
import urllib.parse
import urllib.request
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

    ds = hf_datasets.load_dataset("hotpotqa/hotpot_qa", setting, split=split)
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

    ds = hf_datasets.load_dataset("dgslibisey/MuSiQue", split=split)
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
# FRAMES adapter (variable 2–15 hop — depth substrate for the C2 curve)
# ---------------------------------------------------------------------------

def load_frames(
    split: str = "test",
    max_samples: Optional[int] = None,
    repo_id: str = "google/frames-benchmark",
    fetch_passages: bool = False,
    passage_chars: int = 600,
    max_links_per_q: Optional[int] = None,
    max_passages_per_q: Optional[int] = None,
    cache_dir: str = ".agenticrag_cache",
    timeout: float = 20.0,
) -> List[QASample]:
    """Load FRAMES samples (variable 2–15-hop factual QA over Wikipedia).

    FRAMES supplies the variable hop depth needed for the propagation-depth
    identifiability curve. The benchmark gives Wikipedia *links* rather than gold
    passages, so by default ``supporting_docs`` carries the link list (a
    lightweight stand-in). Set ``fetch_passages=True`` to build a real retrieval
    corpus: each linked page's plain text is fetched from the MediaWiki API,
    chunked into ~``passage_chars`` passages, and used as ``supporting_docs`` —
    giving the agent something to actually retrieve over. Pages are cached on disk
    under ``cache_dir`` so repeated runs are network-free and reproducible.

    Falls back to built-in examples spanning 2–4 hops when `datasets` / HF access
    is unavailable.

    Parameters
    ----------
    fetch_passages:
        Build a Wikipedia passage corpus instead of returning bare links.
    passage_chars:
        Target characters per passage chunk.
    max_links_per_q / max_passages_per_q:
        Caps to bound corpus size (and fetch cost) per question.
    cache_dir:
        Directory for the persistent page-text cache (created if absent).
    timeout:
        Per-request fetch timeout (seconds).
    """
    if not _hf_available():
        return _frames_fallback()

    import datasets as hf_datasets  # type: ignore

    try:
        ds = hf_datasets.load_dataset(repo_id, split=split)
    except Exception:
        return _frames_fallback()

    cache = _WikiPageCache(cache_dir, timeout=timeout) if fetch_passages else None

    samples: List[QASample] = []
    for i, row in enumerate(ds):
        if max_samples is not None and i >= max_samples:
            break
        question = row.get("Prompt") or row.get("question") or row.get("prompt") or ""
        answer = row.get("Answer") or row.get("answer") or ""
        links = _normalize_links(row.get("wiki_links") or row.get("wikipedia_links"))
        hop_count = _frames_hop_count(row, links)

        if fetch_passages and cache is not None:
            docs = _build_frames_corpus(
                links, cache, passage_chars=passage_chars,
                max_links=max_links_per_q, max_passages=max_passages_per_q,
            )
        else:
            docs = [str(x) for x in links]

        samples.append(
            QASample(
                question=question,
                answer=str(answer),
                supporting_docs=docs,
                hop_count=hop_count,
                dataset="frames",
                sample_id=str(row.get("id", i)),
            )
        )

    if cache is not None:
        cache.flush()
    return samples


# -- FRAMES Wikipedia passage corpus -------------------------------------- #

_WIKI_API = "https://en.wikipedia.org/w/api.php"


def _normalize_links(raw: Any) -> List[str]:
    """Coerce a FRAMES ``wiki_links`` field to a list of URL strings.

    Handles native lists, stringified Python lists, and newline/comma-separated
    strings.
    """
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if isinstance(raw, str):
        s = raw.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                val = ast.literal_eval(s)
                if isinstance(val, list):
                    return [str(x) for x in val]
            except (ValueError, SyntaxError):
                pass
        return [u.strip() for u in re.split(r"[\n,]+", s) if "http" in u]
    return []


def _wiki_title_from_url(url: str) -> str:
    """Extract a page title from a Wikipedia URL (``.../wiki/Albert_Einstein``)."""
    path = urllib.parse.urlparse(url).path
    raw = path.split("/wiki/", 1)[1] if "/wiki/" in path else path.rsplit("/", 1)[-1]
    return urllib.parse.unquote(raw).replace("_", " ").strip()


def _chunk_text(text: str, title: str, max_chars: int) -> List[str]:
    """Pack a page's plain text into ~``max_chars`` passages, each title-prefixed."""
    paras = [
        p.strip()
        for p in text.split("\n")
        if p.strip() and not p.strip().startswith("==")
    ]
    chunks: List[str] = []
    buf = ""
    for p in paras:
        if len(buf) + len(p) + 1 <= max_chars:
            buf = (buf + " " + p).strip()
            continue
        if buf:
            chunks.append(f"{title}: {buf}")
            buf = ""
        while len(p) > max_chars:
            chunks.append(f"{title}: {p[:max_chars]}")
            p = p[max_chars:]
        buf = p
    if buf:
        chunks.append(f"{title}: {buf}")
    return chunks


class _WikiPageCache:
    """On-disk cache of Wikipedia page plain-text extracts, keyed by title."""

    def __init__(self, cache_dir: str, timeout: float = 20.0) -> None:
        self.cache_dir = cache_dir
        self.timeout = timeout
        self.path = os.path.join(cache_dir, "frames_wiki.json")
        self._store: Dict[str, str] = {}
        self._dirty = False
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    self._store = json.load(f)
            except (OSError, json.JSONDecodeError):
                self._store = {}

    def extract(self, title: str) -> str:
        if title in self._store:
            return self._store[title]
        text = _fetch_wikipedia_extract(title, timeout=self.timeout)
        self._store[title] = text
        self._dirty = True
        return text

    def flush(self) -> None:
        if not self._dirty:
            return
        os.makedirs(self.cache_dir, exist_ok=True)
        try:
            with open(self.path, "w") as f:
                json.dump(self._store, f)
            self._dirty = False
        except OSError:
            pass


_WIKI_USER_AGENT = (
    "agenticrag-research/0.1 (https://github.com/anote-ai/Research-AgenticRAG; "
    "failure-propagation benchmark)"
)


def _ssl_context() -> Any:
    """SSL context using certifi's CA bundle when available.

    Fixes the macOS framework-Python 'CERTIFICATE_VERIFY_FAILED' issue (no system
    cert store) without requiring the user to run Install Certificates.command.
    """
    import ssl

    try:
        import certifi  # type: ignore

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _fetch_wikipedia_extract(title: str, timeout: float = 20.0) -> str:
    """Fetch a page's plain-text extract via the MediaWiki API ('' on failure)."""
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": "1",
        "redirects": "1",
        "format": "json",
        "titles": title,
    }
    url = _WIKI_API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": _WIKI_USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as resp:
            data = json.load(resp)
    except Exception:
        return ""
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        extract = page.get("extract")
        if extract:
            return extract
    return ""


def _build_frames_corpus(
    links: List[str],
    cache: _WikiPageCache,
    passage_chars: int = 600,
    max_links: Optional[int] = None,
    max_passages: Optional[int] = None,
) -> List[str]:
    """Build a chunked passage corpus from a question's Wikipedia links."""
    docs: List[str] = []
    for url in links[:max_links] if max_links else links:
        title = _wiki_title_from_url(url)
        if not title:
            continue
        extract = cache.extract(title)
        if extract:
            docs.extend(_chunk_text(extract, title, passage_chars))
        else:
            docs.append(title)  # degrade to the bare title if the page is missing
    if max_passages is not None:
        docs = docs[:max_passages]
    return docs


def _frames_hop_count(row: Dict[str, Any], links: List[Any]) -> int:
    for key in ("num_hops", "hop_count", "n_hops"):
        val = row.get(key)
        if val is not None:
            try:
                return int(val)
            except (TypeError, ValueError):
                pass
    return max(2, len(links)) if links else 2


def _frames_fallback() -> List[QASample]:
    """Built-in FRAMES-style examples spanning a 2–4-hop range."""
    return [
        QASample(
            question="What is the capital of the country whose national animal is the Bengal tiger?",
            answer="New Delhi",
            supporting_docs=[
                "The Bengal tiger is the national animal of India.",
                "The capital of India is New Delhi.",
            ],
            hop_count=2,
            dataset="frames_fallback",
            sample_id="fallback-0",
        ),
        QASample(
            question=(
                "How many years after the company that makes the iPhone was founded "
                "did its current CEO take office?"
            ),
            answer="34",
            supporting_docs=[
                "Apple Inc. produces the iPhone.",
                "Apple Inc. was founded in 1976.",
                "Tim Cook became CEO of Apple in 2010.",
            ],
            hop_count=3,
            dataset="frames_fallback",
            sample_id="fallback-1",
        ),
        QASample(
            question=(
                "What is the elevation of the capital of the country that hosted the "
                "Olympic Games in the year the Eiffel Tower was completed?"
            ),
            answer="35 metres",
            supporting_docs=[
                "The Eiffel Tower was completed in 1889.",
                "The 1900 Olympic Games were held in Paris, France.",
                "The capital of France is Paris.",
                "Paris has an elevation of about 35 metres.",
            ],
            hop_count=4,
            dataset="frames_fallback",
            sample_id="fallback-2",
        ),
    ]


# ---------------------------------------------------------------------------
# CRAG adapter (multi-domain; false-premise / long-tail / temporal)
# ---------------------------------------------------------------------------

def load_crag(
    split: str = "train",
    max_samples: Optional[int] = None,
    repo_id: str = "Tevatron/crag",
) -> List[QASample]:
    """Load CRAG samples (5 domains, 8 question types incl. false-premise / temporal).

    CRAG's false-premise / long-tail / temporal questions are the natural
    triggers for the live false-premise and stale-evidence interventions. Each
    sample's ``supporting_docs`` is drawn from the provided search-result
    snippets when present; ``hop_count`` defaults to 1 (CRAG is single-turn but
    failure-mode-diverse). Domain and question type are preserved in
    ``sample_id`` for slicing. Falls back to built-in examples covering several
    question types when `datasets` / HF access is unavailable.
    """
    if not _hf_available():
        return _crag_fallback()

    import datasets as hf_datasets  # type: ignore

    try:
        ds = hf_datasets.load_dataset(repo_id, split=split)
    except Exception:
        return _crag_fallback()

    samples: List[QASample] = []
    for i, row in enumerate(ds):
        if max_samples is not None and i >= max_samples:
            break
        question = row.get("query") or row.get("question") or ""
        answer = row.get("answer") or row.get("answers") or ""
        if isinstance(answer, list):
            answer = answer[0] if answer else ""
        docs = _crag_search_docs(row)
        domain = row.get("domain", "")
        qtype = row.get("question_type") or row.get("static_or_dynamic") or ""
        samples.append(
            QASample(
                question=question,
                answer=str(answer),
                supporting_docs=docs,
                hop_count=1,
                dataset="crag",
                sample_id=f"{domain}:{qtype}:{row.get('interaction_id', i)}",
            )
        )
    return samples


def _crag_search_docs(row: Dict[str, Any]) -> List[str]:
    """Extract snippet/page text from CRAG search results, defensively."""
    results = row.get("search_results") or row.get("search_result") or []
    docs: List[str] = []
    if isinstance(results, list):
        for r in results:
            if isinstance(r, dict):
                text = (
                    r.get("page_snippet")
                    or r.get("page_result")
                    or r.get("snippet")
                    or r.get("text")
                    or ""
                )
                if text:
                    docs.append(str(text))
            elif isinstance(r, str):
                docs.append(r)
    return docs


def _crag_fallback() -> List[QASample]:
    """Built-in CRAG-style examples across question types (incl. false-premise)."""
    return [
        QASample(
            question="What year did the company Tesla release its first car?",
            answer="2008",
            supporting_docs=[
                "Tesla released its first car, the Roadster, in 2008.",
            ],
            hop_count=1,
            dataset="crag_fallback",
            sample_id="finance:simple:fallback-0",
        ),
        QASample(
            question="Who is the current king of the United States?",
            answer="The United States does not have a king (false premise).",
            supporting_docs=[
                "The United States is a federal republic with a president, not a monarchy.",
            ],
            hop_count=1,
            dataset="crag_fallback",
            sample_id="politics:false_premise:fallback-1",
        ),
        QASample(
            question="What was the population of Tokyo as of the most recent estimate?",
            answer="about 14 million",
            supporting_docs=[
                "As of the most recent estimate, the population of Tokyo is about 14 million.",
            ],
            hop_count=1,
            dataset="crag_fallback",
            sample_id="geography:temporal:fallback-2",
        ),
    ]


# ---------------------------------------------------------------------------
# Unified loader
# ---------------------------------------------------------------------------

_LOADERS = {
    "hotpotqa": load_hotpotqa,
    "musique": load_musique,
    "frames": load_frames,
    "crag": load_crag,
}


def frames_corpus_stats(samples: List[QASample]) -> Dict[str, Any]:
    """Compute corpus quality statistics for a list of QASamples (primarily for FRAMES).

    Returns a dict with:
    - ``n_samples``: total number of samples.
    - ``mean_docs_per_sample``: average passages per sample.
    - ``n_empty_corpus``: samples with zero supporting docs.
    - ``mean_passage_length``: average characters per passage (0 if no docs).
    - ``mean_answer_recall_in_corpus``: average fraction of answer tokens
      found anywhere in the combined passage corpus.
    - ``link_only_count`` / ``link_only_fraction``: samples whose docs appear
      to be bare URLs or short titles rather than actual passage text.  A high
      fraction means the dataset was loaded without ``fetch_passages=True``.

    Use the ``link_only_fraction`` to detect that FRAMES is running on bare
    Wikipedia links rather than fetched passages — a major caveat for headline
    FRAMES results.
    """
    n = len(samples)
    if n == 0:
        return {
            "n_samples": 0,
            "mean_docs_per_sample": 0.0,
            "n_empty_corpus": 0,
            "mean_passage_length": 0.0,
            "mean_answer_recall_in_corpus": 0.0,
            "link_only_count": 0,
            "link_only_fraction": 0.0,
        }

    total_docs = 0
    empty_count = 0
    total_chars = 0
    total_doc_count = 0
    recall_sum = 0.0
    recall_denom = 0
    link_only = 0

    for s in samples:
        docs = s.supporting_docs
        n_docs = len(docs)
        total_docs += n_docs
        if n_docs == 0:
            empty_count += 1
            continue

        for doc in docs:
            total_chars += len(doc)
            total_doc_count += 1

        corpus_text = " ".join(docs).lower()
        ans_tokens = set(s.answer.lower().split())
        if ans_tokens:
            found = sum(1 for t in ans_tokens if t in corpus_text)
            recall_sum += found / len(ans_tokens)
            recall_denom += 1

        is_link_only = all(
            "http" in doc or " " not in doc.strip() or len(doc.strip()) < 80
            for doc in docs
        )
        if is_link_only:
            link_only += 1

    return {
        "n_samples": n,
        "mean_docs_per_sample": total_docs / n,
        "n_empty_corpus": empty_count,
        "mean_passage_length": total_chars / total_doc_count if total_doc_count else 0.0,
        "mean_answer_recall_in_corpus": recall_sum / recall_denom if recall_denom else 0.0,
        "link_only_count": link_only,
        "link_only_fraction": link_only / n,
    }


def load_dataset(
    name: str,
    split: Optional[str] = None,
    max_samples: Optional[int] = None,
    **loader_kwargs: Any,
) -> List[QASample]:
    """Load a named dataset by string key.

    Parameters
    ----------
    name:
        One of ``"hotpotqa"``, ``"musique"``, ``"frames"``, ``"crag"``.
    split:
        Dataset split to load. When ``None``, each loader's own default is used
        (FRAMES/CRAG default to ``test`` / ``train``; HotpotQA/MuSiQue to
        ``validation``).
    max_samples:
        Optional cap.
    **loader_kwargs:
        Dataset-specific options forwarded to the underlying loader — e.g.
        ``fetch_passages=True`` for FRAMES to build a Wikipedia passage corpus.
    """
    name = name.lower()
    loader = _LOADERS.get(name)
    if loader is None:
        raise ValueError(
            f"Unknown dataset '{name}'. Choose from: {sorted(_LOADERS)}."
        )
    kwargs: Dict[str, Any] = {"max_samples": max_samples, **loader_kwargs}
    if split is not None:
        kwargs["split"] = split
    return loader(**kwargs)


def iter_batches(samples: List[QASample], batch_size: int = 32) -> Iterator[List[QASample]]:
    """Yield successive batches from a sample list."""
    for i in range(0, len(samples), batch_size):
        yield samples[i : i + batch_size]
