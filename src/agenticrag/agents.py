"""Real LLM agentic RAG pipeline (W1).

This module adds a provider-agnostic :class:`LLMAgent` that performs *real*,
self-directed iterative retrieval (ReAct-style decompose -> retrieve -> reason
-> re-retrieve -> answer) over the same ``retriever`` seam used by the
heuristic :class:`~agenticrag.core.AgenticRAGPipeline`.  The agent emits a
:class:`~agenticrag.core.PipelineTrace` with ``hop_queries`` / ``hop_docs`` /
``tool_calls`` populated from real steps, so failure injection and
``PropagationGraph`` work unchanged.

Three things make this the unblocker described in the research plan:

1. **Provider adapters** keep the agent loop identical across backbones.
   ``ClaudeProvider`` (primary, Opus 4.8 / Sonnet 4.6), ``OpenAIProvider``,
   and a deterministic ``MockProvider`` that needs no API key — so the test
   suite and offline smoke runs stay green.  Gemini / open-model adapters slot
   in behind the same :class:`LLMProvider` protocol.
2. **Resumability.**  ``LLMAgent.resume_from_hops`` re-runs the *suffix* of a
   trajectory from a (possibly corrupted) prefix, which is what turns static
   fault injection into a live causal intervention (W2): corrupt hop ``k`` then
   let the agent react for real.
3. **Cost accounting.**  Every provider call reports token usage; the agent
   accumulates it onto ``PipelineTrace.tokens_used`` for the deployability /
   cost-aware angle.

The heuristic pipeline is retained as a deterministic *control* condition.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple

from .core import (
    PipelineTrace,
    _token_overlap,
    reformulate_query,
)

# Default Claude backbone.  Opus 4.8 is the primary backbone in the plan; swap
# to "claude-sonnet-4-6" for cheaper headline grids via ClaudeProvider(model=...).
DEFAULT_CLAUDE_MODEL = "claude-opus-4-8"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

# Retry budget for the API clients. The SDKs retry connection errors + 429/5xx
# with exponential backoff; a higher budget rides out transient network outages
# (e.g. swapping Wi-Fi/ethernet) instead of failing a run. Tune via env var.
_DEFAULT_MAX_RETRIES = int(os.environ.get("AGENTICRAG_MAX_RETRIES", "8"))
# Per-request timeout (seconds). Our agent calls are short, so a low ceiling makes
# a dead socket (e.g. one orphaned by a mid-call network drop) fail fast and retry
# instead of hanging on the SDK's default 10-minute timeout.
_DEFAULT_TIMEOUT = float(os.environ.get("AGENTICRAG_TIMEOUT", "90"))


# --------------------------------------------------------------------------- #
# Provider abstraction                                                          #
# --------------------------------------------------------------------------- #

@dataclass
class LLMResponse:
    """A single provider completion plus token accounting."""

    text: str
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class LLMProvider(Protocol):
    """Minimal text-completion interface every backbone implements.

    Keeping the surface this small is what lets the agent loop stay identical
    across Claude / OpenAI / open models — only the adapter differs.
    """

    name: str
    model: str

    def generate(self, system: str, user: str, max_tokens: int = 1024) -> LLMResponse:
        ...


class ClaudeProvider:
    """Anthropic Claude backbone (primary).

    Uses ``client.messages.create``; no ``temperature`` / ``thinking`` config is
    sent, so it is valid on Opus 4.8 / 4.7 (which reject sampling params) as
    well as Sonnet 4.6.  Pass ``model="claude-sonnet-4-6"`` for cheaper grids.
    """

    def __init__(self, model: str = DEFAULT_CLAUDE_MODEL, client: Any = None) -> None:
        self.model = model
        self.name = f"claude:{model}"
        if client is None:
            import anthropic  # imported lazily so the package stays importable

            client = anthropic.Anthropic(
                max_retries=_DEFAULT_MAX_RETRIES, timeout=_DEFAULT_TIMEOUT
            )
        self._client = client

    def generate(self, system: str, user: str, max_tokens: int = 1024) -> LLMResponse:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        usage = getattr(resp, "usage", None)
        return LLMResponse(
            text=text,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
        )


class OpenAIProvider:
    """OpenAI chat-completions backbone (secondary)."""

    def __init__(self, model: str = DEFAULT_OPENAI_MODEL, client: Any = None) -> None:
        self.model = model
        self.name = f"openai:{model}"
        if client is None:
            import openai  # lazy import

            client = openai.OpenAI(
                max_retries=_DEFAULT_MAX_RETRIES, timeout=_DEFAULT_TIMEOUT
            )
        self._client = client

    def generate(self, system: str, user: str, max_tokens: int = 1024) -> LLMResponse:
        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        text = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        return LLMResponse(
            text=text,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )


class MockProvider:
    """Deterministic, API-key-free backbone for tests and offline smoke runs.

    It implements the same decode-the-prompt -> decide-an-action contract a real
    LLM would, using token overlap as a stand-in for reasoning.  Behaviour:

    - If no evidence has been gathered yet, it issues a search for the question.
    - Once evidence overlaps the question (a proxy for "I can answer"), or the
      hop budget is exhausted, it answers with the best-overlapping sentence.
    - Otherwise it reformulates and searches again, producing genuine multi-hop
      trajectories.

    This is *not* a real agent — it exists so the framework (injection,
    propagation, diagnosis) is exercised end-to-end without network access.
    Token counts are approximated from word counts so cost metrics are non-zero.
    """

    name = "mock:heuristic"
    model = "mock"

    # Marker lines the agent embeds; kept in sync with LLMAgent._build_prompt.
    _Q_RE = re.compile(r"^QUESTION:\s*(.*)$", re.MULTILINE)
    _HOP_RE = re.compile(r"^HOP:\s*(\d+)\s*/\s*(\d+)\s*$", re.MULTILINE)
    _EVIDENCE_RE = re.compile(r"^\[\d+\]\s*(.*)$", re.MULTILINE)

    def generate(self, system: str, user: str, max_tokens: int = 1024) -> LLMResponse:
        question = self._search(self._Q_RE, user)
        m = self._HOP_RE.search(user)
        # No HOP line => the agent's forced-answer prompt: always answer.
        is_answer_prompt = m is None
        hop, max_hop = (1, 3) if is_answer_prompt else (int(m.group(1)), int(m.group(2)))
        evidence = self._EVIDENCE_RE.findall(user)

        if is_answer_prompt:
            if evidence:
                best_doc, _ = self._best_evidence(question, evidence)
                decision: Dict[str, Any] = {"action": "answer", "answer": self._extract_answer(best_doc)}
            else:
                decision = {"action": "answer", "answer": ""}
        elif not evidence:
            decision = {"action": "search", "query": question}
        else:
            best_doc, best_score = self._best_evidence(question, evidence)
            sufficient = best_score > 0.0
            if sufficient or hop >= max_hop:
                decision = {"action": "answer", "answer": self._extract_answer(best_doc)}
            else:
                decision = {"action": "search", "query": reformulate_query(question, evidence)}

        text = json.dumps(decision)
        # Approximate token usage so cost-aware metrics have signal offline.
        in_toks = len(user.split()) + len(system.split())
        out_toks = max(1, len(text.split()))
        return LLMResponse(text=text, input_tokens=in_toks, output_tokens=out_toks)

    @staticmethod
    def _search(pattern: re.Pattern, text: str) -> str:
        m = pattern.search(text)
        return m.group(1).strip() if m else ""

    @staticmethod
    def _best_evidence(question: str, evidence: List[str]) -> Tuple[str, float]:
        scored = [(doc, _token_overlap(question, doc)) for doc in evidence]
        scored.sort(key=lambda x: -x[1])
        return scored[0]

    @staticmethod
    def _extract_answer(doc: str) -> str:
        first = doc.split(".")[0].strip()
        return first if first else doc


def make_provider(name: str, model: Optional[str] = None, client: Any = None) -> LLMProvider:
    """Construct a provider by short name: ``claude`` | ``openai`` | ``mock``.

    ``name`` also accepts ``provider:model`` (e.g. ``claude:claude-sonnet-4-6``).
    """
    if ":" in name and model is None:
        name, model = name.split(":", 1)
    key = name.lower()
    if key in ("mock", "heuristic", "offline"):
        return MockProvider()
    if key in ("claude", "anthropic"):
        return ClaudeProvider(model=model or DEFAULT_CLAUDE_MODEL, client=client)
    if key in ("openai", "gpt"):
        return OpenAIProvider(model=model or DEFAULT_OPENAI_MODEL, client=client)
    raise ValueError(
        f"Unknown provider '{name}'. Choose 'claude', 'openai', or 'mock'."
    )


# --------------------------------------------------------------------------- #
# Agent decision parsing                                                         #
# --------------------------------------------------------------------------- #

@dataclass
class AgentDecision:
    """Parsed agent action: either retrieve again or emit a final answer."""

    action: str  # "search" | "answer"
    query: str = ""
    answer: str = ""


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_decision(text: str) -> AgentDecision:
    """Parse a provider completion into an :class:`AgentDecision`.

    Accepts a JSON object ``{"action": "search"|"answer", ...}``.  Falls back to
    treating the raw text as a final answer when no parseable action is found —
    a real LLM that ignores the protocol still yields a usable answer rather
    than crashing the loop.
    """
    m = _JSON_OBJ_RE.search(text or "")
    if m:
        try:
            data = json.loads(m.group(0))
            action = str(data.get("action", "")).lower()
            if action == "search":
                query = str(data.get("query", "")).strip()
                if query:
                    return AgentDecision(action="search", query=query)
            if action == "answer":
                return AgentDecision(action="answer", answer=str(data.get("answer", "")).strip())
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass
    return AgentDecision(action="answer", answer=(text or "").strip())


# --------------------------------------------------------------------------- #
# LLMAgent                                                                       #
# --------------------------------------------------------------------------- #

_SYSTEM_PROMPT = (
    "You are a retrieval agent answering a multi-hop question. You have one tool: "
    "a document retriever. At each step decide whether to retrieve more evidence "
    "or to answer.\n"
    "Respond with a single JSON object and nothing else:\n"
    '  {"action": "search", "query": "<a focused sub-question>"} to retrieve more, or\n'
    '  {"action": "answer", "answer": "<final answer>"} when the evidence is sufficient.\n'
    "Decompose the question across hops: each search should target the next missing "
    "fact, not repeat a prior query. Answer concisely with the specific fact asked for."
)


@dataclass
class HopState:
    """One executed retrieval hop: the query issued and the docs returned."""

    query: str
    docs: List[str] = field(default_factory=list)


class LLMAgent:
    """ReAct-style agent doing real, self-directed iterative retrieval.

    The agent owns the loop; the LLM owns each decision (which sub-query to issue,
    when to stop). It is resumable so that injecting a fault at hop ``k`` and
    re-running the suffix produces the agent's genuine reaction (W2), not a
    static edit.

    Parameters
    ----------
    provider:
        Backbone implementing :class:`LLMProvider`.  Defaults to
        :class:`MockProvider` so construction never requires an API key.
    retriever:
        Object with ``retrieve(query, corpus, top_k) -> [(doc, score), ...]``
        (``BM25Retriever``, ``DenseRetriever``, ``TokenOverlapRetriever``).  When
        ``None``, falls back to token-overlap ranking — identical to
        ``AgenticRAGPipeline`` so the control and LLM conditions share retrieval.
    max_iterations:
        Hop budget.  The agent may answer earlier.
    top_k:
        Documents returned per retrieval.
    """

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        retriever: Any = None,
        max_iterations: int = 3,
        top_k: int = 5,
        system_prompt: str = _SYSTEM_PROMPT,
    ) -> None:
        self.provider = provider or MockProvider()
        self._retriever = retriever
        self.max_iterations = max_iterations
        self.top_k = top_k
        self.system_prompt = system_prompt

    # ------------------------------------------------------------------ #
    # Retrieval (mirrors AgenticRAGPipeline._retrieve so conditions match) #
    # ------------------------------------------------------------------ #

    def _retrieve(self, query: str, docs: List[str]) -> List[str]:
        if not docs:
            return []
        if self._retriever is not None:
            ranked = self._retriever.retrieve(query, corpus=docs, top_k=self.top_k)
            relevant = [doc for doc, score in ranked if score > 0]
            return relevant if relevant else docs[:2]
        ranked = sorted(
            ((doc, _token_overlap(query, doc)) for doc in docs), key=lambda x: -x[1]
        )[: self.top_k]
        relevant = [doc for doc, score in ranked if score > 0]
        return relevant if relevant else docs[:2]

    # ------------------------------------------------------------------ #
    # Public entry points                                                  #
    # ------------------------------------------------------------------ #

    def run(
        self,
        query: str,
        corpus: List[str],
        reference_answer: str = "",
    ) -> PipelineTrace:
        """Run the agent from scratch and return a populated PipelineTrace."""
        return self._run(query, corpus, reference_answer, prefix=[], start_hop=1)

    def resume_from_hops(
        self,
        query: str,
        corpus: List[str],
        prefix: List[HopState],
        reference_answer: str = "",
        start_hop: Optional[int] = None,
    ) -> PipelineTrace:
        """Re-run the trajectory suffix from a (possibly corrupted) prefix.

        ``prefix`` is the list of already-executed hops — exactly as they should
        appear in the resumed trace (a caller may have emptied / replaced one
        hop's docs to inject a fault). The agent treats those docs as the
        evidence it has so far and continues deciding from ``start_hop`` (default
        ``len(prefix) + 1``), letting downstream propagation be the agent's real
        reaction rather than a static edit.
        """
        if start_hop is None:
            start_hop = len(prefix) + 1
        return self._run(query, corpus, reference_answer, prefix=prefix, start_hop=start_hop)

    def force_answer(
        self,
        query: str,
        prefix: List[HopState],
        reference_answer: str = "",
    ) -> PipelineTrace:
        """Make the agent answer *now* from ``prefix`` evidence, with no further retrieval.

        Models premature collapse / early termination (W2): the agent is forced
        to commit an answer from partial evidence and we observe its real
        (degraded) output rather than a static edit.
        """
        hops = [HopState(query=h.query, docs=list(h.docs)) for h in prefix]
        tool_calls = [
            {"name": "retrieve", "args": {"q": h.query}, "iteration": i + 1}
            for i, h in enumerate(hops)
        ]
        evidence = self._evidence(hops)
        resp = self.provider.generate(
            self.system_prompt, self._build_answer_prompt(query, evidence)
        )
        final_answer = self._answer_from(resp)

        all_docs: List[str] = []
        for h in hops:
            all_docs.extend(h.docs)
        return PipelineTrace(
            query=query,
            retrieved_docs=list(dict.fromkeys(all_docs)),
            tool_calls=tool_calls,
            final_answer=final_answer,
            reference_answer=reference_answer,
            hop_queries=[h.query for h in hops],
            hop_docs=[h.docs for h in hops],
            iterations_used=max(1, len(hops)),
            tokens_used=resp.total_tokens,
        )

    # ------------------------------------------------------------------ #
    # Core loop                                                            #
    # ------------------------------------------------------------------ #

    def _run(
        self,
        query: str,
        corpus: List[str],
        reference_answer: str,
        prefix: List[HopState],
        start_hop: int,
    ) -> PipelineTrace:
        hops: List[HopState] = [HopState(query=h.query, docs=list(h.docs)) for h in prefix]
        tool_calls: List[Dict[str, Any]] = [
            {"name": "retrieve", "args": {"q": h.query}, "iteration": i + 1}
            for i, h in enumerate(hops)
        ]
        tokens_used = 0
        final_answer = ""

        hop = start_hop
        while hop <= self.max_iterations:
            evidence = self._evidence(hops)
            resp = self.provider.generate(
                self.system_prompt, self._build_prompt(query, evidence, hop)
            )
            tokens_used += resp.total_tokens
            decision = parse_decision(resp.text)

            if decision.action == "answer" and (evidence or hop > 1):
                final_answer = decision.answer
                break

            sub_query = decision.query or query
            retrieved = self._retrieve(sub_query, corpus)
            hops.append(HopState(query=sub_query, docs=retrieved))
            tool_calls.append(
                {"name": "retrieve", "args": {"q": sub_query}, "iteration": len(hops)}
            )
            hop = len(hops) + 1

        # Force a final answer if the agent exhausted its budget without one.
        if not final_answer:
            evidence = self._evidence(hops)
            resp = self.provider.generate(
                self.system_prompt, self._build_answer_prompt(query, evidence)
            )
            tokens_used += resp.total_tokens
            final_answer = self._answer_from(resp)

        all_docs: List[str] = []
        for h in hops:
            all_docs.extend(h.docs)

        return PipelineTrace(
            query=query,
            retrieved_docs=list(dict.fromkeys(all_docs)),
            tool_calls=tool_calls,
            final_answer=final_answer,
            reference_answer=reference_answer,
            hop_queries=[h.query for h in hops],
            hop_docs=[h.docs for h in hops],
            iterations_used=max(1, len(hops)),
            tokens_used=tokens_used,
        )

    # ------------------------------------------------------------------ #
    # Prompt construction                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _answer_from(resp: LLMResponse) -> str:
        """Extract the final answer from a forced-answer completion.

        Respects an explicit ``answer`` action even when the answer is the empty
        string (a genuine "I can't answer"); only falls back to the raw text when
        the model failed to emit an answer action.
        """
        decision = parse_decision(resp.text)
        if decision.action == "answer":
            return decision.answer
        return (resp.text or "").strip()

    @staticmethod
    def _evidence(hops: List[HopState]) -> List[str]:
        seen: List[str] = []
        for h in hops:
            for doc in h.docs:
                if doc not in seen:
                    seen.append(doc)
        return seen

    def _build_prompt(self, question: str, evidence: List[str], hop: int) -> str:
        return (
            f"QUESTION: {question}\n"
            f"HOP: {hop} / {self.max_iterations}\n\n"
            f"EVIDENCE:\n{self._format_evidence(evidence)}\n\n"
            "Decide your next action (search for the next missing fact, or answer)."
        )

    def _build_answer_prompt(self, question: str, evidence: List[str]) -> str:
        return (
            f"QUESTION: {question}\n\n"
            f"EVIDENCE:\n{self._format_evidence(evidence)}\n\n"
            'You have used your retrieval budget. Respond now with '
            '{"action": "answer", "answer": "<final answer>"} using the evidence above.'
        )

    @staticmethod
    def _format_evidence(evidence: List[str]) -> str:
        if not evidence:
            return "(none retrieved yet)"
        return "\n".join(f"[{i + 1}] {doc}" for i, doc in enumerate(evidence))
