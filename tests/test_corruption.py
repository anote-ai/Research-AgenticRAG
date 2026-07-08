"""Tests for certified content corruption (corruption.py + live injection wiring)."""
from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agenticrag.agents import LLMAgent, MockProvider
from agenticrag.core import FailureStage, FailureType
from agenticrag.corruption import (
    CorruptionRecord,
    classify_absorption,
    corrupt_docs,
    make_replacement,
    select_target_span,
)
from agenticrag.evaluate import corruption_absorption_rates
from agenticrag.experiment import run_identifiability
from agenticrag.injection import LiveFailureInjector
from agenticrag.retrievers import BM25Retriever

CORPUS = [
    "Apple Inc. produces the iPhone.",
    "Tim Cook is the CEO of Apple Inc.",
    "Bananas are a good source of potassium.",
    "The Eiffel Tower is in Paris.",
]
QUESTION = "Who is the CEO of the company that produces the iPhone?"


def _agent():
    return LLMAgent(provider=MockProvider(), retriever=BM25Retriever(), max_iterations=3)


# --------------------------------------------------------------------------- #
# Span selection                                                                #
# --------------------------------------------------------------------------- #

class TestSelectTargetSpan:
    def test_gold_answer_present_wins(self):
        docs = ["Tim Cook is the CEO of Apple Inc."]
        span, strategy = select_target_span(
            docs, later_queries=[], question=QUESTION, reference_answer="Tim Cook"
        )
        assert span == "Tim Cook"
        assert strategy == "answer_fact"

    def test_bridge_entity_when_gold_absent(self):
        docs = ["The film Parasite won Best Picture in 2020."]
        later = ["Who directed Parasite?"]
        span, strategy = select_target_span(
            docs,
            later_queries=later,
            question="Who directed the film that won Best Picture in 2020?",
            reference_answer="Bong Joon-ho",
        )
        assert span == "Parasite"
        assert strategy == "bridge_entity"

    def test_question_entities_are_not_bridges(self):
        # "Best Picture" appears in the question, so it must not be selected as
        # a bridge even though it also appears in a later query.
        docs = ["The film Parasite won Best Picture."]
        later = ["Which film won Best Picture?", "Who directed Parasite?"]
        span, strategy = select_target_span(
            docs,
            later_queries=later,
            question="Who directed the film that won Best Picture in 2020?",
            reference_answer="Bong Joon-ho",
        )
        assert span == "Parasite"
        assert strategy == "bridge_entity"

    def test_salient_entity_fallback(self):
        docs = ["The Eiffel Tower is in Paris. The Eiffel Tower opened in 1889."]
        span, strategy = select_target_span(
            docs, later_queries=[], question="", reference_answer="something absent"
        )
        assert strategy == "salient_entity"
        assert span in ("Eiffel Tower", "The Eiffel Tower", "Paris", "1889")

    def test_no_span_returns_none(self):
        assert select_target_span([], reference_answer="x") is None
        assert (
            select_target_span(
                ["and or but if then maybe"], reference_answer="absent gold"
            )
            is None
        )

    def test_yes_no_gold_falls_through(self):
        docs = ["Yes, the Eiffel Tower is in Paris."]
        span, strategy = select_target_span(
            docs, later_queries=[], question="", reference_answer="yes"
        )
        assert strategy != "answer_fact"


# --------------------------------------------------------------------------- #
# Replacement generation                                                        #
# --------------------------------------------------------------------------- #

class TestMakeReplacement:
    def test_year_perturbed(self):
        repl, kind = make_replacement("1889", random.Random(0))
        assert kind == "numeric"
        assert repl != "1889"
        assert repl.isdigit() and abs(int(repl) - 1889) <= 5

    def test_entity_drawn_from_pool(self):
        pool = ["Sundar Pichai", "Satya Nadella"]
        repl, kind = make_replacement("Tim Cook", random.Random(0), pool)
        assert kind == "entity"
        assert repl in pool

    def test_pool_entries_overlapping_original_are_filtered(self):
        pool = ["Tim Apple", "Satya Nadella"]  # "Tim" shared with original
        repl, _ = make_replacement("Tim Cook", random.Random(0), pool)
        assert repl == "Satya Nadella"

    def test_replacement_not_already_in_docs(self):
        pool = ["Paris", "Satya Nadella"]
        repl, _ = make_replacement(
            "Tim Cook", random.Random(0), pool,
            avoid_texts=["The Eiffel Tower is in Paris."],
        )
        assert repl == "Satya Nadella"

    def test_fallback_pool_when_no_pool_given(self):
        repl, kind = make_replacement("Tim Cook", random.Random(0))
        assert kind == "entity"
        assert repl and repl != "Tim Cook"

    def test_deterministic_for_same_seed(self):
        pool = ["Sundar Pichai", "Satya Nadella", "Jensen Huang"]
        a, _ = make_replacement("Tim Cook", random.Random(7), pool)
        b, _ = make_replacement("Tim Cook", random.Random(7), pool)
        assert a == b


# --------------------------------------------------------------------------- #
# Document rewriting                                                            #
# --------------------------------------------------------------------------- #

class TestCorruptDocs:
    def test_case_insensitive_whole_word_replacement(self):
        docs = ["Tim Cook leads Apple.", "TIM COOK spoke today.", "Timothy Cooke is unrelated."]
        new_docs, n_docs, n_repl = corrupt_docs(docs, "Tim Cook", "Satya Nadella")
        assert new_docs[0] == "Satya Nadella leads Apple."
        assert new_docs[1] == "Satya Nadella spoke today."
        assert new_docs[2] == "Timothy Cooke is unrelated."  # no substring damage
        assert n_docs == 2 and n_repl == 2

    def test_originals_not_mutated(self):
        docs = ["Tim Cook leads Apple."]
        corrupt_docs(docs, "Tim Cook", "X")
        assert docs[0] == "Tim Cook leads Apple."


# --------------------------------------------------------------------------- #
# Deterministic generation-level evaluation                                     #
# --------------------------------------------------------------------------- #

class TestClassifyAbsorption:
    def test_correct_answer_is_resisted(self):
        assert classify_absorption("Tim Cook", "Tim Cook", "Satya Nadella") == "resisted"

    def test_corrupted_value_in_answer_is_absorbed(self):
        assert (
            classify_absorption("The CEO is Satya Nadella.", "Tim Cook", "Satya Nadella")
            == "absorbed"
        )

    def test_neither_is_derailed(self):
        assert classify_absorption("I don't know", "Tim Cook", "Satya Nadella") == "derailed"
        assert classify_absorption("", "Tim Cook", "Satya Nadella") == "derailed"

    def test_correctness_dominates_absorption(self):
        # Answer contains both the gold and the corrupted value: correctness wins.
        assert (
            classify_absorption(
                "Tim Cook, not Satya Nadella", "Tim Cook", "Satya Nadella"
            )
            == "resisted"
        )


# --------------------------------------------------------------------------- #
# Live injector integration                                                     #
# --------------------------------------------------------------------------- #

class TestInjectCorruptedEvidence:
    def test_returns_certified_record(self):
        agent = _agent()
        base = agent.run(QUESTION, CORPUS, reference_answer="Tim Cook")
        inj = LiveFailureInjector(agent, distractor_pool=["Satya Nadella"])
        res = inj.inject_corrupted_evidence(base, CORPUS, hop=1)
        assert res is not None
        assert res.injected_failure_type == FailureType.CORRUPTED_EVIDENCE
        assert res.injected_stage == FailureStage.RETRIEVAL
        assert isinstance(res.corruption, CorruptionRecord)
        assert res.corruption.n_replacements >= 1
        # The corrupted span must actually be gone from the injected hop's docs
        # and the replacement present.
        hop_docs = " ".join(res.injected_trace.hop_docs[0])
        assert res.corruption.original_span not in hop_docs
        assert res.corruption.corrupted_span in hop_docs

    def test_docs_stay_topically_intact(self):
        agent = _agent()
        base = agent.run(QUESTION, CORPUS, reference_answer="Tim Cook")
        inj = LiveFailureInjector(agent, distractor_pool=["Satya Nadella"])
        res = inj.inject_corrupted_evidence(base, CORPUS, hop=1)
        assert res is not None
        # Unlike noise-doc injection, the rest of each document survives.
        assert any("CEO" in d or "Apple" in d for d in res.injected_trace.hop_docs[0])

    def test_deterministic_across_calls(self):
        agent = _agent()
        base = agent.run(QUESTION, CORPUS, reference_answer="Tim Cook")
        inj = LiveFailureInjector(
            agent, distractor_pool=["Satya Nadella", "Sundar Pichai"], seed=13
        )
        a = inj.inject_corrupted_evidence(base, CORPUS, hop=1)
        b = inj.inject_corrupted_evidence(base, CORPUS, hop=1)
        assert a.corruption.corrupted_span == b.corruption.corrupted_span

    def test_returns_none_without_corruptible_span(self):
        agent = _agent()
        base = agent.run(QUESTION, CORPUS, reference_answer="Tim Cook")
        # Force the hop docs to contain nothing corruptible.
        base.hop_docs[0] = ["and or but if then"]
        base.reference_answer = "absent gold"
        inj = LiveFailureInjector(agent)
        assert inj.inject_corrupted_evidence(base, CORPUS, hop=1) is None


# --------------------------------------------------------------------------- #
# Driver integration (run_identifiability with the corruption arm)              #
# --------------------------------------------------------------------------- #

class _Sample:
    def __init__(self, question, answer, docs):
        self.question = question
        self.answer = answer
        self.supporting_docs = docs
        self.hop_count = 1
        self.dataset = "unit"
        self.sample_id = "s1"


class TestDriverIntegration:
    def test_corruption_metadata_and_skip_accounting(self):
        agent = _agent()
        samples = [_Sample(QUESTION, "Tim Cook", CORPUS)]
        injector = LiveFailureInjector(agent, distractor_pool=["Satya Nadella"])
        from agenticrag.diagnosers import RuleBasedDiagnoser

        result = run_identifiability(
            agent,
            samples,
            {"rule_based": RuleBasedDiagnoser()},
            hops=[1],
            injection_methods=["inject_corrupted_evidence"],
            injector=injector,
        )
        assert 1 in result.n_skipped_no_span_by_depth  # counter present
        raw = result.raw_by_depth[1]
        assert "recovered_metadata" in raw
        all_meta = raw["metadata"] + raw["recovered_metadata"]
        assert all_meta, "expected at least one injected corruption case"
        m = all_meta[0]
        assert m["intervention_method"] == "inject_corrupted_evidence"
        assert m["absorption"] in ("absorbed", "resisted", "derailed")
        assert m["original_span"] and m["corrupted_span"]
        assert "query_contaminated" in m
        # Round-trips through to_dict for persistence.
        d = result.to_dict()
        assert "n_skipped_no_span_by_depth" in d

    def test_absorption_rates_from_persisted_dict(self):
        persisted = {
            "raw_by_depth": {
                "1": {
                    "metadata": [
                        {"absorption": "absorbed"},
                        {"absorption": "derailed"},
                        {"intervention_method": "inject_empty_retrieval"},  # no label
                    ],
                    "recovered_metadata": [{"absorption": "resisted"}],
                }
            }
        }
        rates = corruption_absorption_rates(persisted)
        assert rates[1]["n"] == 3.0
        assert abs(rates[1]["absorbed"] - 1 / 3) < 1e-9
        assert abs(rates[1]["resisted"] - 1 / 3) < 1e-9
        assert abs(rates[1]["derailed"] - 1 / 3) < 1e-9

    def test_absorption_rates_empty_when_no_corruption_cases(self):
        assert corruption_absorption_rates({"raw_by_depth": {}}) == {}
