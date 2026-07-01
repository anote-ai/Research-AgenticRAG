"""Integration tests for run_identifiability: raw persistence + checkpointing."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agenticrag.agents import LLMAgent, MockProvider
from agenticrag.datasets import QASample
from agenticrag.diagnosers import PropagationAwareDiagnoser, RuleBasedDiagnoser
from agenticrag.experiment import IdentifiabilityResult, run_identifiability
from agenticrag.retrievers import TokenOverlapRetriever


def _agent():
    return LLMAgent(provider=MockProvider(), retriever=TokenOverlapRetriever(), max_iterations=3)


def _samples():
    return [
        QASample(
            question="who is the ceo of apple", answer="tim cook",
            supporting_docs=[
                "who is the ceo of apple tim cook",
                "apple makes the iphone device",
                "bananas are a yellow fruit",
            ], hop_count=1, dataset="demo",
        ),
        QASample(
            question="what city has the eiffel tower", answer="paris",
            supporting_docs=[
                "eiffel tower paris france city",
                "apple makes the iphone device",
            ], hop_count=1, dataset="demo",
        ),
    ]


def _diagnosers(agent):
    return {"rule_based": RuleBasedDiagnoser(), "propagation_aware": PropagationAwareDiagnoser(agent)}


class TestRawPersistence:
    def test_result_has_raw_by_depth(self):
        agent = _agent()
        res = run_identifiability(agent, _samples(), _diagnosers(agent), hops=(1, 2))
        assert isinstance(res, IdentifiabilityResult)
        assert set(res.raw_by_depth.keys()) == {1, 2}
        for depth in (1, 2):
            entry = res.raw_by_depth[depth]
            assert "truth" in entry and "predictions" in entry
            # One prediction list per diagnoser, aligned with truth length.
            for name in ("rule_based", "propagation_aware"):
                assert len(entry["predictions"][name]) == len(entry["truth"])

    def test_to_dict_serializable_and_roundtrips_rescore(self):
        agent = _agent()
        res = run_identifiability(agent, _samples(), _diagnosers(agent), hops=(1, 2))
        d = res.to_dict()
        s = json.dumps(d)  # must be JSON-serializable
        from agenticrag.evaluate import rescore_identifiability

        out = rescore_identifiability(json.loads(s), criterion="stage")
        assert set(out.keys()) == {"rule_based", "propagation_aware"}


class TestPerDepthResume:
    CKPT = {
        "max_samples": 2, "propagation_budget": 2, "retriever": "bm25",
        "accuracy": {"a": {"1": 0.5}},
        "cost": {"a": {"1": {"total_cost": 1.0}}},
        "recovery_rate_by_depth": {"1": 0.3},
        "n_total_by_depth": {"1": 4},
        "n_failed_by_depth": {"1": 2},
        "raw_by_depth": {"1": {"truth": [["retrieval", 1]], "predictions": {"a": [["retrieval", 1, 5]]}}},
    }
    MATCH = {"max_samples": 2, "propagation_budget": 2, "retriever": "bm25"}

    def _accumulators(self):
        return {"a": {}}, {"a": {}}, {}, {}, {}, {}

    def test_reuses_matching_config(self, tmp_path):
        from agenticrag.experiment import _preload_checkpoint
        ckpt = tmp_path / "c.json"
        ckpt.write_text(json.dumps(self.CKPT))
        acc, cost, rec, nt, nf, raw = self._accumulators()
        done = _preload_checkpoint(str(ckpt), self.MATCH, ["a"], acc, cost, rec, nt, nf, raw)
        assert done == {1}
        assert acc["a"][1] == 0.5
        assert raw[1]["truth"] == [["retrieval", 1]]

    def test_skips_on_config_mismatch(self, tmp_path):
        from agenticrag.experiment import _preload_checkpoint
        ckpt = tmp_path / "c.json"
        ckpt.write_text(json.dumps(self.CKPT))
        acc, cost, rec, nt, nf, raw = self._accumulators()
        # Different max_samples -> must NOT reuse (recompute from scratch).
        done = _preload_checkpoint(str(ckpt), {"max_samples": 999}, ["a"], acc, cost, rec, nt, nf, raw)
        assert done == set()
        assert acc["a"] == {}

    def test_missing_file_returns_empty(self, tmp_path):
        from agenticrag.experiment import _preload_checkpoint
        acc, cost, rec, nt, nf, raw = self._accumulators()
        done = _preload_checkpoint(str(tmp_path / "nope.json"), self.MATCH, ["a"], acc, cost, rec, nt, nf, raw)
        assert done == set()

    def test_resume_end_to_end_skips_done_depth(self, tmp_path):
        # Depth 1 done with matching config -> a resume run over hops (1,2) keeps
        # depth 1 and only computes depth 2.
        agent = _agent()
        ckpt = tmp_path / "c.json"
        ckpt.write_text(json.dumps({
            **self.MATCH,
            "accuracy": {"rule_based": {"1": 0.11}, "propagation_aware": {"1": 0.22}},
            "cost": {"rule_based": {"1": {}}, "propagation_aware": {"1": {}}},
            "recovery_rate_by_depth": {"1": 0.5}, "n_total_by_depth": {"1": 4},
            "n_failed_by_depth": {"1": 2},
            "raw_by_depth": {"1": {"truth": [], "predictions": {"rule_based": [], "propagation_aware": []}}},
        }))
        res = run_identifiability(
            agent, _samples(), _diagnosers(agent), hops=(1, 2),
            checkpoint_path=str(ckpt), checkpoint_extra=self.MATCH, resume=True,
        )
        # Depth 1 carried over verbatim from the checkpoint (not recomputed).
        assert res.accuracy["propagation_aware"][1] == 0.22
        assert 2 in res.raw_by_depth  # depth 2 freshly computed


class TestCheckpointing:
    def test_checkpoint_written_per_depth(self, tmp_path):
        agent = _agent()
        ckpt = tmp_path / "ckpt.json"
        run_identifiability(
            agent, _samples(), _diagnosers(agent), hops=(1, 2),
            checkpoint_path=str(ckpt), checkpoint_extra={"provider": "mock", "dataset": "demo"},
        )
        assert ckpt.exists()
        data = json.loads(ckpt.read_text())
        assert data["provider"] == "mock"
        assert "raw_by_depth" in data
        # Both depths present after completion.
        assert set(data["raw_by_depth"].keys()) == {"1", "2"}

    def test_checkpoint_atomic_no_tmp_left(self, tmp_path):
        agent = _agent()
        ckpt = tmp_path / "ckpt.json"
        run_identifiability(
            agent, _samples(), _diagnosers(agent), hops=(1,),
            checkpoint_path=str(ckpt),
        )
        assert ckpt.exists()
        assert not (tmp_path / "ckpt.json.tmp").exists()
