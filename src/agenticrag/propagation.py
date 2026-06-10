from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

from pydantic import BaseModel, Field

from .core import FailureRecord, FailureStage
from .injection import InjectionLabel


class PropagationEdge(BaseModel):
    """A directed propagation event between two pipeline stages."""

    source: FailureStage
    target: FailureStage
    count: int = 0


class PropagationGraph(BaseModel):
    """Count and estimate stage-to-stage failure propagation probabilities."""

    edges: Dict[Tuple[FailureStage, FailureStage], int] = Field(default_factory=dict)

    def add_edge(
        self,
        source: FailureStage,
        target: FailureStage,
        count: int = 1,
    ) -> None:
        if count < 1:
            raise ValueError(f"count must be >= 1, got {count}")
        key = (source, target)
        self.edges[key] = self.edges.get(key, 0) + count

    def add_label(self, label: InjectionLabel) -> None:
        self.add_edge(label.root_cause_stage, label.observed_stage)

    def add_record(
        self,
        record: FailureRecord,
        root_cause_stage: FailureStage | str | None = None,
    ) -> None:
        source = _coerce_stage(root_cause_stage or record.root_cause or record.stage)
        self.add_edge(source, record.stage)

    def fit_labels(self, labels: Iterable[InjectionLabel]) -> "PropagationGraph":
        for label in labels:
            self.add_label(label)
        return self

    def fit_records(self, records: Iterable[FailureRecord]) -> "PropagationGraph":
        for record in records:
            if record.stage == FailureStage.NONE:
                continue
            self.add_record(record)
        return self

    def edge_count(self, source: FailureStage, target: FailureStage) -> int:
        return self.edges.get((source, target), 0)

    def outgoing_total(self, source: FailureStage) -> int:
        return sum(count for (src, _), count in self.edges.items() if src == source)

    def edge_probability(self, source: FailureStage, target: FailureStage) -> float:
        total = self.outgoing_total(source)
        if total == 0:
            return 0.0
        return self.edge_count(source, target) / total

    def transition_matrix(self) -> Dict[str, Dict[str, float]]:
        matrix: Dict[str, Dict[str, float]] = defaultdict(dict)
        stages = sorted(
            {stage for edge in self.edges for stage in edge},
            key=lambda stage: stage.value,
        )
        for source in stages:
            for target in stages:
                matrix[source.value][target.value] = self.edge_probability(source, target)
        return dict(matrix)

    def most_likely_targets(self, source: FailureStage) -> List[PropagationEdge]:
        ranked = [
            PropagationEdge(source=src, target=target, count=count)
            for (src, target), count in self.edges.items()
            if src == source
        ]
        return sorted(ranked, key=lambda edge: (-edge.count, edge.target.value))

    @classmethod
    def from_labels(cls, labels: Iterable[InjectionLabel]) -> "PropagationGraph":
        return cls().fit_labels(labels)

    @classmethod
    def from_records(cls, records: Iterable[FailureRecord]) -> "PropagationGraph":
        return cls().fit_records(records)


def _coerce_stage(stage: FailureStage | str) -> FailureStage:
    if isinstance(stage, FailureStage):
        return stage
    return FailureStage(stage)
