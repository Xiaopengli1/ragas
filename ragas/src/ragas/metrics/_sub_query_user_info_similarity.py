from __future__ import annotations

import typing as t
from dataclasses import dataclass, field

import numpy as np

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics.base import (
    MetricOutputType,
    MetricType,
    MetricWithEmbeddings,
    SingleTurnMetric,
)

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks


@dataclass
class SubQueryUserInfoSimilarity(MetricWithEmbeddings, SingleTurnMetric):
    """Semantic similarity between sub-queries and the original query with user profile."""

    name: str = "sub_query_user_info_similarity"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {"user_input", "reference_contexts", "user_profile"}
        }
    )
    output_type = MetricOutputType.CONTINUOUS

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        row = sample.to_dict()
        return await self._ascore(row, callbacks)

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        assert self.embeddings is not None, "Embeddings must be set"

        query: str = row["user_input"]
        sub_queries: t.Sequence[str] = row["reference_contexts"]
        user_profile: str = row.get("user_profile", "") or ""

        combined = f"{user_profile} {query}".strip()

        base_emb = np.asarray(await self.embeddings.embed_text(combined))
        sub_embs = np.asarray(await self.embeddings.embed_documents(list(sub_queries)))

        base_norm = np.linalg.norm(base_emb)
        sub_norms = np.linalg.norm(sub_embs, axis=1)
        similarities = (sub_embs @ base_emb) / (sub_norms * base_norm + 1e-8)
        return float(np.mean(similarities))


sub_query_user_info_similarity = SubQueryUserInfoSimilarity()
