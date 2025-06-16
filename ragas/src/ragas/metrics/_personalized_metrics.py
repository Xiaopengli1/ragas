from __future__ import annotations

import math
import typing as t
from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics.base import (
    MetricOutputType,
    MetricType,
    MetricWithEmbeddings,
    MetricWithLLM,
    SingleTurnMetric,
)
from ragas.prompt import PydanticPrompt

if t.TYPE_CHECKING:
    from langchain_core.callbacks.base import Callbacks


# Metric 1: Personalized Retrieval Success
@dataclass
class PersonalizedRetrievalNDCG(SingleTurnMetric):
    """Compute nDCG for personalized document retrieval."""

    name: str = "personalized_retrieval_ndcg"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {"retrieved_contexts", "reference_contexts"}
        }
    )
    output_type: MetricOutputType = MetricOutputType.CONTINUOUS

    def init(self, run_config):
        pass

    def _ndcg(self, relevances: t.List[int]) -> float:
        dcg = sum(r / math.log2(i + 2) for i, r in enumerate(relevances))
        ideal = sorted(relevances, reverse=True)
        idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
        return dcg / idcg if idcg > 0 else np.nan

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        retrieved = sample.retrieved_contexts or []
        refs = set(sample.reference_contexts or [])
        relevances = [1 if doc in refs else 0 for doc in retrieved]
        if not relevances:
            return np.nan
        return self._ndcg(relevances)

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        return await self._single_turn_ascore(SingleTurnSample(**row), callbacks)


@dataclass
class PersonalizedRetrievalRecall(SingleTurnMetric):
    """Recall@k for personalized document retrieval."""

    k: int = 10
    name: str = field(init=False)
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {"retrieved_contexts", "reference_contexts"}
        }
    )
    output_type: MetricOutputType = MetricOutputType.CONTINUOUS

    def __post_init__(self):
        self.name = f"personalized_retrieval_recall@{self.k}"

    def init(self, run_config):
        pass

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        retrieved = sample.retrieved_contexts or []
        refs = set(sample.reference_contexts or [])
        if not refs:
            return np.nan
        retrieved_k = retrieved[: self.k]
        hits = sum(1 for r in refs if r in retrieved_k)
        return hits / len(refs)

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        return await self._single_turn_ascore(SingleTurnSample(**row), callbacks)


# Metric 2: Semantic relevance
@dataclass
class QueryDocSemanticRelevance(MetricWithEmbeddings, SingleTurnMetric):
    """Semantic similarity between query and retrieved documents."""

    name: str = "query_doc_semantic_relevance"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {"user_input", "retrieved_contexts"}
        }
    )
    output_type: MetricOutputType = MetricOutputType.CONTINUOUS

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        assert self.embeddings is not None, "Embeddings not set"
        query = sample.user_input or ""
        docs = sample.retrieved_contexts or []
        if not docs:
            return np.nan
        q_emb = np.array(await self.embeddings.embed_text(query))
        d_embs = np.array(await self.embeddings.embed_texts(docs))
        q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-10)
        d_embs = d_embs / (np.linalg.norm(d_embs, axis=1, keepdims=True) + 1e-10)
        sims = d_embs @ q_emb
        return float(np.mean(sims))

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        return await self._single_turn_ascore(SingleTurnSample(**row), callbacks)


# Metric 3: Personalized relevance using LLM
class RelevanceInput(BaseModel):
    query: str
    document: str


class RelevanceOutput(BaseModel):
    score: float


class PersonalizedRelevancePrompt(PydanticPrompt[RelevanceInput, RelevanceOutput]):
    name: str = "personalized_relevance"
    instruction: str = (
        "Given a query and a personalized document, rate their relevance on a scale of 0 (not relevant) to 5 (fully relevant)."
    )
    input_model = RelevanceInput
    output_model = RelevanceOutput


@dataclass
class QueryDocPersonalizedRelevance(MetricWithLLM, SingleTurnMetric):
    """LLM judged personalized relevance between query and documents."""

    name: str = "query_doc_personalized_relevance"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {"user_input", "retrieved_contexts"}
        }
    )
    output_type: MetricOutputType = MetricOutputType.CONTINUOUS
    relevance_prompt: PydanticPrompt = field(
        default_factory=PersonalizedRelevancePrompt
    )
    max_retries: int = 1

    def init(self, run_config):
        if self.llm is None:
            raise ValueError("LLM is not set")

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        assert self.llm is not None, "LLM not set"
        query = sample.user_input or ""
        docs = sample.retrieved_contexts or []
        if not docs:
            return np.nan
        scores = []
        for doc in docs:
            resp = await self.relevance_prompt.generate(
                data=RelevanceInput(query=query, document=doc),
                llm=self.llm,
                callbacks=callbacks,
            )
            scores.append(resp.score)
        return float(np.mean(scores)) if scores else np.nan

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        return await self._single_turn_ascore(SingleTurnSample(**row), callbacks)
