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


# Metric 4: Semantic relevance between query and generated response
@dataclass
class QueryResponseSemanticRelevance(MetricWithEmbeddings, SingleTurnMetric):
    """Semantic similarity between the query and generated response."""

    name: str = "query_response_semantic_relevance"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {MetricType.SINGLE_TURN: {"user_input", "response"}}
    )
    output_type: MetricOutputType = MetricOutputType.CONTINUOUS

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        assert self.embeddings is not None, "Embeddings not set"
        query = sample.user_input or ""
        resp = sample.response or ""
        if resp == "":
            return np.nan
        q_emb = np.array(await self.embeddings.embed_text(query))
        r_emb = np.array(await self.embeddings.embed_text(resp))
        q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-10)
        r_emb = r_emb / (np.linalg.norm(r_emb) + 1e-10)
        return float((q_emb @ r_emb.T).item())

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        return await self._single_turn_ascore(SingleTurnSample(**row), callbacks)


# Metric 5: Semantic coverage of query by generated response using LLM
class SemanticCoverageInput(BaseModel):
    query: str
    response: str


class SemanticCoverageOutput(BaseModel):
    score: int


class SemanticCoveragePrompt(
    PydanticPrompt[SemanticCoverageInput, SemanticCoverageOutput]
):
    name: str = "semantic_coverage"
    instruction: str = (
        "Given a query and a generated response, rate how well the response covers "
        "all key aspects of the query on a scale of 0 (no coverage) to 5 (complete coverage)."
    )
    input_model = SemanticCoverageInput
    output_model = SemanticCoverageOutput


@dataclass
class QueryResponseSemanticCoverage(MetricWithLLM, SingleTurnMetric):
    """LLM judged semantic coverage of the response against the query."""

    name: str = "query_response_semantic_coverage"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {MetricType.SINGLE_TURN: {"user_input", "response"}}
    )
    output_type: MetricOutputType = MetricOutputType.CONTINUOUS
    coverage_prompt: PydanticPrompt = field(default_factory=SemanticCoveragePrompt)
    max_retries: int = 1

    def init(self, run_config):
        if self.llm is None:
            raise ValueError("LLM is not set")

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        assert self.llm is not None, "LLM not set"
        query = sample.user_input or ""
        resp = sample.response or ""
        if resp == "":
            return np.nan
        output = await self.coverage_prompt.generate(
            data=SemanticCoverageInput(query=query, response=resp),
            llm=self.llm,
            callbacks=callbacks,
        )
        return float(output.score)

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        return await self._single_turn_ascore(SingleTurnSample(**row), callbacks)


# Metric 6: Style consistency of generated response with predefined style
class StyleConsistencyInput(BaseModel):
    style: str
    response: str


class StyleConsistencyOutput(BaseModel):
    score: int


class StyleConsistencyPrompt(
    PydanticPrompt[StyleConsistencyInput, StyleConsistencyOutput]
):
    name: str = "style_consistency"
    instruction: str = (
        "Given a target writing style and a generated response, rate how well the response matches the style "
        "on a scale of 0 (not consistent) to 5 (fully consistent)."
    )
    input_model = StyleConsistencyInput
    output_model = StyleConsistencyOutput


@dataclass
class StyleConsistencyScore(MetricWithLLM, SingleTurnMetric):
    """LLM judged style consistency of the response."""

    name: str = "style_consistency_score"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {MetricType.SINGLE_TURN: {"user_profile", "response"}}
    )
    output_type: MetricOutputType = MetricOutputType.CONTINUOUS
    style_prompt: PydanticPrompt = field(default_factory=StyleConsistencyPrompt)
    max_retries: int = 1

    def init(self, run_config):
        if self.llm is None:
            raise ValueError("LLM is not set")

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        assert self.llm is not None, "LLM not set"
        style = sample.user_profile or ""
        resp = sample.response or ""
        if resp == "":
            return np.nan
        output = await self.style_prompt.generate(
            data=StyleConsistencyInput(style=style, response=resp),
            llm=self.llm,
            callbacks=callbacks,
        )
        return float(output.score)

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        return await self._single_turn_ascore(SingleTurnSample(**row), callbacks)


# Metric 7: Format consistency of generated response with predefined format
class FormatConsistencyInput(BaseModel):
    format: str
    response: str


class FormatConsistencyOutput(BaseModel):
    score: int


class FormatConsistencyPrompt(
    PydanticPrompt[FormatConsistencyInput, FormatConsistencyOutput]
):
    name: str = "format_consistency"
    instruction: str = (
        "Given a target response format and a generated response, rate how well the response follows the format "
        "on a scale of 0 (not consistent) to 5 (fully consistent)."
    )
    input_model = FormatConsistencyInput
    output_model = FormatConsistencyOutput


@dataclass
class FormatConsistencyScore(MetricWithLLM, SingleTurnMetric):
    """LLM judged format consistency of the response."""

    name: str = "format_consistency_score"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {MetricType.SINGLE_TURN: {"reference", "response"}}
    )
    output_type: MetricOutputType = MetricOutputType.CONTINUOUS
    format_prompt: PydanticPrompt = field(default_factory=FormatConsistencyPrompt)
    max_retries: int = 1

    def init(self, run_config):
        if self.llm is None:
            raise ValueError("LLM is not set")

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        assert self.llm is not None, "LLM not set"
        fmt = sample.reference or ""
        resp = sample.response or ""
        if resp == "":
            return np.nan
        output = await self.format_prompt.generate(
            data=FormatConsistencyInput(format=fmt, response=resp),
            llm=self.llm,
            callbacks=callbacks,
        )
        return float(output.score)

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        return await self._single_turn_ascore(SingleTurnSample(**row), callbacks)


# Metric 8: Knowledge conflict evaluation via refusal accuracy
class RefusalJudgeInput(BaseModel):
    query: str
    user_profile: str | None = None
    response: str


class RefusalJudgeOutput(BaseModel):
    refused: bool


class RefusalJudgePrompt(PydanticPrompt[RefusalJudgeInput, RefusalJudgeOutput]):
    name: str = "refusal_judge"
    instruction: str = (
        "Determine whether the response refuses to answer the query based on the user profile. "
        "Return `true` if it is a refusal, otherwise `false`."
    )
    input_model = RefusalJudgeInput
    output_model = RefusalJudgeOutput


@dataclass
class KnowledgeConflictAccuracy(MetricWithLLM, SingleTurnMetric):
    """Accuracy of refusal behavior when query conflicts with knowledge boundaries."""

    name: str = "knowledge_conflict_accuracy"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {
                "user_input",
                "user_profile",
                "response",
                "reference",
            }
        }
    )
    output_type: MetricOutputType = MetricOutputType.DISCRETE
    judge_prompt: PydanticPrompt = field(default_factory=RefusalJudgePrompt)
    max_retries: int = 1

    def init(self, run_config):
        if self.llm is None:
            raise ValueError("LLM is not set")

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        assert self.llm is not None, "LLM not set"
        query = sample.user_input or ""
        profile = sample.user_profile or ""
        resp = sample.response or ""
        expected = bool(sample.reference)
        output = await self.judge_prompt.generate(
            data=RefusalJudgeInput(query=query, user_profile=profile, response=resp),
            llm=self.llm,
            callbacks=callbacks,
        )
        predicted = output.refused
        return float(predicted == expected)

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        return await self._single_turn_ascore(SingleTurnSample(**row), callbacks)
