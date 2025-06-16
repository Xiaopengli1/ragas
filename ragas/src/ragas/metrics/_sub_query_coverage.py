from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics.base import (
    MetricOutputType,
    MetricType,
    MetricWithLLM,
    SingleTurnMetric,
)
from ragas.prompt import PydanticPrompt

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks

logger = logging.getLogger(__name__)


class SubQueryCoverageInput(BaseModel):
    query: str = Field(..., description="Original query")
    sub_queries: t.List[str] = Field(..., description="List of sub queries")


class SubQueryCoverageOutput(BaseModel):
    reason: str = Field(..., description="Reason for the score")
    score: int = Field(..., description="Coverage score from 0 to 5")


class SubQueryCoveragePrompt(
    PydanticPrompt[SubQueryCoverageInput, SubQueryCoverageOutput]
):
    """Prompt for judging semantic coverage of sub queries."""

    name: str = "sub_query_semantic_coverage"
    instruction: str = (
        "Given an original query and a list of sub-queries, assess how completely "
        "the sub-queries cover the semantic meaning of the original query. "
        "Return a score from 0 to 5 where 5 means complete coverage and 0 means no "
        "coverage at all."
    )
    input_model = SubQueryCoverageInput
    output_model = SubQueryCoverageOutput
    examples = [
        (
            SubQueryCoverageInput(
                query="Books about quantum mechanics and relativity",
                sub_queries=["quantum mechanics books", "relativity theory books"],
            ),
            SubQueryCoverageOutput(
                reason="Both main topics are captured.",
                score=5,
            ),
        ),
        (
            SubQueryCoverageInput(
                query="How to bake a chocolate cake",
                sub_queries=["cake ingredients", "how to frost a cake"],
            ),
            SubQueryCoverageOutput(
                reason="Sub-queries miss the baking steps",
                score=2,
            ),
        ),
    ]


@dataclass
class SubQueryCoverage(MetricWithLLM, SingleTurnMetric):
    """Evaluate how well sub queries cover the original query semantics."""

    name: str = "sub_query_semantic_coverage"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {"user_input", "reference_contexts"}
        }
    )
    output_type: t.Optional[MetricOutputType] = MetricOutputType.DISCRETE
    coverage_prompt: PydanticPrompt = field(default_factory=SubQueryCoveragePrompt)
    max_retries: int = 1

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        row = sample.to_dict()
        return await self._ascore(row, callbacks)

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        assert self.llm is not None, "LLM is not set"

        query = row["user_input"]
        sub_queries = row["reference_contexts"]
        output: SubQueryCoverageOutput = await self.coverage_prompt.generate(
            data=SubQueryCoverageInput(query=query, sub_queries=sub_queries),
            llm=self.llm,
            callbacks=callbacks,
        )
        return output.score


sub_query_semantic_coverage = SubQueryCoverage()
