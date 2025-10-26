
from enum import Enum
from textwrap import dedent
from typing import Any, Generator
from pydantic import Field, BaseModel
from xml.sax.saxutils import escape

from ask.core.agent import AgentASK
from ask.core.context import example
from ask.core.memory import NoMemory
from ask.core.config import EmbedderConfig, Config, LLMConfig


class AnalysisInput(BaseModel):
    query: str = Field(..., description="The analysis query or prompt")
    response: str = Field(..., description="The content to be analyzed")

class AnalysisOutput(BaseModel):
    context: str = Field("Summary of the content", description=(
        "One sentence summarizing with: "
        "- Main topic/domain.\n"
        "- Key arguments/points.\n"
        "- Intended audience/purpose.\n"
    ))
    keywords: list[str] = Field(..., description=(
        "Several specific, distinct keywords that capture key concepts and terminology."
        "Order from most to least important."
        "Don't include keywords that are the name of the speaker or time."
        "At least three keywords, but don't be too redundant."
    ))
    tags: list[str] = Field(..., description=(
        "Several broad categories/themes for classification."
        "Include domain, format, and type tags."
        "At least three tags, but don't be too redundant."
    ))


class RerankInput(BaseModel):
    query: str = Field(..., description="The search query to rank responses against")
    responses: list[str] = Field(..., description="Array of responses to be ranked")


class RerankOutput(BaseModel):
    ranks: list[int] = Field(..., description=(
        "Array of rank scores from 0 to 100 for each response. "
        "Each score represents how relevant the corresponding response is to the query. "
        "100 = highly relevant, 0 = not relevant at all. "
        "The array must have the same length as the input responses array."
    ))


def create_analysis_agent(llm: LLMConfig): return AgentASK[AnalysisInput, AnalysisOutput].create_from_dict({
    "agent": {
        "name": "Analysis",
        "instructions": dedent(f"""
            Generate a structured analysis of the following content by:
            1. Identifying the most salient keywords (focus on nouns, verbs, and key concepts)
            2. Extracting core themes and contextual elements
            3. Creating relevant categorical tags

            Print only the structured analysis in JSON format.
            No additional text or explanation.
            
            Output example:
            {example(AnalysisOutput(
                context="A brief summary of the content.",
                keywords=["keyword1", "keyword2", "keyword3"],
                tags=["tag1", "tag2", "tag3"]
            ))}
        """),
        "input_type": AnalysisInput,
        "output_type": AnalysisOutput,
    },
    "llm": llm,
}, 
NoMemory())


def create_rerank_agent(llm: LLMConfig): return AgentASK[RerankInput, RerankOutput].create_from_dict({
    "agent": {
        "name": "Rerank",
        "instructions": dedent(f"""
            You are a relevance ranking system. Given a query and a list of responses, 
            assign each response a relevance score from 0 to 100:
            - 100: Highly relevant, directly answers or relates to the query
            - 75-99: Very relevant, substantial overlap with query topic
            - 50-74: Moderately relevant, partial relation to query
            - 25-49: Somewhat relevant, tangential connection
            - 0-24: Not relevant, little to no connection
            
            Evaluate each response independently based on:
            1. Semantic similarity to the query
            2. How well it addresses the query's intent
            3. Topical overlap and context relevance
            
            Return the ranks as an array of integers from 0 to 100, in the same order as the input responses.
            
            Output example:
            {example(RerankOutput(
                ranks=[95, 78, 45, 12, 88]
            ))}
        """),
        "input_type": RerankInput,
        "output_type": RerankOutput,
    },
    "llm": llm,
}, 
NoMemory())
