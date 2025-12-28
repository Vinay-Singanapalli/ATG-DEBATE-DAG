from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, TypedDict


Speaker = Literal["A", "B"]
Status = Literal["OK", "ERROR"]


class Turn(TypedDict, total=False):
    round: int
    agent: str
    speaker: Speaker
    text: str
    meta: Dict[str, Any]

class Verdict(TypedDict, total=False):
    summary: str
    winner: str
    justification: str
    coherence_flags: List[Dict[str, Any]]



class DebateState(TypedDict, total=False):
    # User input
    raw_topic: str
    topic: str

    verdict: Verdict

    argument_embeddings: List[List[float]]
    topic_embedding: List[float]
    repetition_max_cosine: float
    topic_min_cosine: float
    format_violations: List[Dict[str, Any]]

    # LLM / embeddings runtime config (must be in schema or it may be filtered)
    llm_model: str
    judge_model: str
    embed_model: str
    llm_temperature: float
    llm_max_tokens: int


    # Control
    max_rounds: int
    round_idx: int  # number of ACCEPTED turns already in memory (0..max_rounds)
    next_speaker: Speaker
    status: Status
    error: str

    # Debate memory (structured)
    turns: List[Turn]
    summary: str

    # Memory slices (each agent only gets its own slice)
    memory_for_a: Dict[str, Any]
    memory_for_b: Dict[str, Any]

    # Validation indexes
    argument_norms: List[str]          # normalized strings for repetition checks
    coherence_flags: List[Dict[str, Any]]

    # Execution config
    seed: Optional[int]
    log_path: str

    # Pipeline handoff fields
    pending_speaker: Speaker
    pending_agent_name: str
    pending_text: str
    last_node: str
