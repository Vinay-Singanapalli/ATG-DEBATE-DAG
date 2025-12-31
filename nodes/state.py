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
    winner: Literal["AgentA", "AgentB"]
    justification: str
    coherenceflags: List[Dict[str, Any]]


class DebateState(TypedDict, total=False):
    # ---- runtime / config ----
    logpath: str
    seed: Optional[int]

    maxrounds: int
    maxretries: int
    gotojudge: bool

    agentaname: str
    agentbname: str

    llmmodel: str
    llmtemperature: float
    llmmaxtokens: int
    judgemodel: str

    # ---- user input ----
    rawtopic: str
    topic: str

    # ---- controller ----
    roundidx: int
    nextspeaker: Speaker

    status: Status
    error: str

    lastnode: str

    # ---- pending handoff (CRITICAL: must exist in schema) ----
    pendingspeaker: Speaker
    pendingagentname: str
    pendingtext: str

    # ---- debate memory ----
    turns: List[Turn]
    summary: str

    memoryfora: Dict[str, Any]
    memoryforb: Dict[str, Any]

    # ---- validation / checks ----
    coherenceflags: List[Dict[str, Any]]
    formatviolations: List[Dict[str, Any]]

    retrycount: int
    retryreason: str
    lastrejectedtext: str
    rejectionhistory: List[Dict[str, Any]]

    usedquotes: List[str]

    # judge output
    verdict: Optional[Verdict]
