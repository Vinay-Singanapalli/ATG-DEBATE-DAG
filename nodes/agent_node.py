from __future__ import annotations

import json
from typing import Dict, List

from langchain_ollama import ChatOllama

from nodes.state import DebateState
from nodes.semantic import cosine_similarity
from nodes.llm_provider import build_embeddings


used_quotes = set(state.get("used_quotes", []))

def _embed_text(emb, text: str) -> List[float]:
    return emb.embed_query(text)


def _max_similarity(vec: List[float], prev: List[List[float]]) -> float:
    best = 0.0
    for p in prev:
        best = max(best, cosine_similarity(vec, p))
    return best


def _first_sentence(s: str) -> str:
    s = (s or "").replace("\r", "\n").replace("\n", " ").strip()
    if not s:
        return ""
    # crude first sentence split
    for sep in [". ", "? ", "! "]:
        if sep in s:
            return s.split(sep, 1)[0].strip() + sep.strip()
    return s[:180].strip()


def agent_node(state: DebateState) -> DebateState:
    speaker = state["pending_speaker"]          # "A" or "B"
    agent_name = state["pending_agent_name"]    # "Scientist"/"Philosopher" or AgentA/AgentB

    # Ensure correct turn order
    if speaker != state.get("next_speaker"):
        return {
            "status": "ERROR",
            "error": "Out-of-turn agent execution.",
            "last_node": f"Agent({agent_name})(ERROR)",
        }

    turn_no = int(state.get("round_idx", 0)) + 1
    topic = state["topic"]
    memory = state["memory_for_a"] if speaker == "A" else state["memory_for_b"]

    # Always initialize violations so NameError can never happen
    vio = list(state.get("format_violations", []))

    # Models / params
    llm_model = state.get("llm_model", "llama3.2:1b")
    temperature = float(state.get("llm_temperature", 0.2))
    max_tokens = int(state.get("llm_max_tokens", 220))

    # Embeddings + thresholds
    emb = build_embeddings(state.get("embed_model", "mxbai-embed-large"))

    repetition_thr = float(state.get("repetition_max_cosine", 0.97))
    topic_min = float(state.get("topic_min_cosine", 0.40))
    prev_embs = list(state.get("argument_embeddings", []))

    # Opponent last turn text (if your memory stores it)
    last_opp = (memory or {}).get("last_opponent_turn") or {}
    opp_text = (last_opp.get("text") or "").strip()

    # Topic embedding once
    topic_emb = state.get("topic_embedding") or []
    if not topic_emb:
        topic_emb = _embed_text(emb, topic)

    persona_hint = (
        "Scientist: use evidence, feasibility, safety, measurable constraints."
        if speaker == "A"
        else "Philosopher: use ethics, definitions, governance, societal impact."
    )

    # ---- Structured output schema for Ollama ----
    # Ollama supports constraining output using a JSON schema passed via `format`. [web:266][web:269]
    turn_schema: Dict = {
        "type": "object",
        "properties": {
            "quote": {"type": "string"},
            "rebut": {"type": "string"},
            "new": {"type": "string"},
            "question": {"type": "string"},
        },
        "required": ["quote", "rebut", "new", "question"],
        "additionalProperties": False,
    }

    llm = ChatOllama(
        model=llm_model,
        temperature=temperature,
        num_predict=max_tokens,
        format=turn_schema,
    )

    quote_rule = (
        "Round 1: quote MUST be exactly '(none)'."
        if not opp_text
        else "quote MUST copy ONE sentence verbatim from the opponent last turn."
    )

    system = (
        f"You are {agent_name} in a structured debate.\n"
        f"Topic: {topic}\n"
        f"Persona: {persona_hint}\n"
        "Return ONLY valid JSON matching this schema:\n"
        '{"quote": "...", "rebut": "...", "new": "...", "question": "..."}\n'
        f"{quote_rule}\n"
        "rebut must directly respond to quote.\n"
        "new must be a NEW argument (not a question).\n"
        "question must be ONE pointed question.\n"
        "No markdown. No extra keys.\n"
    )

    user = "Write your next turn."
    if opp_text:
        user += f"\nOpponent last turn:\n{opp_text}"

    # Safe fallbacks (so we never crash)
    last_text = (
        f"QUOTE: {( '(none)' if not opp_text else _first_sentence(opp_text)[:160] )}\n"
        f"REBUT: This point misses an important tradeoff in '{topic}' (Round {turn_no}, Speaker {speaker}).\n"
        f"NEW: A separate consideration is unintended consequences, reversibility, and oversight specific to '{topic}'.\n"
        f"QUESTION: What concrete evidence would change your stance on '{topic}'?\n"
    )
    last_vec = _embed_text(emb, f"{topic} turn {turn_no} speaker {speaker}")

    for _attempt in range(1, 7):
        msg = llm.invoke(
            [{"role": "system", "content": system}, {"role": "user", "content": user}]
        )
        raw = getattr(msg, "content", str(msg)).strip()

        # Parse model JSON
        try:
            data = json.loads(raw)
        except Exception:
            vio.append({"turn": turn_no, "speaker": speaker, "agent": agent_name, "reason": "Agent returned non-JSON."})
            continue

        quote = (data.get("quote") or "").strip()
        rebut = (data.get("rebut") or "").strip()
        new = (data.get("new") or "").strip()
        question = (data.get("question") or "").strip()

        # Enforce quote rule
        if not opp_text:
            quote = "(none)"
        else:
            # Must be substring (verbatim-ish)
            # Prefer quoting the opponent's REBUT sentence (reduces "quoting the question" loops)
            opp_rebut = ""
            for line in opp_text.splitlines():
                if line.strip().lower().startswith("rebut:"):
                    opp_rebut = line.split(":", 1)[1].strip()
                    break
            target_text = opp_rebut if opp_rebut else opp_text
            if quote and quote not in target_text:
                vio.append({"turn": turn_no, "speaker": speaker, "agent": agent_name, "reason": "QUOTE not from opponent REBUT/turn."})
                continue


        # Basic non-empty checks
        if not rebut:
            vio.append({"turn": turn_no, "speaker": speaker, "agent": agent_name, "reason": "Empty rebut."})
            continue
        if not new:
            vio.append({"turn": turn_no, "speaker": speaker, "agent": agent_name, "reason": "Empty new."})
            continue
        if not question or len(question) < 5:
            vio.append({"turn": turn_no, "speaker": speaker, "agent": agent_name, "reason": "Empty/too-short question."})
            continue

        # Build canonical text (what your existing pipeline prints/logs)
        text = f"QUOTE: {quote}\nREBUT: {rebut}\nNEW: {new}\nQUESTION: {question}\n"
        last_text = text

        # Embedding-based checks on argument only
        arg_text = f"{rebut}\n{new}\n{question}"
        vec = _embed_text(emb, arg_text)
        last_vec = vec

        rep = _max_similarity(vec, prev_embs) if prev_embs else 0.0
        if rep >= repetition_thr:
            vio.append({"turn": turn_no, "speaker": speaker, "agent": agent_name, "reason": f"Repetition too high (cos={rep:.2f})."})
            continue

        top = cosine_similarity(vec, topic_emb)
        if top < topic_min:
            vio.append({"turn": turn_no, "speaker": speaker, "agent": agent_name, "reason": f"Topic drift too high (cos={top:.2f})."})
            continue

        if quote in used_quotes and quote != "(none)":
            vio.append({"turn": turn_no, "speaker": speaker, "agent": agent_name, "reason": "Repeated QUOTE."})
            continue


        # SUCCESS
        return {
            "pending_text": text,
            "argument_embeddings": prev_embs + [vec],
            "topic_embedding": topic_emb,
            "format_violations": vio,
            "last_node": f"Agent({agent_name})",
            "used_quotes": list(used_quotes | {quote}),
        }

    # If all attempts fail, return last safe text/vector
    return {
        "pending_text": last_text,
        "argument_embeddings": prev_embs + [last_vec],
        "topic_embedding": topic_emb,
        "format_violations": vio,
        "last_node": f"Agent({agent_name})(FALLBACK_ACCEPTED)",
    }




