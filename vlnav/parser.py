"""Instruction parser. Deterministic rule-based parser for scenario 1 (no VLM needed during development).
`parse_fields` is plain Python (tested without pydantic); `MissionSpec` is the pydantic spec used downstream.
A prompt the rules do not understand raises ParseError -> the mission manager may ask the VLM parser instead."""
from __future__ import annotations

import re
from typing import Dict

ORDINALS = {"first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3, "next": 1, "nearest": 1}
TARGET_WORDS = ("intersection", "junction", "crossroad", "crossroads", "crossing")
UNSUPPORTED = ("turn", "then", "after that", "and then", "afterwards", "left at", "right at", "land", "return")


class ParseError(ValueError):
    pass


def parse_fields(prompt: str) -> Dict:
    t = re.sub(r"[^a-z0-9\s]", " ", prompt.lower())
    t = re.sub(r"\s+", " ", t).strip()
    if not any(w in t for w in TARGET_WORDS):
        raise ParseError("no road intersection target found")
    if "road" not in t and "street" not in t:
        raise ParseError("no road to follow")
    if not re.search(r"\bstop\b|\bhover\b", t):
        raise ParseError("no stop action")
    for w in UNSUPPORTED:
        if re.search(rf"\b{w}\b", t):
            raise ParseError(f"unsupported instruction ('{w}'): scenario-1 rules cover a single road-follow-and-stop")
    ordinal = next((n for w, n in ORDINALS.items() if re.search(rf"\b{w}\b", t)), 1)
    if "left" in t.split() or "right" in t.split():
        raise ParseError("road direction left/right not supported by the rules")
    return {"takeoff": bool(re.search(r"\btake ?off\b", t)), "road_direction": "straight_ahead",
            "target_type": "road_intersection", "ordinal": ordinal, "action_at_target": "stop",
            "goal_text": "a road intersection", "source": "rules"}


try:
    from typing import Literal
    from pydantic import BaseModel, Field

    class MissionSpec(BaseModel):
        takeoff: bool = True
        road_direction: Literal["straight_ahead"] = "straight_ahead"
        target_type: Literal["road_intersection"] = "road_intersection"
        ordinal: int = Field(1, ge=1, le=5)
        action_at_target: Literal["stop"] = "stop"
        goal_text: str = "a road intersection"
        source: Literal["rules", "vlm"] = "rules"

    def parse_rules(prompt: str) -> "MissionSpec":
        return MissionSpec(**parse_fields(prompt))
except ImportError:                                          # pydantic missing: parse_fields still works
    MissionSpec = None                                       # type: ignore

    def parse_rules(prompt: str):                            # type: ignore
        raise ImportError("pip install pydantic")
