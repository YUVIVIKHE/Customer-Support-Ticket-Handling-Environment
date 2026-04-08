"""Pydantic v2 data models for the Customer Support Ticket Environment."""

from typing import Literal
from pydantic import BaseModel, field_validator


class Observation(BaseModel):
    """Input provided to the agent at each environment step."""

    ticket_text: str
    customer_type: Literal["premium", "normal"]
    urgency_level: Literal["low", "medium", "high"]
    previous_actions: list[str]


class Action(BaseModel):
    """Output produced by the agent at each environment step."""

    classify_ticket: Literal["billing", "technical", "complaint", "general"]
    priority: Literal["low", "medium", "high"]
    response_action: Literal["reply", "escalate", "ignore", "request_info"]


class Reward(BaseModel):
    """Reward signal returned after each environment step."""

    score: float
    breakdown: dict[str, float]
    done: bool

    @field_validator("score")
    @classmethod
    def clamp_score(cls, v: float) -> float:
        """Clamp score to [0.0, 1.0]."""
        return max(0.0, min(1.0, v))
