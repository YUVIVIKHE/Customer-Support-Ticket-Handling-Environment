"""Property and unit tests for env/models.py."""

import json
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from env.models import Action, Observation, Reward

# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

classify_ticket_st = st.sampled_from(["billing", "technical", "complaint", "general"])
priority_st = st.sampled_from(["low", "medium", "high"])
response_action_st = st.sampled_from(["reply", "escalate", "ignore", "request_info"])
customer_type_st = st.sampled_from(["premium", "normal"])
urgency_level_st = st.sampled_from(["low", "medium", "high"])


@st.composite
def action_strategy(draw):
    return Action(
        classify_ticket=draw(classify_ticket_st),
        priority=draw(priority_st),
        response_action=draw(response_action_st),
    )


@st.composite
def observation_strategy(draw):
    return Observation(
        ticket_text=draw(st.text(min_size=1, max_size=200)),
        customer_type=draw(customer_type_st),
        urgency_level=draw(urgency_level_st),
        previous_actions=draw(st.lists(st.text(max_size=50), max_size=5)),
    )


# ---------------------------------------------------------------------------
# Property 4: Action serialization round-trip
# Feature: customer-support-ticket-env, Property 4: Action serialization round-trip
# ---------------------------------------------------------------------------

@given(action_strategy())
@settings(max_examples=20)
def test_action_round_trip(action: Action):
    """For any valid Action, serializing to JSON and back produces an equivalent object."""
    serialized = action.model_dump_json()
    restored = Action.model_validate_json(serialized)
    assert restored == action


# ---------------------------------------------------------------------------
# Property 5: Observation serialization round-trip
# Feature: customer-support-ticket-env, Property 5: Observation serialization round-trip
# ---------------------------------------------------------------------------

@given(observation_strategy())
@settings(max_examples=20)
def test_observation_round_trip(obs: Observation):
    """For any valid Observation, serializing to JSON and back produces an equivalent object."""
    serialized = obs.model_dump_json()
    restored = Observation.model_validate_json(serialized)
    assert restored == obs


# ---------------------------------------------------------------------------
# Property 2: Reward score clamping
# Feature: customer-support-ticket-env, Property 2: Reward score clamping
# ---------------------------------------------------------------------------

@given(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6))
@settings(max_examples=20)
def test_reward_score_clamped(raw_score: float):
    """For any float, Reward.score is clamped to [0.0, 1.0]."""
    reward = Reward(score=raw_score, breakdown={}, done=False)
    assert 0.0 <= reward.score <= 1.0


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_reward_score_in_range_stays():
    r = Reward(score=0.5, breakdown={"a": 0.5}, done=False)
    assert r.score == 0.5


def test_reward_score_above_one_clamped():
    r = Reward(score=1.5, breakdown={}, done=False)
    assert r.score == 1.0


def test_reward_score_below_zero_clamped():
    r = Reward(score=-0.5, breakdown={}, done=False)
    assert r.score == 0.0
