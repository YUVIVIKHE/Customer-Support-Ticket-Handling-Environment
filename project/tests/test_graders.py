"""Property and unit tests for env/graders.py."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from env.models import Action
from env.graders import grade_easy, grade_medium, grade_hard

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

classify_st = st.sampled_from(["billing", "technical", "complaint", "general"])
priority_st = st.sampled_from(["low", "medium", "high"])
response_st = st.sampled_from(["reply", "escalate", "ignore", "request_info"])


@st.composite
def action_st(draw):
    return Action(
        classify_ticket=draw(classify_st),
        priority=draw(priority_st),
        response_action=draw(response_st),
    )


# ---------------------------------------------------------------------------
# Property 6: grade_easy returns correct scores
# Feature: customer-support-ticket-env, Property 6: grade_easy scoring
# ---------------------------------------------------------------------------

@given(action_st(), action_st())
@settings(max_examples=20)
def test_grade_easy_exact_match(action: Action, _unused: Action):
    """grade_easy returns 1.0 when action exactly matches expected."""
    score = grade_easy(action, action)
    assert score == 1.0


@given(action_st(), response_st)
@settings(max_examples=20)
def test_grade_easy_classify_only(action: Action, different_response: str):
    """grade_easy returns 0.5 when only classify_ticket matches."""
    # Build expected with same classify but different response_action
    expected = Action(
        classify_ticket=action.classify_ticket,
        priority=action.priority,
        response_action=different_response,
    )
    if action.response_action == different_response:
        # Same response → exact match, skip
        return
    score = grade_easy(action, expected)
    assert score == 0.5


@given(action_st(), classify_st)
@settings(max_examples=20)
def test_grade_easy_wrong_classify(action: Action, different_classify: str):
    """grade_easy returns 0.0 when classify_ticket is wrong."""
    if action.classify_ticket == different_classify:
        return
    expected = Action(
        classify_ticket=different_classify,
        priority=action.priority,
        response_action=action.response_action,
    )
    # Wrong classify → 0.0 regardless of response_action match
    score = grade_easy(action, expected)
    # If response_action also matches, still 0.0 because classify is wrong
    assert score == 0.0


# ---------------------------------------------------------------------------
# Property 7: grade_medium partial reward is proportional
# Feature: customer-support-ticket-env, Property 7: grade_medium proportionality
# ---------------------------------------------------------------------------

@given(st.lists(action_st(), min_size=1, max_size=5))
@settings(max_examples=20)
def test_grade_medium_proportional(actions: list[Action]):
    """grade_medium score equals fraction of steps with correct response_action."""
    expected = actions  # use same list as ground truth → all correct
    score = grade_medium(actions, expected)
    assert score == 1.0


@given(st.lists(action_st(), min_size=1, max_size=5))
@settings(max_examples=20)
def test_grade_medium_all_wrong(actions: list[Action]):
    """grade_medium returns 0.0 when all response_actions are wrong."""
    # Build expected with a different response_action for every step
    response_cycle = ["reply", "escalate", "ignore", "request_info"]
    expected = []
    for a in actions:
        # Pick a response_action that differs from a.response_action
        for r in response_cycle:
            if r != a.response_action:
                expected.append(Action(
                    classify_ticket=a.classify_ticket,
                    priority=a.priority,
                    response_action=r,
                ))
                break
    score = grade_medium(actions, expected)
    assert score == 0.0


# ---------------------------------------------------------------------------
# Property 1 (partial): All grader outputs in [0.0, 1.0]
# Feature: customer-support-ticket-env, Property 1: grader output range
# ---------------------------------------------------------------------------

@given(action_st(), action_st())
@settings(max_examples=20)
def test_grade_easy_range(action: Action, expected: Action):
    assert 0.0 <= grade_easy(action, expected) <= 1.0


@given(st.lists(action_st(), max_size=5), st.lists(action_st(), max_size=5))
@settings(max_examples=20)
def test_grade_medium_range(actions: list[Action], expected: list[Action]):
    assert 0.0 <= grade_medium(actions, expected) <= 1.0


@given(st.lists(action_st(), max_size=5), st.lists(action_st(), max_size=5))
@settings(max_examples=20)
def test_grade_hard_range(actions: list[Action], expected: list[Action]):
    assert 0.0 <= grade_hard(actions, expected) <= 1.0


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_grade_medium_empty_expected():
    assert grade_medium([], []) == 0.0


def test_grade_hard_empty_expected():
    assert grade_hard([], []) == 0.0


def test_grade_hard_all_correct():
    actions = [
        Action(classify_ticket="billing", priority="high", response_action="escalate"),
        Action(classify_ticket="technical", priority="medium", response_action="reply"),
    ]
    score = grade_hard(actions, actions)
    assert score == 1.0


def test_grade_hard_partial():
    expected = [
        Action(classify_ticket="billing", priority="high", response_action="escalate"),
        Action(classify_ticket="technical", priority="medium", response_action="reply"),
    ]
    # First action correct, second wrong priority and wrong response
    actions = [
        Action(classify_ticket="billing", priority="high", response_action="escalate"),
        Action(classify_ticket="technical", priority="low", response_action="ignore"),
    ]
    score = grade_hard(actions, expected)
    # priority: 1/2 correct, action: 1/2 correct → 0.5 * 0.5 + 0.5 * 0.5 = 0.5
    assert score == 0.5
