"""Property and unit tests for env/environment.py."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from env.environment import CustomerSupportEnv
from env.models import Action

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

classify_st = st.sampled_from(["billing", "technical", "complaint", "general"])
priority_st = st.sampled_from(["low", "medium", "high"])
response_st = st.sampled_from(["reply", "escalate", "ignore", "request_info"])
task_name_st = st.sampled_from(["task_easy", "task_medium", "task_hard"])


@st.composite
def action_st(draw):
    return Action(
        classify_ticket=draw(classify_st),
        priority=draw(priority_st),
        response_action=draw(response_st),
    )


# ---------------------------------------------------------------------------
# Property 1: Reward score is always in [0.0, 1.0]
# Feature: customer-support-ticket-env, Property 1: reward range
# ---------------------------------------------------------------------------

@given(task_name_st, st.lists(action_st(), min_size=1, max_size=6))
@settings(max_examples=20)
def test_reward_always_in_range(task_name: str, actions: list[Action]):
    """For any task and any sequence of actions, all rewards are in [0.0, 1.0]."""
    env = CustomerSupportEnv()
    env.reset(task_name)
    for action in actions:
        if env._done:
            break
        _, reward, _ = env.step(action)
        assert 0.0 <= reward.score <= 1.0, f"Reward out of range: {reward.score}"


# ---------------------------------------------------------------------------
# Property 3: Reset clears all episode state
# Feature: customer-support-ticket-env, Property 3: reset clears state
# ---------------------------------------------------------------------------

@given(task_name_st, st.lists(action_st(), min_size=0, max_size=4), task_name_st)
@settings(max_examples=20)
def test_reset_clears_state(
    task_name1: str, actions: list[Action], task_name2: str
):
    """After any sequence of steps, reset() produces a clean state."""
    env = CustomerSupportEnv()
    env.reset(task_name1)
    for action in actions:
        if env._done:
            break
        env.step(action)

    # Now reset to a (possibly different) task
    env.reset(task_name2)
    state = env.state()

    assert state["step_count"] == 0
    assert state["action_history"] == []
    assert state["done"] is False
    assert state["total_reward"] == 0.0
    assert state["rewards_per_step"] == []


# ---------------------------------------------------------------------------
# Property 8: Loop penalty applies after 3 identical actions
# Feature: customer-support-ticket-env, Property 8: loop penalty
# ---------------------------------------------------------------------------

def test_loop_penalty_applied():
    """Submitting the same action 3 times in a row triggers loop penalty."""
    env = CustomerSupportEnv()
    env.reset("task_medium")

    # Use an action that is wrong so we can isolate the loop penalty
    action = Action(
        classify_ticket="general",
        priority="low",
        response_action="ignore",
    )

    # Step 1 and 2 — no loop penalty yet
    _, r1, _ = env.step(action)
    assert r1.breakdown.get("loop_penalty", 0.0) == 0.0

    _, r2, _ = env.step(action)
    assert r2.breakdown.get("loop_penalty", 0.0) == 0.0

    # Step 3 — loop penalty should apply
    _, r3, _ = env.step(action)
    assert r3.breakdown.get("loop_penalty", 0.0) == -0.2


# ---------------------------------------------------------------------------
# Property 9: step() before reset() raises RuntimeError
# Feature: customer-support-ticket-env, Property 9: step before reset
# ---------------------------------------------------------------------------

def test_step_before_reset_raises():
    """Calling step() on a fresh environment raises RuntimeError."""
    env = CustomerSupportEnv()
    action = Action(
        classify_ticket="billing",
        priority="medium",
        response_action="reply",
    )
    with pytest.raises(RuntimeError, match="not initialized"):
        env.step(action)


# ---------------------------------------------------------------------------
# Property 10: done flag is sticky
# Feature: customer-support-ticket-env, Property 10: done flag is sticky
# ---------------------------------------------------------------------------

def test_step_after_done_raises():
    """Calling step() after episode is done raises RuntimeError."""
    env = CustomerSupportEnv()
    env.reset("task_easy")
    action = Action(
        classify_ticket="billing",
        priority="medium",
        response_action="reply",
    )
    _, _, done = env.step(action)
    assert done is True

    with pytest.raises(RuntimeError, match="Episode is done"):
        env.step(action)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_unknown_task_raises():
    env = CustomerSupportEnv()
    with pytest.raises(ValueError, match="Unknown task"):
        env.reset("nonexistent")


def test_reset_returns_observation():
    from env.models import Observation
    env = CustomerSupportEnv()
    obs = env.reset("task_easy")
    assert isinstance(obs, Observation)


def test_step_returns_correct_types():
    from env.models import Observation, Reward
    env = CustomerSupportEnv()
    env.reset("task_easy")
    action = Action(
        classify_ticket="billing",
        priority="medium",
        response_action="reply",
    )
    obs, reward, done = env.step(action)
    assert isinstance(obs, Observation)
    assert isinstance(reward, Reward)
    assert isinstance(done, bool)


def test_state_is_json_serializable():
    import json
    env = CustomerSupportEnv()
    env.reset("task_easy")
    state = env.state()
    # Should not raise
    json.dumps(state)


def test_correct_action_gives_high_reward():
    """A perfectly correct action on task_easy should give reward >= 0.9."""
    env = CustomerSupportEnv()
    env.reset("task_easy")
    # task_easy expected: billing, medium, reply
    action = Action(
        classify_ticket="billing",
        priority="medium",
        response_action="reply",
    )
    _, reward, _ = env.step(action)
    assert reward.score >= 0.9


def test_step_count_increments():
    env = CustomerSupportEnv()
    env.reset("task_medium")
    action = Action(
        classify_ticket="technical",
        priority="high",
        response_action="request_info",
    )
    env.step(action)
    assert env.state()["step_count"] == 1
    env.step(action)
    assert env.state()["step_count"] == 2


def test_all_tasks_run_to_completion():
    """All three tasks can be run to completion without errors."""
    for task_name in ["task_easy", "task_medium", "task_hard"]:
        env = CustomerSupportEnv()
        env.reset(task_name)
        action = Action(
            classify_ticket="general",
            priority="low",
            response_action="reply",
        )
        done = False
        steps = 0
        while not done and steps < 10:
            _, _, done = env.step(action)
            steps += 1
        assert done
