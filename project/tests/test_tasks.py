"""Tests for env/tasks.py."""

import pytest
from env.tasks import get_task


TASK_NAMES = ["task_easy", "task_medium", "task_hard"]


# ---------------------------------------------------------------------------
# Property 3 (partial): Tasks are deterministic
# Feature: customer-support-ticket-env, Property 3: tasks are deterministic
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("task_name", TASK_NAMES)
def test_task_deterministic(task_name: str):
    """Calling get_task multiple times returns identical results."""
    t1 = get_task(task_name)
    t2 = get_task(task_name)
    assert t1.initial_observation == t2.initial_observation
    assert t1.expected_actions == t2.expected_actions
    assert t1.expected_steps == t2.expected_steps
    assert t1.max_steps == t2.max_steps


def test_unknown_task_raises():
    with pytest.raises(ValueError, match="Unknown task"):
        get_task("nonexistent_task")


def test_task_easy_single_step():
    t = get_task("task_easy")
    assert len(t.expected_actions) == 1
    assert t.expected_steps == 1


def test_task_medium_three_steps():
    t = get_task("task_medium")
    assert len(t.expected_actions) == 3
    assert t.expected_steps == 3


def test_task_hard_three_tickets():
    t = get_task("task_hard")
    assert len(t.expected_actions) == 3
    # First ticket should be highest priority
    assert t.expected_actions[0].priority == "high"
