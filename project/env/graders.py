"""Deterministic graders for each task.

All graders return a float in [0.0, 1.0].
Given the same inputs, a grader always returns the same score.
"""

from env.models import Action


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def grade_easy(action: Action, expected: Action) -> float:
    """Grade a single-step ticket response.

    Returns:
        1.0 — exact match (classify_ticket AND response_action correct)
        0.5 — partial match (classify_ticket correct only)
        0.0 — wrong classification and wrong action
    """
    classify_ok = action.classify_ticket == expected.classify_ticket
    action_ok = action.response_action == expected.response_action

    if classify_ok and action_ok:
        return 1.0
    if classify_ok:
        return 0.5
    return 0.0


def grade_medium(actions: list[Action], expected: list[Action]) -> float:
    """Grade a multi-step ticket sequence.

    Score = (number of steps with correct response_action) / len(expected).
    Returns 0.0 if expected is empty.
    """
    if not expected:
        return 0.0

    n_steps = min(len(actions), len(expected))
    correct = sum(
        1 for i in range(n_steps)
        if actions[i].response_action == expected[i].response_action
    )
    return _clamp(correct / len(expected))


def grade_hard(
    actions: list[Action],
    expected: list[Action],
) -> float:
    """Grade a multi-ticket queue scenario.

    Score = 0.5 * prioritization_score + 0.5 * action_score

    prioritization_score: fraction of actions where priority matches expected
    action_score: fraction of actions where response_action matches expected
    """
    if not expected:
        return 0.0

    n_steps = min(len(actions), len(expected))

    priority_correct = sum(
        1 for i in range(n_steps)
        if actions[i].priority == expected[i].priority
    )
    action_correct = sum(
        1 for i in range(n_steps)
        if actions[i].response_action == expected[i].response_action
    )

    prioritization_score = priority_correct / len(expected)
    action_score = action_correct / len(expected)

    return _clamp(0.5 * prioritization_score + 0.5 * action_score)
