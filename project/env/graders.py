"""Deterministic graders for each task.

All graders:
  - are fully deterministic (no randomness)
  - return a float in [0.0, 1.0]
  - given the same inputs, always return the same score
"""

from env.models import Action


def _clamp(value: float) -> float:
    """Clamp value to [0.0, 1.0]."""
    return max(0.0, min(1.0, value))


def grade_easy(action: Action, expected: Action) -> float:
    """Grade a single-step ticket response.

    Scoring logic:
      1.0 — exact match: both classify_ticket AND response_action correct
      0.5 — partial match: classify_ticket correct, response_action wrong
      0.0 — wrong classification (response_action irrelevant)

    Args:
        action:   The agent's action.
        expected: The ground-truth action.

    Returns:
        Score in [0.0, 1.0].
    """
    classify_ok = action.classify_ticket == expected.classify_ticket
    action_ok = action.response_action == expected.response_action

    if classify_ok and action_ok:
        return 1.0   # full credit: correct category + correct response
    if classify_ok:
        return 0.5   # partial credit: right category, wrong response
    return 0.0       # no credit: wrong category


def grade_medium(actions: list[Action], expected: list[Action]) -> float:
    """Grade a multi-step ticket sequence.

    Scoring logic:
      Score = (steps with correct response_action) / len(expected)
      Partial credit awarded for each correct step.
      Returns 0.0 if expected is empty.

    Args:
        actions:  List of agent actions (one per step).
        expected: List of ground-truth actions.

    Returns:
        Score in [0.0, 1.0].
    """
    if not expected:
        return 0.0

    n_steps = min(len(actions), len(expected))
    correct = sum(
        1 for i in range(n_steps)
        if actions[i].response_action == expected[i].response_action
    )
    # Divide by total expected steps (not just completed) to penalize short episodes
    return _clamp(correct / len(expected))


def grade_hard(
    actions: list[Action],
    expected: list[Action],
) -> float:
    """Grade a multi-ticket queue scenario.

    Scoring logic:
      Score = 0.5 * prioritization_score + 0.5 * action_score

      prioritization_score: fraction of steps where priority matches expected
        — rewards correct ordering of tickets by urgency
      action_score: fraction of steps where response_action matches expected
        — rewards correct handling of each ticket

      Returns 0.0 if expected is empty.

    Args:
        actions:  List of agent actions (one per ticket in queue).
        expected: List of ground-truth actions in correct priority order.

    Returns:
        Score in [0.0, 1.0].
    """
    if not expected:
        return 0.0

    n_steps = min(len(actions), len(expected))

    # Count correct priorities (measures ordering quality)
    priority_correct = sum(
        1 for i in range(n_steps)
        if actions[i].priority == expected[i].priority
    )

    # Count correct response actions (measures handling quality)
    action_correct = sum(
        1 for i in range(n_steps)
        if actions[i].response_action == expected[i].response_action
    )

    prioritization_score = priority_correct / len(expected)
    action_score = action_correct / len(expected)

    return _clamp(0.5 * prioritization_score + 0.5 * action_score)
