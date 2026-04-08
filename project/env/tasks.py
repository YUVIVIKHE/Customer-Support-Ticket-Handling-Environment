"""Task definitions for the Customer Support Ticket Environment.

All tasks are fully deterministic — no randomness is used.
"""

from dataclasses import dataclass, field

from env.models import Action, Observation


@dataclass
class TaskDefinition:
    """Describes a single evaluation task."""

    name: str
    initial_observation: Observation
    expected_actions: list[Action]  # ground truth per step
    expected_steps: int             # steps threshold for fast-resolution bonus
    max_steps: int                  # hard episode limit


# ---------------------------------------------------------------------------
# Task Easy: single-step refund ticket
# ---------------------------------------------------------------------------

_TASK_EASY = TaskDefinition(
    name="task_easy",
    initial_observation=Observation(
        ticket_text="I want a refund for my last order.",
        customer_type="normal",
        urgency_level="medium",
        previous_actions=[],
    ),
    expected_actions=[
        Action(
            classify_ticket="billing",
            priority="medium",
            response_action="reply",
        )
    ],
    expected_steps=1,
    max_steps=3,
)

# ---------------------------------------------------------------------------
# Task Medium: multi-step internet outage ticket
# ---------------------------------------------------------------------------

_TASK_MEDIUM = TaskDefinition(
    name="task_medium",
    initial_observation=Observation(
        ticket_text="My internet is down and I've tried restarting the router.",
        customer_type="premium",
        urgency_level="high",
        previous_actions=[],
    ),
    expected_actions=[
        Action(
            classify_ticket="technical",
            priority="high",
            response_action="request_info",
        ),
        Action(
            classify_ticket="technical",
            priority="high",
            response_action="reply",
        ),
        Action(
            classify_ticket="technical",
            priority="high",
            response_action="reply",
        ),
    ],
    expected_steps=3,
    max_steps=6,
)

# ---------------------------------------------------------------------------
# Task Hard: multi-ticket queue with prioritization
# Each entry in expected_actions corresponds to one ticket in the queue.
# The agent must handle them in priority order: high → medium → low.
# ---------------------------------------------------------------------------

_TASK_HARD = TaskDefinition(
    name="task_hard",
    initial_observation=Observation(
        ticket_text=(
            "QUEUE: [1] 'My account was charged twice.' (normal, high) | "
            "[2] 'How do I reset my password?' (normal, low) | "
            "[3] 'The app crashes on startup.' (premium, medium)"
        ),
        customer_type="normal",
        urgency_level="high",
        previous_actions=[],
    ),
    expected_actions=[
        # Ticket 1: billing, high urgency → escalate
        Action(
            classify_ticket="billing",
            priority="high",
            response_action="escalate",
        ),
        # Ticket 3: technical, medium urgency → reply (premium customer)
        Action(
            classify_ticket="technical",
            priority="medium",
            response_action="reply",
        ),
        # Ticket 2: general, low urgency → reply
        Action(
            classify_ticket="general",
            priority="low",
            response_action="reply",
        ),
    ],
    expected_steps=3,
    max_steps=6,
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_TASKS: dict[str, TaskDefinition] = {
    "task_easy": _TASK_EASY,
    "task_medium": _TASK_MEDIUM,
    "task_hard": _TASK_HARD,
}


def get_task(task_name: str) -> TaskDefinition:
    """Return the TaskDefinition for the given task name.

    Raises:
        ValueError: If task_name is not recognized.
    """
    if task_name not in _TASKS:
        raise ValueError(
            f"Unknown task: '{task_name}'. Valid tasks: {list(_TASKS.keys())}"
        )
    return _TASKS[task_name]
