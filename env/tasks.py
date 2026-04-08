"""Task definitions for the Customer Support Ticket Environment.

All tasks are fully deterministic — no randomness is used.
Ticket texts use realistic, noisy real-world language.
"""

from dataclasses import dataclass

from env.models import Action, Observation


@dataclass
class TaskDefinition:
    """Describes a single evaluation task."""

    name: str
    difficulty: str                 # "easy" | "medium" | "hard"
    initial_observation: Observation
    expected_actions: list[Action]  # ground truth per step
    expected_steps: int             # steps threshold for speed bonus
    max_steps: int                  # hard episode limit


# ---------------------------------------------------------------------------
# Task Easy: single-step refund ticket (noisy real-world text)
# ---------------------------------------------------------------------------

_TASK_EASY = TaskDefinition(
    name="task_easy",
    difficulty="easy",
    initial_observation=Observation(
        ticket_text="plz refund asap!!! i got charged and never received my order",
        customer_type="normal",
        urgency_level="medium",
        previous_actions=[],
    ),
    expected_actions=[
        # Correct: billing issue → medium priority → reply to customer
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
# Task Medium: multi-step internet outage (noisy real-world text)
# ---------------------------------------------------------------------------

_TASK_MEDIUM = TaskDefinition(
    name="task_medium",
    difficulty="medium",
    initial_observation=Observation(
        ticket_text=(
            "internet not working since morning idk what to do "
            "i tried restarting everything still nothing"
        ),
        customer_type="premium",
        urgency_level="high",
        previous_actions=[],
    ),
    expected_actions=[
        # Step 1: ask for more diagnostic info before acting
        Action(
            classify_ticket="technical",
            priority="high",
            response_action="request_info",
        ),
        # Step 2: provide solution after gathering info
        Action(
            classify_ticket="technical",
            priority="high",
            response_action="reply",
        ),
        # Step 3: confirm resolution
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
# Task Hard: multi-ticket queue with prioritization (noisy real-world text)
# Agent must handle tickets in priority order: high → medium → low
# ---------------------------------------------------------------------------

_TASK_HARD = TaskDefinition(
    name="task_hard",
    difficulty="hard",
    initial_observation=Observation(
        ticket_text=(
            "QUEUE: "
            "[1] 'charged twice ??? fix this NOW' (normal, high) | "
            "[2] 'how do i reset password i forgot it' (normal, low) | "
            "[3] 'app keeps crashing on startup every time' (premium, medium)"
        ),
        customer_type="normal",
        urgency_level="high",
        previous_actions=[],
    ),
    expected_actions=[
        # Ticket 1 first: billing, high urgency → escalate (double charge)
        Action(
            classify_ticket="billing",
            priority="high",
            response_action="escalate",
        ),
        # Ticket 3 second: technical, medium urgency → reply (premium customer)
        Action(
            classify_ticket="technical",
            priority="medium",
            response_action="reply",
        ),
        # Ticket 2 last: general, low urgency → reply
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
