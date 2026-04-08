"""Stateful Customer Support Ticket Environment.

Implements the OpenEnv interface: reset(), step(action), state().
step() returns a 4-tuple: (observation, reward, done, info)
"""

import json
from typing import Optional

from env.models import Action, Observation, Reward
from env.tasks import TaskDefinition, get_task


class CustomerSupportEnv:
    """OpenEnv-compliant environment for customer support ticket handling."""

    ENV_NAME = "customer-support-ticket-env"

    def __init__(self) -> None:
        self._task: Optional[TaskDefinition] = None
        self._step_count: int = 0
        self._action_history: list[Action] = []
        self._total_reward: float = 0.0
        self._rewards_per_step: list[float] = []
        self._done: bool = False
        self._initialized: bool = False

    # ------------------------------------------------------------------
    # Public OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self, task_name: str) -> Observation:
        """Initialize a new episode for the given task.

        Args:
            task_name: One of "task_easy", "task_medium", "task_hard".

        Returns:
            The initial Observation for the task.

        Raises:
            ValueError: If task_name is not recognized.
        """
        self._task = get_task(task_name)  # raises ValueError on unknown task
        self._step_count = 0
        self._action_history = []
        self._total_reward = 0.0
        self._rewards_per_step = []
        self._done = False
        self._initialized = True
        return self._task.initial_observation

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        """Advance the episode by one step.

        Args:
            action: The agent's action for this step.

        Returns:
            (next_observation, reward, done, info)
            info contains reward_breakdown and step_count.

        Raises:
            RuntimeError: If called before reset() or after episode is done.
        """
        if not self._initialized:
            raise RuntimeError(
                "Environment not initialized. Call reset() first."
            )
        if self._done:
            raise RuntimeError(
                "Episode is done. Call reset() to start a new episode."
            )

        assert self._task is not None

        # Determine expected action for this step
        step_idx = self._step_count
        expected = (
            self._task.expected_actions[step_idx]
            if step_idx < len(self._task.expected_actions)
            else self._task.expected_actions[-1]
        )

        # Compute structured reward
        reward = self._compute_reward(action, expected)

        # Record action
        self._action_history.append(action)
        self._step_count += 1
        self._total_reward += reward.score
        self._rewards_per_step.append(reward.score)

        # Determine done
        done = (
            self._step_count >= len(self._task.expected_actions)
            or self._step_count >= self._task.max_steps
        )
        self._done = done

        # Build next observation (append serialized action to previous_actions)
        next_obs = Observation(
            ticket_text=self._task.initial_observation.ticket_text,
            customer_type=self._task.initial_observation.customer_type,
            urgency_level=self._task.initial_observation.urgency_level,
            previous_actions=[
                a.model_dump_json() for a in self._action_history
            ],
        )

        # info dict — always a plain JSON-serializable dict
        info: dict = {
            "reward_breakdown": reward.breakdown,
            "step_count": self._step_count,
        }

        return next_obs, reward, done, info

    def state(self) -> dict:
        """Return a JSON-serializable snapshot of the current episode state."""
        return {
            "task_name": self._task.name if self._task else None,
            "step_count": self._step_count,
            "total_reward": self._total_reward,
            "rewards_per_step": self._rewards_per_step,
            "done": self._done,
            "initialized": self._initialized,
            "action_history": [
                json.loads(a.model_dump_json()) for a in self._action_history
            ],
        }

    # ------------------------------------------------------------------
    # Internal reward computation
    # ------------------------------------------------------------------

    def _compute_reward(self, action: Action, expected: Action) -> Reward:
        """Compute per-step reward with full breakdown.

        Components:
          classify  +0.3  correct classify_ticket
          priority  +0.3  correct priority
          response  +0.3  correct response_action
          speed_bonus +0.1  step within expected_steps threshold
          penalty   -0.2  wrong response_action
          penalty   -0.2  loop (last 3 actions identical)
        Final score clamped to [0.0, 1.0].
        """
        assert self._task is not None
        breakdown: dict[str, float] = {}

        classify_ok = action.classify_ticket == expected.classify_ticket
        priority_ok = action.priority == expected.priority
        response_ok = action.response_action == expected.response_action

        # Core correctness components
        breakdown["classify"] = 0.3 if classify_ok else 0.0
        breakdown["priority"] = 0.3 if priority_ok else 0.0
        breakdown["response"] = 0.3 if response_ok else 0.0

        # Speed bonus: reward early correct decisions
        fast = self._step_count < self._task.expected_steps
        breakdown["speed_bonus"] = 0.1 if fast else 0.0

        # Penalty: wrong response action
        breakdown["penalty"] = -0.2 if not response_ok else 0.0

        # Loop penalty: last 3 actions (including current) all identical
        recent = self._action_history[-2:] + [action]
        if (
            len(recent) == 3
            and len(set(a.model_dump_json() for a in recent)) == 1
        ):
            breakdown["penalty"] = min(breakdown["penalty"] - 0.2, -0.2)

        raw_score = sum(breakdown.values())
        clamped = max(0.0, min(1.0, raw_score))

        return Reward(score=clamped, breakdown=breakdown)
