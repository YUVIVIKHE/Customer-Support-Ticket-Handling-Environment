"""Inference script for evaluating LLMs against the Customer Support Ticket Environment.

Reads API_BASE_URL, MODEL_NAME, HF_TOKEN from environment variables.
Runs all three tasks sequentially and prints structured logs.

Log format (EXACT — do not modify):
  [START] task=<task_name> env=<env_name> model=<model_name>
  [STEP] step=<n> action=<action_json> reward=<0.00> done=<true|false> error=<msg|null>
  [END] success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...>
"""

import json
import os
import sys

# Allow running as `python -m env.inference` from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from openai import OpenAI
from pydantic import ValidationError

from env.environment import CustomerSupportEnv
from env.models import Action, Observation, Reward

# ---------------------------------------------------------------------------
# Configuration from environment variables
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN: str = os.environ.get("HF_TOKEN", "")

ENV_NAME = CustomerSupportEnv.ENV_NAME

TASK_NAMES = ["task_easy", "task_medium", "task_hard"]

# Fallback action used when LLM output cannot be parsed
DEFAULT_ACTION = Action(
    classify_ticket="general",
    priority="low",
    response_action="reply",
)

SYSTEM_PROMPT = """You are a customer support agent AI. Given a support ticket observation, output a JSON action.

The action must be a JSON object with exactly these fields:
{
  "classify_ticket": one of ["billing", "technical", "complaint", "general"],
  "priority": one of ["low", "medium", "high"],
  "response_action": one of ["reply", "escalate", "ignore", "request_info"]
}

Respond with ONLY the JSON object, no explanation or markdown.
"""


def build_user_prompt(obs: Observation) -> str:
    """Build the user prompt from the current observation."""
    return f"""Current observation:
{obs.model_dump_json(indent=2)}

Output your action as a JSON object."""


def call_llm(client: OpenAI, obs: Observation) -> tuple[Action, str | None]:
    """Call the LLM and parse the response into an Action.

    Returns:
        (action, error_message) — error_message is None on success.
    """
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(obs)},
            ],
            temperature=0.0,
            max_tokens=200,
        )
        content = response.choices[0].message.content or ""
        # Strip markdown code fences if present
        content = content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1]) if len(lines) > 2 else content
        action = Action.model_validate_json(content)
        return action, None
    except (ValidationError, json.JSONDecodeError, Exception) as e:
        return DEFAULT_ACTION, str(e)


def run_task(client: OpenAI, task_name: str) -> None:
    """Run a single task episode and print structured logs."""
    env = CustomerSupportEnv()
    obs = env.reset(task_name)

    print(f"[START] task={task_name} env={ENV_NAME} model={MODEL_NAME}")

    step_n = 0
    rewards: list[float] = []
    done = False

    while not done:
        step_n += 1
        action, error = call_llm(client, obs)

        # step() now returns 4-tuple: (obs, reward, done, info)
        obs, reward, done, info = env.step(action)

        # Extract scalar score from Reward object
        reward_value = reward.score if isinstance(reward, Reward) else float(reward)
        rewards.append(reward_value)

        error_str = error if error is not None else "null"
        done_str = "true" if done else "false"
        print(
            f"[STEP] step={step_n} action={action.model_dump_json()} "
            f"reward={reward_value:.2f} done={done_str} error={error_str}"
        )

    final_state = env.state()
    total_score = final_state["total_reward"]
    avg_score = total_score / step_n if step_n > 0 else 0.0
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success = avg_score >= 0.5

    print(
        f"[END] success={'true' if success else 'false'} steps={step_n} "
        f"score={avg_score:.2f} rewards={rewards_str}"
    )


def main() -> None:
    """Run all tasks sequentially."""
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "dummy")

    for task_name in TASK_NAMES:
        run_task(client, task_name)


if __name__ == "__main__":
    main()
