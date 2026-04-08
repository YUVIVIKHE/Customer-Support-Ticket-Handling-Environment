"""FastAPI server exposing the Customer Support Ticket Environment over HTTP.

Runs on port 7860 for Hugging Face Spaces compatibility.

Endpoints:
  POST /reset  — start a new episode, returns initial Observation
  POST /step   — advance one step, returns {observation, reward, done, info}
  GET  /state  — inspect current episode state
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env.environment import CustomerSupportEnv
from env.models import Action, Observation, Reward

app = FastAPI(
    title="Customer Support Ticket Environment",
    description="OpenEnv-compliant RL environment for customer support ticket handling.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single shared environment instance (stateful per-server process)
_env = CustomerSupportEnv()


class ResetRequest(BaseModel):
    task_name: str


class StepResponse(BaseModel):
    """Response from POST /step — matches OpenEnv 4-tuple convention."""
    observation: Observation
    reward: float          # scalar score in [0.0, 1.0]
    done: bool
    info: dict             # reward_breakdown + step_count


@app.post("/reset", response_model=Observation)
async def reset(body: ResetRequest) -> Observation:
    """Initialize a new episode for the given task.

    Returns the initial Observation (HTTP 200) or 400 on unknown task.
    """
    try:
        obs = _env.reset(body.task_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return obs


@app.post("/step", response_model=StepResponse)
async def step(action: Action) -> StepResponse:
    """Advance the episode by one step.

    Returns observation, scalar reward, done flag, and info dict.
    """
    try:
        obs, reward, done, info = _env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return StepResponse(
        observation=obs,
        reward=reward.score,   # expose scalar for easy consumption
        done=done,
        info=info,
    )


@app.get("/state")
async def state() -> dict:
    """Return the current episode state as a JSON object."""
    return _env.state()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
