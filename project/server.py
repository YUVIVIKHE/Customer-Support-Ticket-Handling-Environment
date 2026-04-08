"""FastAPI server exposing the Customer Support Ticket Environment over HTTP.

Runs on port 7860 for Hugging Face Spaces compatibility.

Endpoints:
  POST /reset  — start a new episode
  POST /step   — advance the episode by one step
  GET  /state  — inspect current episode state
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root to path so `env` package is importable
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

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

# Single shared environment instance (stateful per-server)
_env = CustomerSupportEnv()


class ResetRequest(BaseModel):
    task_name: str


class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool


@app.post("/reset", response_model=Observation)
async def reset(body: ResetRequest) -> Observation:
    """Initialize a new episode for the given task."""
    try:
        obs = _env.reset(body.task_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return obs


@app.post("/step", response_model=StepResponse)
async def step(action: Action) -> StepResponse:
    """Advance the episode by one step."""
    try:
        obs, reward, done = _env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return StepResponse(observation=obs, reward=reward, done=done)


@app.get("/state")
async def state() -> dict:
    """Return the current episode state."""
    return _env.state()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
