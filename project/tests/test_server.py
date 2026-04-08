"""Unit tests for server.py FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server import app

client = TestClient(app)


def test_reset_returns_observation():
    resp = client.post("/reset", json={"task_name": "task_easy"})
    assert resp.status_code == 200
    data = resp.json()
    assert "ticket_text" in data
    assert "customer_type" in data
    assert "urgency_level" in data
    assert "previous_actions" in data


def test_reset_unknown_task_returns_400():
    resp = client.post("/reset", json={"task_name": "nonexistent"})
    assert resp.status_code == 400


def test_step_before_reset_returns_400():
    # Use a fresh app instance to ensure no prior reset
    from fastapi.testclient import TestClient
    from env.environment import CustomerSupportEnv
    import server as srv

    # Reset the shared env to uninitialized state
    srv._env = CustomerSupportEnv()

    resp = client.post("/step", json={
        "classify_ticket": "billing",
        "priority": "medium",
        "response_action": "reply",
    })
    assert resp.status_code == 400


def test_step_returns_correct_shape():
    client.post("/reset", json={"task_name": "task_easy"})
    resp = client.post("/step", json={
        "classify_ticket": "billing",
        "priority": "medium",
        "response_action": "reply",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "observation" in data
    assert "reward" in data
    assert "done" in data
    assert "score" in data["reward"]
    assert "breakdown" in data["reward"]
    assert "done" in data["reward"]


def test_state_returns_json():
    client.post("/reset", json={"task_name": "task_easy"})
    resp = client.get("/state")
    assert resp.status_code == 200
    data = resp.json()
    assert "step_count" in data
    assert "done" in data
    assert "initialized" in data


def test_full_easy_episode():
    """Run a complete task_easy episode via HTTP."""
    client.post("/reset", json={"task_name": "task_easy"})
    resp = client.post("/step", json={
        "classify_ticket": "billing",
        "priority": "medium",
        "response_action": "reply",
    })
    assert resp.status_code == 200
    assert resp.json()["done"] is True
