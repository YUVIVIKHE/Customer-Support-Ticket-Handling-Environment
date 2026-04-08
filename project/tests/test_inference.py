"""Unit tests for env/inference.py log format."""

import io
import json
import re
import sys
import os
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from env.inference import run_task, ENV_NAME, MODEL_NAME


def _make_mock_client(action_json: str):
    """Build a mock OpenAI client that always returns action_json."""
    mock_message = MagicMock()
    mock_message.content = action_json

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


VALID_ACTION_JSON = json.dumps({
    "classify_ticket": "billing",
    "priority": "medium",
    "response_action": "reply",
})


def _capture_run_task(client, task_name: str) -> list[str]:
    """Run run_task and capture stdout lines."""
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        run_task(client, task_name)
    finally:
        sys.stdout = old_stdout
    return buf.getvalue().strip().split("\n")


def test_start_log_format():
    """[START] line matches exact format."""
    client = _make_mock_client(VALID_ACTION_JSON)
    lines = _capture_run_task(client, "task_easy")
    start_line = lines[0]
    assert start_line.startswith("[START]")
    assert f"task=task_easy" in start_line
    assert f"env={ENV_NAME}" in start_line
    assert f"model={MODEL_NAME}" in start_line


def test_step_log_format():
    """[STEP] lines match exact format."""
    client = _make_mock_client(VALID_ACTION_JSON)
    lines = _capture_run_task(client, "task_easy")
    step_lines = [l for l in lines if l.startswith("[STEP]")]
    assert len(step_lines) >= 1

    for line in step_lines:
        assert re.search(r"step=\d+", line), f"Missing step= in: {line}"
        assert re.search(r"reward=\d+\.\d{2}", line), f"Missing reward= in: {line}"
        assert re.search(r"done=(true|false)", line), f"Missing done= in: {line}"
        assert re.search(r"error=", line), f"Missing error= in: {line}"


def test_end_log_format():
    """[END] line matches exact format."""
    client = _make_mock_client(VALID_ACTION_JSON)
    lines = _capture_run_task(client, "task_easy")
    end_line = lines[-1]
    assert end_line.startswith("[END]")
    assert re.search(r"success=(true|false)", end_line)
    assert re.search(r"steps=\d+", end_line)
    assert re.search(r"score=\d+\.\d{2}", end_line)
    assert re.search(r"rewards=", end_line)


def test_invalid_action_logs_error():
    """When LLM returns invalid JSON, error is logged in [STEP] line."""
    client = _make_mock_client("not valid json at all")
    lines = _capture_run_task(client, "task_easy")
    step_lines = [l for l in lines if l.startswith("[STEP]")]
    assert len(step_lines) >= 1
    # Error should not be null
    assert "error=null" not in step_lines[0]


def test_all_three_tasks_produce_logs():
    """Running all tasks produces START/END for each."""
    from env.inference import main
    client = _make_mock_client(VALID_ACTION_JSON)

    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        with patch("env.inference.OpenAI", return_value=client):
            main()
    finally:
        sys.stdout = old_stdout

    output = buf.getvalue()
    assert output.count("[START]") == 3
    assert output.count("[END]") == 3
