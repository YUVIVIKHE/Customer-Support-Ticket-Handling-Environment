# Customer Support Ticket Handling Environment

An [OpenEnv](https://openenv.dev)-compliant reinforcement learning environment that simulates a customer support agent handling incoming tickets.

## What is this?

Real support teams receive hundreds of tickets daily. Agents must quickly classify each ticket, assign the right priority, and choose the best response action. This environment lets you train and evaluate LLMs on exactly that task — with deterministic grading and three difficulty levels.

**Real-world usefulness**: Benchmarks how well an LLM can act as a first-line support agent, useful for building automated triage systems, evaluating instruction-following, and testing multi-step reasoning.

---

## Observation Space

| Field | Type | Values |
|---|---|---|
| `ticket_text` | string | Raw ticket text from the customer |
| `customer_type` | enum | `premium`, `normal` |
| `urgency_level` | enum | `low`, `medium`, `high` |
| `previous_actions` | list[string] | JSON-serialized actions taken so far |

## Action Space

| Field | Type | Values |
|---|---|---|
| `classify_ticket` | enum | `billing`, `technical`, `complaint`, `general` |
| `priority` | enum | `low`, `medium`, `high` |
| `response_action` | enum | `reply`, `escalate`, `ignore`, `request_info` |

## Reward

Rewards are in **[0.0, 1.0]** and computed per step:

| Component | Value |
|---|---|
| Correct `classify_ticket` | +0.3 |
| Correct `priority` | +0.3 |
| Correct `response_action` | +0.3 |
| Fast resolution (within expected steps) | +0.1 |
| Wrong `response_action` | -0.2 |
| Loop penalty (3+ identical consecutive actions) | -0.2 |

---

## Tasks

### task_easy (Easy)
Single-step scenario. The agent receives a simple refund request and must correctly classify it as `billing` and respond with `reply`.

**Grader**: Exact match → 1.0 | Correct classification only → 0.5 | Wrong → 0.0

**Baseline score**: ~0.7 (GPT-4o-mini with no fine-tuning)

---

### task_medium (Medium)
Multi-step internet outage ticket. The agent must handle the ticket across 3 steps:
1. `request_info` — ask for more details
2. `reply` — provide a solution
3. `reply` — confirm resolution

**Grader**: Score = fraction of steps with correct `response_action`

**Baseline score**: ~0.5

---

### task_hard (Hard)
Three-ticket queue. The agent must handle tickets in priority order (high → medium → low) with correct classification and response actions.

**Grader**: 0.5 × prioritization_score + 0.5 × action_score

**Baseline score**: ~0.4

---

## Setup

```bash
cd project
pip install -r requirements.txt
```

## Run the Server

```bash
python server.py
# or
uvicorn server:app --host 0.0.0.0 --port 7860
```

The server runs on port **7860** (Hugging Face Spaces compatible).

### API Endpoints

```bash
# Start a new episode
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name": "task_easy"}'

# Take a step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"classify_ticket": "billing", "priority": "medium", "response_action": "reply"}'

# Inspect state
curl http://localhost:7860/state
```

## Run Inference

Set the required environment variables and run:

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="your-token-here"

python -m env.inference
```

**Log output example**:
```
[START] task=task_easy env=customer-support-ticket-env model=gpt-4o-mini
[STEP] step=1 action={"classify_ticket":"billing","priority":"medium","response_action":"reply"} reward=1.00 done=true error=null
[END] success=true steps=1 score=1.00 rewards=1.00
```

## Run with Docker

```bash
# Build
docker build -t customer-support-env .

# Run
docker run -p 7860:7860 customer-support-env

# Run inference inside container
docker run --rm \
  -e API_BASE_URL="https://api.openai.com/v1" \
  -e MODEL_NAME="gpt-4o-mini" \
  -e HF_TOKEN="your-token" \
  customer-support-env \
  python -m env.inference
```

## Run Tests

```bash
cd project
pytest tests/ -v
```

---

## Project Structure

```
project/
├── env/
│   ├── __init__.py
│   ├── models.py        # Pydantic v2 data models
│   ├── environment.py   # Stateful environment (reset/step/state)
│   ├── tasks.py         # Task definitions (easy/medium/hard)
│   ├── graders.py       # Deterministic graders
│   └── inference.py     # LLM evaluation script
├── tests/
│   ├── test_models.py
│   ├── test_tasks.py
│   ├── test_graders.py
│   ├── test_environment.py
│   ├── test_server.py
│   └── test_inference.py
├── server.py            # FastAPI server
├── openenv.yaml         # OpenEnv configuration
├── Dockerfile
├── requirements.txt
└── README.md
```

## Resource Requirements

- CPU: ≤ 2 vCPU
- RAM: ≤ 8 GB
- Inference runtime: < 20 minutes for all three tasks
