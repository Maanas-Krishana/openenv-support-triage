---
title: OpenEnv Customer Support Triage
emoji: 🛠️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: false
---

# OpenEnv: Customer Support Triage

This is an **OpenEnv** compliant environment that simulates a real-world task: **Customer Support Ticket Triage**.

## Description & Motivation

Customer support is a universal task for LLMs. This environment tests an agent's ability to read an incoming support ticket, extract meaning or intent, and take action. Taking action is non-trivial and often involves routing to specialized departments, responding immediately to resolve the ticket if criteria (e.g., apology and refund requirements) are met, or escalating in the event of complex/angry bugs. This goes beyond simple NLP classification—it is a functional step-wise environment modeling real systems.

## Spaces

### Observation Space
The agent receives a `CustomerSupportObservation` Pydantic model with:
- `ticket_id` (str): Unique ticket identifier.
- `subject` (str): Email/Ticket subject line.
- `message` (str): The body of the ticket from the user.
- `department` (str): The current department the ticket is sitting in.
- `status` (str): "Open" or "Resolved".
- `reward` (float): The step's reward.
- `done` (bool): If the task is completed for the episode.

### Action Space
The agent takes a `CustomerSupportAction` Pydantic model:
- `action_type` (Literal["assign", "reply_and_resolve", "escalate"]): The core intent.
- `department` (Optional[Literal["IT", "Billing", "Sales", "General"]]): Required if assigning.
- `reply_message` (Optional[str]): Required if replying to resolve.

## Tasks & Difficulty

The environment cycles through 3 primary tasks representing different difficulties:
1. **Easy (`TKT-100`)**: A straightforward issue regarding login issues. Reward depends on simply routing it to `IT`.
2. **Medium (`TKT-200`)**: A user double-charged and requesting a refund and apology. The agent must `reply_and_resolve` and the `reply_message` must explicitly contain apology terms and a refund acknowledgment to get the full reward. Partial reward is granted for identifying one or routing to Billing.
3. **Hard (`TKT-300`)**: An angry bug report causing app freezing. While an agent might guess to just resolve it or route it, the correct path is to `escalate` directly.

## Usage & Execution

### Running the Evaluation Script
Using OpenAI Client, testing the Qwen model. Make sure `OPENAI_API_KEY` (or `HF_TOKEN`) is set:
```bash
python inference.py
```

### Docker
```bash
docker build -t openenv-support .
docker run -p 8000:8000 openenv-support
```

### Validation
To test compliance with the OpenEnv spec:
```bash
openenv validate
```
