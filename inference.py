import asyncio
import os
import textwrap
import json
import inspect
from typing import List, Optional

from openai import OpenAI

# Removed openenv client import for raw fallback test

from models import CustomerSupportAction
from server.customer_support_environment import CustomerSupportEnvironment

IMAGE_NAME = os.getenv("IMAGE_NAME") # If you are using docker image 
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("CUSTOMER_SUPPORT_TASK", "triage")
BENCHMARK = os.getenv("CUSTOMER_SUPPORT_BENCHMARK", "customer_support_env")
MAX_STEPS = 5
TEMPERATURE = 0.5
MAX_TOKENS = 150
SUCCESS_SCORE_THRESHOLD = 0.8  # normalized score in [0, 1]

# This environment has 3 tasks run sequentially. Each gives up to 1.0 reward.
MAX_TOTAL_REWARD = 1.0

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a Customer Support Triage AI in a complex Multi-Turn environment.
    Your task is to review an incoming ticket and take one of four actions to progress or resolve the ticket.
    
    Valid action formats (JSON):
    {"action_type": "search_knowledge_base", "search_query": "refund policy"}
    {"action_type": "assign", "department": "IT"} 
    {"action_type": "reply_and_resolve", "reply_message": "Sorry about that! I have processed your refund."}
    {"action_type": "escalate"}
    
    Important: If the customer replies negatively after you try to resolve, you must try a different approach or escalate.
    Output ONLY valid JSON representing the action. No markdown formatting, no code blocks (`), just the JSON.
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    # Remove newlines for the stdout line rule
    clean_action = action.replace("\n", " ").replace("\r", "")
    print(
        f"[STEP] step={step} action={clean_action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def build_user_prompt(step: int, observation: dict, last_reward: float, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    obs_str = json.dumps(observation, indent=2)
    return textwrap.dedent(
        f"""
        Step: {step}
        Last observation:
        {obs_str}
        Last reward: {last_reward:.2f}
        Previous steps:
        {history_block}
        Decide your next action (JSON).
        """
    ).strip()


def get_model_action(client: OpenAI, step: int, observation_dict: dict, last_reward: float, history: List[str]) -> CustomerSupportAction:
    user_prompt = build_user_prompt(step, observation_dict, last_reward, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        # Clean potential markdown and parse JSON
        if text.startswith("```json"):
            text = text[7:-3].strip()
        elif text.startswith("```"):
            text = text[3:-3].strip()
        data = json.loads(text)
        return CustomerSupportAction(**data), text
    except Exception as exc:
        print(f"[DEBUG] Model request failed or failed parsing: {exc}", flush=True)
        # Fallback to an escalating safe action on parse error
        fallback = CustomerSupportAction(action_type="escalate")
        return fallback, str(fallback.model_dump())


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Let's use the environment directly if image isn't specified (for local testing parity)
    # The user example imported MyEnvV4Env.from_docker_image
    # We will instantiate it directly to avoid needing docker running if testing local
    # Wait, the spec mandates exactly `env.step(...)` and `env.reset(...)`.
    # OpenEnv's client might be an EnvironmentClient wrapped in async
    
    # Use locally built env
    env = CustomerSupportEnvironment()

    # The Environment base has step() non async, but if it is an EnvironmentClient, it has step_async
    # So we'll try to await it, and if it fails, run it synchronously if local
    
    for task_num in range(3):
        # We run 3 episodes (for the 3 tasks)
        history: List[str] = []
        rewards: List[float] = []
        steps_taken = 0
        score = 0.0
        success = False

        log_start(task=f"{TASK_NAME}_{task_num+1}", env=BENCHMARK, model=MODEL_NAME)

        try:
            if inspect.iscoroutinefunction(env.reset):
                result = await env.reset()
            else:
                result = env.reset()
                
            last_reward = 0.0

            for step in range(1, MAX_STEPS + 1):
                # Format observation dict
                obs_dict = {
                    "ticket_id": result.ticket_id,
                    "subject": result.subject,
                    "message": result.message,
                    "department": result.department,
                    "status": result.status
                }
                if hasattr(result, 'system_response') and result.system_response:
                    obs_dict["system_response"] = result.system_response

                action_obj, action_str = get_model_action(client, step, obs_dict, last_reward, history)

                if inspect.iscoroutinefunction(env.step):
                    result = await env.step(action_obj)
                else:
                    result = env.step(action_obj)

                reward = result.reward or 0.0
                done = result.done
                error = None

                rewards.append(reward)
                steps_taken = step
                last_reward = reward

                log_step(step=step, action=action_str, reward=reward, done=done, error=error)

                history.append(f"Step {step}: {action_str!r} -> reward {reward:+.2f}")

                if done:
                    break

            score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
            score = min(max(score, 0.0), 1.0)  # clamp to [0, 1]
            success = score >= SUCCESS_SCORE_THRESHOLD

        finally:
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
            
    # Cleanup env
    if hasattr(env, 'close'):
        if inspect.iscoroutinefunction(env.close):
            await env.close()
        else:
            env.close()


if __name__ == "__main__":
    asyncio.run(main())
