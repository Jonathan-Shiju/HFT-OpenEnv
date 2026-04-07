from collections import defaultdict
from dataclasses import asdict, dataclass
import websockets
import httpx
import argparse, asyncio, json, os, sys
from dotenv import load_dotenv
from openai import AsyncOpenAI
import re
from typing import Any, Awaitable, Callable, Dict, Optional


try:
    try:
        from models import HftAction, HftObservation, tasks
    except ModuleNotFoundError:
        from hft.models import HftAction, HftObservation, tasks

    from openenv.core.env_server.http_server import (
        WSErrorResponse,
        WSObservationResponse,
    )
    from openenv.core.client_types import StepResult

except Exception as e:  # pragma: no cover
    raise ImportError(
        "Required modules not found. Ensure you have installed dependencies with '\n    uv sync\n'"
    ) from e


load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
BENCHMARK = os.getenv("BENCHMARK", "hft")
TASK_NAME = os.getenv("TASK", "flash_crash")
VERBOSE = os.getenv("VERBOSE", "false")


_transcript = ""
_rewards = defaultdict(list)
_rewards_avg = defaultdict(list)

SYSTEM_PROMPT = """
**Role:** You are an expert High-Frequency Trading (HFT) Market Making agent. Your objective is to manage a 1,000-unit trade execution task within a Limit Order Book (LOB) simulation. You must maximize a reward function that balances Execution PnL, Passive Fill Incentives, and Execution Urgency.

**Core Mechanics:**

* **Inventory Management (inv):** You must avoid holding large positions (Long or Short) for too long. Large inventory triggers a risk penalty.
* **Urgency (remaining):** You have a target of 1,000 units. If you are behind the linear execution pace (TWAP), your reward decreases.
* **Passive vs. Active:** You receive a bonus for "Passive" fills (resting limit orders) and a penalty for "Active" fills (hitting the spread). Use Active fills only to neutralize dangerous inventory.
* **The Spread:** You should typically quote at the Best Bid and Best Ask to "capture the spread."

**Action Schema:**
You must return a JSON list of actions or `null` to skip a turn.

* **Limit Order:** `{"type": "limit", "side": "buy"|"ask", "price": float, "size": int}`
* **Market Order:** `{"type": "market", "side": "buy"|"ask", "size": int}`
* **Cancel Order:** `{"type": "cancel", "order_id": "string"}`

**Note:** Active orders are provided as [id, side, price, size]. Side 'B' is Buy, 'A' is Ask.

**Strategy Guidelines:**
* **Join the Best:** If the spread is wide, "penny-jump" the best bid/ask to get priority.
* **Inventory Skew:** If you are Long (+), lower your Ask price to encourage a fill. If you are Short (-), raise your Bid price to cover.
* **Terminal Liquidation:** As $t$ approaches **1.0**, use Market orders to zero out your inventory to avoid the heavy terminal liquidation penalty.

**CRITICAL CONSTRAINTS:**
1.  **Output Format:** Return **ONLY** a JSON list. Do not include any prose, markdown commentary, or explanations.
2.  **Cardinality:** Your list **MUST NOT** contain more than **one** order of the same type. You are permitted a maximum of one `limit` order, one `market` order, and one `cancel` order per response.
3.  **Idle:** Return `null` only if the volume goal is met and inventory is 0.


**You cannot cancel more than one single order at a time**
"""

VALID_TYPES = {"limit", "market", "cancel"}

REQUIRED_KEYS = {
    "limit": {"type", "side", "price", "size"},
    "market": {"type", "side", "size"},
    "cancel": {"type", "order_id"},
}

VALID_SIDES = {"buy", "ask"}

temperature = 0.0
top_p = 0.8
max_tokens = 100


@dataclass
class HftParams:
    max_steps: Optional[int] = None
    tick_size: Optional[float] = None
    inventory: Optional[int] = None
    cash: Optional[float] = None
    arrival_price: Optional[float] = None
    target_shares: Optional[int] = None

    def to_env_dict(self) -> Dict[str, str]:
        """Returns a dict of non-None values as strings."""
        return {k.upper(): str(v) for k, v in self.__dict__.items() if v is not None}


class WebSocketError(Exception):
    pass


def log_transcript(message: str) -> None:
    global _transcript
    _transcript += f"{message}\n"
    print(message, file=sys.stderr, flush=True)


def log_start(task: str, benchmark: str, model: str) -> None:
    print(f"[START] task={task} env={benchmark} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error=null",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_value = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_value}",
        flush=True,
    )


def validate_task_name(task: str) -> None:
    if task not in tasks:
        log_transcript(f"Invalid task name: {task}. Enter one of: {tasks}")
        sys.exit(1)


async def call_llm_for_actions(
    client: AsyncOpenAI, user_prompt: str
) -> Optional[list[dict]]:
    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    llm_reply = response.choices[0].message.content
    actions = extract_actions(llm_reply)
    log_transcript(f"LLM response: {llm_reply}")
    log_transcript(f"Parsed actions: {actions}")
    return actions


def actions_to_hft_action(actions: Optional[list[dict]]) -> HftAction:
    limit_buy_price = None
    limit_buy_size = None
    limit_ask_price = None
    limit_ask_size = None
    market_buy_size = None
    market_ask_size = None
    cancel_order_id = None

    if actions not in (None, []):
        for action in actions:
            if action["type"] == "limit":
                if action["side"] == "buy":
                    limit_buy_price = action["price"]
                    limit_buy_size = action["size"]
                elif action["side"] == "ask":
                    limit_ask_price = action["price"]
                    limit_ask_size = action["size"]
            elif action["type"] == "market":
                if action["side"] == "buy":
                    market_buy_size = action["size"]
                elif action["side"] == "ask":
                    market_ask_size = action["size"]
            elif action["type"] == "cancel":
                cancel_order_id = action["order_id"]

    return HftAction(
        limit_buy_price=limit_buy_price,
        limit_buy_size=limit_buy_size,
        limit_ask_price=limit_ask_price,
        limit_ask_size=limit_ask_size,
        market_buy_size=market_buy_size,
        market_ask_size=market_ask_size,
        cancel_order_id=cancel_order_id,
    )


def normalize_observation(
    observation: Dict[str, Any],
    **kwargs,
) -> HftObservation:
    return HftObservation(**observation, **kwargs)


def build_user_prompt(
    observation: HftObservation | Dict[str, Any], **kwargs
) -> tuple[HftObservation, str]:
    obs_model = normalize_observation(observation, **kwargs)

    if getattr(obs_model, "history", None):
        mid = obs_model.history[-1].get("mid")
        if mid is not None and hasattr(obs_model, "active_orders"):
            obs_model.active_orders = compress_active_orders(
                obs_model.active_orders, mid
            )

    return obs_model, parse_obs(obs_model)


async def run_episode_loop(
    client: AsyncOpenAI,
    task: str,
    initial_observation: HftObservation | Dict[str, Any],
    initial_done: bool,
    step_fn: Callable[[HftAction], Awaitable[HftObservation | Dict[str, Any]]],
) -> None:
    _rewards[task] = []
    observation, user_prompt = build_user_prompt(initial_observation)
    done = initial_done

    while not done:
        actions = await call_llm_for_actions(client, user_prompt)
        action_payload = actions_to_hft_action(actions)
        observation = await step_fn(action_payload)
        step_reward = None
        step_done = False
        if isinstance(observation, dict):
            step_reward = observation.get("reward")
            step_done = observation.get("done", False)
        else:
            step_reward = getattr(observation, "reward", None)
            step_done = getattr(observation, "done", False)

        log_step(
            step=len(_rewards[task]) + 1,
            action=json.dumps(
                action_payload.model_dump(exclude_none=True), separators=(",", ":")
            ),
            reward=float(step_reward or 0.0),
            done=bool(step_done),
            error=None,
        )
        if not isinstance(observation, dict):
            observation = asdict(observation)["observation"].model_dump()
            observation, user_prompt = build_user_prompt(observation)

        else:
            observation, user_prompt = build_user_prompt(
                observation.get("observation"),
                reward=observation.get("reward"),
                done=observation.get("done"),
            )

        reward = observation.reward
        done = observation.done
        history = observation.history
        _rewards[task].append(reward)

        log_transcript(
            f"Step reward: {reward}, Done: {done}"
            if VERBOSE.lower() == "false"
            else f"Step reward: {reward}, Done: {done}, History: {history}"
        )


def coerce_action(obj: dict) -> Optional[dict]:
    """Validate and coerce a single action dict into a clean, typed action."""
    if not isinstance(obj, dict):
        return None

    # Normalize keys
    obj = {k.strip().lower(): v for k, v in obj.items()}

    action_type = str(obj.get("type", "")).strip().lower()
    if action_type not in VALID_TYPES:
        return None

    required = REQUIRED_KEYS[action_type]
    if not required.issubset(obj.keys()):
        return None

    if action_type == "limit":
        side = str(obj["side"]).strip().lower()
        if side not in VALID_SIDES:
            return None
        return {
            "type": "limit",
            "side": side,
            "price": float(obj["price"]),
            "size": int(obj["size"]),
        }

    if action_type == "market":
        side = str(obj["side"]).strip().lower()
        if side not in VALID_SIDES:
            return None
        return {
            "type": "market",
            "side": side,
            "size": int(obj["size"]),
        }

    if action_type == "cancel":
        return {
            "type": "cancel",
            "order_id": str(obj["order_id"]).strip(),
        }

    return None


def extract_actions(response: str) -> Optional[list[HftAction]]:
    """
    Extract a list of valid LOB actions from any LLM response string.

    Handles:
      - Clean JSON arrays / objects
      - Markdown code fences (```json ... ``` or ``` ... ```)
      - null / None responses (idle signal)
      - Partial prose with embedded JSON fragments
      - Single objects that should be wrapped in a list

    Returns:
      - List of validated action dicts, or
      - None  (idle — volume complete, inventory zero)
    """
    if response is None:
        return None

    text = response.strip()

    if text.lower() in {"null", "none", ""}:
        return None

    text = re.sub(r"```(?:json)?\s*", "", text, flags=re.IGNORECASE).strip()
    text = text.rstrip("`").strip()

    json_object_pattern = re.compile(r"\{[^{}]*\}", re.DOTALL)
    json_array_pattern = re.compile(r"\[.*?\]", re.DOTALL)

    candidates = []

    try:
        parsed = json.loads(text)
        if parsed is None:
            return None
        if isinstance(parsed, list):
            candidates = parsed
        elif isinstance(parsed, dict):
            candidates = [parsed]
    except json.JSONDecodeError:
        pass

    if not candidates:
        for match in json_array_pattern.finditer(text):
            try:
                parsed = json.loads(match.group())
                if isinstance(parsed, list):
                    candidates = parsed
                    break
            except json.JSONDecodeError:
                continue

    if not candidates:
        for match in json_object_pattern.finditer(text):
            try:
                obj = json.loads(match.group())
                if isinstance(obj, dict):
                    candidates.append(obj)
            except json.JSONDecodeError:
                continue

    actions = []
    for item in candidates:
        action = coerce_action(item)
        if action is not None:
            actions.append(action)

    if not actions:
        return None

    return actions


def parse_obs(obs: HftObservation | Dict[str, Any]) -> str:
    """Convert HftObservation to a clean, JSON-serializable dict for LLM input."""
    if isinstance(obs, HftObservation):
        obs_dict = obs.model_dump()
    elif isinstance(obs, dict):
        obs_dict = obs
    else:
        raise ValueError("Observation must be HftObservation or dict")

    return json.dumps(obs_dict)


def compress_active_orders(orders, mid_price, tick_window=0.10):
    """
    Filters orders within a price window and flattens them.
    Window of 0.10 means +/- 10 ticks from mid for a 0.01 tick size.
    """
    compressed = []
    for o in orders:
        # 1. Skip orders with no ID
        if o["id"] is None:
            continue

        # 2. Filter by distance to mid-price
        if abs(o["price"] - mid_price) <= tick_window:
            # 3. Use 1-character codes for side: Buy=B, Ask=A
            side_code = "B" if o["side"] == "buy" else "A"
            compressed.append(
                [o["id"], side_code, round(o["price"], 2), int(o["size"])]
            )

    return compressed


async def health_check(http_url: str) -> bool:
    async with httpx.AsyncClient(base_url=http_url, timeout=60) as client:
        try:
            response = await client.get(f"{http_url}/health")
            return response.status_code == 200
        except Exception as e:
            print(f"Health check failed: {e}", file=sys.stderr)
            return False


async def run_docker_task(
    tasks: list, pause: int, hftParams: HftParams, client: AsyncOpenAI
):
    from client import HftEnv

    try:
        for task in tasks:
            env = None
            episode_score = 0.0
            episode_steps = 0
            episode_success = False
            validate_task_name(task)
            log_start(task=task, benchmark=BENCHMARK, model=MODEL_NAME)
            try:
                env = await HftEnv.from_docker_image(
                    LOCAL_IMAGE_NAME,
                    env_vars={
                        "TASK": task,
                        **hftParams.to_env_dict(),
                    },
                )
                reset_response: StepResult = await env.reset()
                observation = asdict(reset_response)
                observation = observation["observation"].model_dump()
                log_transcript(f"Initial observation: {observation}")

                def docker_step(action_payload: HftAction) -> HftObservation:
                    return env.step(action_payload)

                await run_episode_loop(
                    client=client,
                    task=task,
                    initial_observation=observation,
                    initial_done=reset_response.done,
                    step_fn=docker_step,
                )

                _rewards_avg[task] = (
                    sum(_rewards[task]) / len(_rewards[task]) if _rewards[task] else 0
                )
                log_transcript(
                    f"Episode finished for task: {task}, Reward: {_rewards_avg[task]}"
                )
                episode_score = min(max(_rewards_avg[task], 0.0), 1.0)
                episode_steps = len(_rewards[task])
                episode_success = episode_score > 0.0
                await asyncio.sleep(pause)
            finally:
                try:
                    if env is not None:
                        await env.close()
                finally:
                    if episode_steps == 0:
                        episode_steps = len(_rewards[task])
                    if episode_score == 0.0 and _rewards[task]:
                        episode_score = min(max(_rewards_avg[task], 0.0), 1.0)
                        episode_success = episode_score > 0.0
                    log_end(
                        success=episode_success,
                        steps=episode_steps,
                        score=episode_score,
                        rewards=_rewards[task],
                    )
        log_transcript("All tasks completed.")
    except Exception as e:
        log_transcript(f"Error running Docker task: {e}")
        sys.exit(1)


async def check_ws_error(
    response: WSObservationResponse | WSErrorResponse,
) -> Optional[str]:
    if response.get("type") == "error":
        raise WebSocketError(
            response.get("message", "Unknown WebSocket error")
            + f" (code: {response.get('code', 'N/A')})"
            + f" errors: {response.get('errors', '')}"
        )
    return None


async def run_online_single_task(
    client: AsyncOpenAI, task: str, params: HftParams, base_url: str
):
    episode_score = 0.0
    episode_steps = 0
    episode_success = False
    async with websockets.connect(
        uri=f"wss://{base_url.lstrip('https://')}/ws"
    ) as websocket:
        try:
            await websocket.send(
                json.dumps(
                    {
                        "type": "reset",
                        "data": {
                            "task_name": task,
                            "max_steps": params.max_steps,
                            "tick_size": params.tick_size,
                            "inventory": params.inventory,
                            "cash": params.cash,
                            "arrival_price": params.arrival_price,
                            "target_shares": params.target_shares,
                        },
                    }
                )
            )
            reset_response: WSObservationResponse | WSErrorResponse = json.loads(
                await websocket.recv()
            )
            await check_ws_error(reset_response)
            observation = reset_response.get("data")
            done = observation.get("done", False)
            log_transcript(f"Initial observation: {observation}")

            async def ws_step(action_payload: HftAction) -> Dict[str, Any]:
                await websocket.send(
                    json.dumps(
                        {
                            "type": "step",
                            "data": {
                                **action_payload.model_dump(),
                            },
                        }
                    )
                )

                step_response: WSObservationResponse | WSErrorResponse = json.loads(
                    await websocket.recv()
                )
                await check_ws_error(step_response)
                return step_response.get("data", {})

            observation = observation.get("observation", {})

            await run_episode_loop(
                client=client,
                task=task,
                initial_observation=observation,
                initial_done=done,
                step_fn=ws_step,
            )

            await websocket.send(json.dumps({"type": "close", "data": {}}))
            _rewards_avg[task] = (
                sum(_rewards[task]) / len(_rewards[task]) if _rewards[task] else 0
            )
            log_transcript(
                f"Episode finished for task: {task}, Reward: {_rewards_avg[task]}"
            )
            episode_score = min(max(_rewards_avg[task], 0.0), 1.0)
            episode_steps = len(_rewards[task])
            episode_success = episode_score > 0.0
        except WebSocketError as e:
            log_transcript(f"WebSocket error: {e}")
            sys.exit(1)
        except Exception as e:
            log_transcript(f"Error during WebSocket interaction: {e}")
            sys.exit(1)
        finally:
            if episode_steps == 0:
                episode_steps = len(_rewards[task])
            if episode_score == 0.0 and _rewards[task]:
                episode_score = min(max(_rewards_avg[task], 0.0), 1.0)
                episode_success = episode_score > 0.0
            log_end(
                success=episode_success,
                steps=episode_steps,
                score=episode_score,
                rewards=_rewards[task],
            )


async def run_online_tasks(
    client: AsyncOpenAI, tasks: list, pause: int, params: HftParams, base_url: str
):
    for task in tasks:
        validate_task_name(task)
        log_start(task=task, benchmark=BENCHMARK, model=MODEL_NAME)
        await run_online_single_task(client, task, params, base_url)
        score = min(max(_rewards_avg[task], 0.0), 1.0)
        log_end(
            success=score > 0.0,
            steps=len(_rewards[task]),
            score=score,
            rewards=_rewards[task],
        )
        await asyncio.sleep(pause)


async def run_task(
    client: AsyncOpenAI,
    tasks: list,
    pause: int,
    args: argparse.Namespace,
    base_url: str,
):
    hftParams = HftParams(
        max_steps=args.max_steps,
        tick_size=args.tick_size,
        inventory=args.inventory,
        cash=args.cash,
        arrival_price=args.arrival_price,
        target_shares=args.target_shares,
    )
    log_transcript(
        "Using parameters - "
        f"max_steps: {hftParams.max_steps}, tick_size: {hftParams.tick_size}, "
        f"inventory: {hftParams.inventory}, cash: {hftParams.cash}, "
        f"arrival_price: {hftParams.arrival_price}, target_shares: {hftParams.target_shares}"
    )
    use_docker = not await health_check(base_url)
    if use_docker:
        log_transcript(
            "HFSpace server is not healthy. Diverting to docker based environment."
        )
        await run_docker_task(tasks, pause, hftParams, client)
    else:
        log_transcript("HFSpace server is healthy. Running inference against it.")
        await run_online_tasks(client, tasks, pause, hftParams, base_url)
    log_transcript("All tasks completed.")


async def main():
    parser = argparse.ArgumentParser(
        description="Run inference against the Hft environment using OpenAI API."
    )
    parser.add_argument(
        "--task",
        type=str,
        nargs="+",
        default=["all"],
        help="Task name to run (basic_execution, false_signal, conflicting_signal, flash_crash, or all)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="https://jonathanshiju12-hft-env.hf.space",
        help="Base URL for the Hft API (default: https://jonathanshiju12-hft-env.hf.space)",
    )
    parser.add_argument(
        "--pause",
        type=int,
        default=3,
        help="Seconds to pause between steps (default: 3)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum steps to run for each task ",
    )
    parser.add_argument(
        "--tick-size",
        type=float,
        default=None,
        help="Tick size for the market simulation",
    )
    parser.add_argument(
        "--inventory",
        type=int,
        default=None,
        help="Starting inventory for the agent",
    )
    parser.add_argument(
        "--cash", type=float, default=None, help="Starting cash for the agent"
    )
    parser.add_argument(
        "--arrival-price",
        type=float,
        default=None,
        help="Arrival price for the market simulation",
    )
    parser.add_argument(
        "--target-shares",
        type=int,
        default=None,
        help="Target shares to execute for the agent",
    )
    args = parser.parse_args()
    tasks_to_run = (
        args.task
        if "all" not in args.task
        else ["basic_execution", "false_signal", "conflicting_signal", "flash_crash"]
    )

    if not HF_TOKEN and not API_BASE_URL:
        raise ValueError("HF_TOKEN and API_BASE_URL must be set when VERBOSE is True")

    try:
        client = AsyncOpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}", file=sys.stderr)
        sys.exit(1)

    await run_task(client, tasks_to_run, args.pause, args, args.base_url)

    with open("transcript.txt", "w") as f:
        f.write(_transcript)

    with open("baseline_scores.json", "w") as f:
        baseline_scores = {
            "avg": _rewards_avg,
            "all": _rewards,
        }
        json.dump(baseline_scores, f, indent=4)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        import traceback

        error_msg = f"CRITICAL FAILURE: {str(e)}"
        print("=" * 40)
        print(error_msg)
        print("TRACEBACK:")
        traceback.print_exc()
        print("=" * 40)

        with open("error_log.txt", "w") as f:
            f.write(error_msg + "\n")
            traceback.print_factory(file=f)

        sys.exit(1)
