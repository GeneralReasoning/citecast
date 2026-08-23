"""Test agent for CiteCast.

Runs a real model (gpt-5.2 via the responses API) against the environment: the
agent researches each paper through the backdated web_search/web_fetch tools
and calls submit_prediction. Every step is appended to citecast_trajectory.jsonl.

Runs against the deployed env by default; set LOCAL=1 for localhost:8080 (the
local server process must have OPENREWARD_API_KEY exported — the backdated
toolset reads the process environment).

    LOCAL=1 NUM_TASKS=5 python test_agent.py
"""

import asyncio
import json
import os
import time
from pathlib import Path

from openai import AsyncOpenAI
from openreward import AsyncOpenReward

TRAJECTORY_FILE = Path(__file__).parent / "citecast_trajectory.jsonl"


def _load_openai_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key
    env_file = Path.home() / "Documents" / "avalon-sft" / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.removeprefix("export ").strip()
            if line.startswith("OPENAI_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"')
                os.environ["OPENAI_API_KEY"] = key
                return key
    raise ValueError("OPENAI_API_KEY not set and not found in ~/Documents/avalon-sft/.env")


def _text_of(response) -> str:
    parts = []
    for item in response.output:
        if item.type == "message":
            for block in item.content:
                if block.type == "output_text":
                    parts.append(block.text)
    return "\n".join(parts).strip()


class Trajectory:
    def __init__(self, path: Path):
        self.file = open(path, "a")

    def record(self, kind: str, **payload):
        self.file.write(json.dumps({"ts": time.time(), "kind": kind, **payload}) + "\n")
        self.file.flush()


async def main():
    or_client = AsyncOpenReward()
    oai_client = AsyncOpenAI(api_key=_load_openai_key())

    MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-5.2")
    ENV_NAME = os.environ.get("ENV_NAME", "GeneralReasoning/CiteCast")
    SPLIT = os.environ.get("SPLIT", "train")
    NUM_TASKS = int(os.environ.get("NUM_TASKS", "1"))
    MAX_TURNS = int(os.environ.get("MAX_TURNS", "15"))
    OPENREWARD_API_KEY = os.getenv("OPENREWARD_API_KEY", "")

    base_url = "http://localhost:8080" if os.environ.get("LOCAL") else None
    environment = or_client.environments.get(name=ENV_NAME, base_url=base_url)
    print(f"Environment: {ENV_NAME} ({base_url or 'deployed'})")

    tasks = await environment.list_tasks(split=SPLIT)
    tools = await environment.list_tools(format="openai")
    print(f"Found {len(tasks)} tasks in split '{SPLIT}'")
    print(f"Tools: {[t['name'] for t in tools]}")

    trajectory = Trajectory(TRAJECTORY_FILE)
    trajectory.record(
        "config", model=MODEL_NAME, env=ENV_NAME, split=SPLIT,
        num_tasks=NUM_TASKS, base_url=base_url or "deployed",
    )

    # The file is shuffled and task ids carry no band, so an index stride is
    # as close to a stratified pick as the client can get.
    stride = max(1, len(tasks) // max(NUM_TASKS, 1))
    chosen = [tasks[i * stride] for i in range(NUM_TASKS)]

    rewards = []
    for task in chosen:
        task_id = task.task_spec["task_id"]
        print(f"\n=== Task {task_id} ===")
        trajectory.record("task_start", task_id=task_id)

        async with environment.session(
            task=task,
            secrets={**({"api_key": OPENREWARD_API_KEY} if OPENREWARD_API_KEY else {})},
        ) as session:
            prompt = await session.get_prompt()
            trajectory.record("prompt", task_id=task_id, text=prompt[0].text)
            input_list = [{"role": "user", "content": prompt[0].text}]

            reward = None
            web_calls = 0
            finished = False
            turn = 0
            while turn < MAX_TURNS and not finished:
                turn += 1
                response = await oai_client.responses.create(
                    model=MODEL_NAME, tools=tools, input=input_list,
                )
                input_list += response.output
                message_text = _text_of(response)
                if message_text:
                    trajectory.record("model_message", task_id=task_id, turn=turn,
                                      text=message_text[:4000])

                calls = [i for i in response.output if i.type == "function_call"]
                if not calls:
                    print(f"[{turn}] model stopped without submitting")
                    trajectory.record("task_end", task_id=task_id, reward=None,
                                      note="no_submit")
                    break

                for item in calls:
                    args = json.loads(str(item.arguments))
                    tool_result = await session.call_tool(item.name, args)
                    output_text = tool_result.blocks[0].text if tool_result.blocks else ""
                    input_list.append({
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": output_text,
                    })
                    if item.name in ("web_search", "web_fetch"):
                        web_calls += 1
                    print(f"[{turn}] {item.name}: {json.dumps(args)[:110]}")
                    trajectory.record(
                        "tool_call", task_id=task_id, turn=turn, tool=item.name,
                        args=args, output=output_text[:2000],
                    )
                    if tool_result.finished:
                        reward = tool_result.reward
                        finished = True
                        print(f"submit -> reward={reward} (web calls: {web_calls})")
                        trajectory.record(
                            "submit_result", task_id=task_id, reward=reward,
                            metadata=tool_result.metadata, web_calls=web_calls,
                        )
                        trajectory.record("task_end", task_id=task_id, reward=reward)
                        break

            rewards.append(reward)

    scored = [r for r in rewards if r is not None]
    mean = sum(scored) / len(scored) if scored else None
    print(f"\n=== Summary ===")
    print(f"num_tasks={len(rewards)} num_scored={len(scored)} mean_reward={mean}")
    print(f"rewards={rewards}")
    print(f"trajectory: {TRAJECTORY_FILE}")
    trajectory.record("summary", rewards=rewards, mean_reward=mean)


if __name__ == "__main__":
    asyncio.run(main())
