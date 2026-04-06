import asyncio
import json
import logging
from pathlib import Path

from llm import *
from prompts import Prompts
from rate_limiter import RateLimiter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("./description.log")],
)
logger = logging.getLogger(__name__)


def format_data(data: dict) -> str:
    # fields:
    # name
    # background (NPC's description)
    # npc_location
    # quest_location
    # player (player's description')
    # quest (quest description)
    return f"Quest information: {data['quest']} NPC Name: {data['name']} NPC description: {data['background']} NPC's location: {data['npc_location']} Quest Location: {data['quest_location']} Player's Description: {data['player']}"


def build_system_prompt(data: dict) -> str:
    return (
        f"Enter roleplay mode. You are {data['name']}. "
        f"Background: {data['background']} "
        f"Current Location: {data['npc_location']} "
        f"Quest: {data['quest_location']} "
        f"Roleplaying Instructions: "
        f"- Speak using appropriate tone and vocabulary "
        f"- Reference your background and current surroundings naturally "
        f"- Keep responses conversational and authentic "
        f"- React to the player's words and intentions. "
        f"Your first response should be a greeting to the player."
    )


def read_jsonl(path: Path) -> list[tuple[str, dict]]:
    rows = []
    with open(path, "r") as f:
        for row in f:
            data = json.loads(row)
            rows.append((format_data(data), data))
    return rows


def parse_response(input: str) -> dict:
    # Strip markdown code fences if present
    text = input.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("No JSON object found in response")
        return json.loads(text[start : end + 1])


async def generate_desc(
    quest: str,
    meta: dict,
    semaphore: asyncio.Semaphore,
    prompts: Prompts,
    file,
    limiter: RateLimiter,
    write_lock: asyncio.Lock,
    max_retries: int = 3,
):
    async with semaphore:
        parsed = None
        for attempt in range(max_retries):
            try:
                await limiter.wait()
                thinking, response = await respond(
                    prompts.generate_conversation(), quest
                )
                parsed = parse_response(response)
                break
            except Exception as e:
                logger.error(f"API Error (attempt {attempt + 1}/{max_retries}): {e}")
        if parsed is None:
            logger.info("FAILED")
            return
    messages = [
        {"role": "system", "content": build_system_prompt(meta)},
        {"role": "user", "content": "Greetings"},
        *parsed["messages"],
    ]
    async with write_lock:
        file.write(json.dumps({"messages": messages}) + "\n")
    logger.info("Done")


async def main():
    quest_souce_file = Path("./rpg-quests-desc.jsonl")
    # expects 1994
    output_file = Path("./rpg-quests-dialogue.jsonl")

    prompts = Prompts()
    limiter = RateLimiter(30)
    semaphore = asyncio.Semaphore(29)

    quest = read_jsonl(quest_souce_file)
    write_lock = asyncio.Lock()
    with open(output_file, "a") as f:
        tasks = [
            generate_desc(row, meta, semaphore, prompts, f, limiter, write_lock)
            for row, meta in quest
        ]
        await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
