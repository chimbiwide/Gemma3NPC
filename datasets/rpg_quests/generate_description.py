import asyncio
import json
import logging
import time
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


def format_quest(data: dict) -> str:
    return f"Quest title: {data['Title']} Quest Objective: {data['Objective']} Text Description: {data['Text']}"


def read_jsonl(path: Path) -> list[str]:
    rows = []
    with open(path, "r") as f:
        for row in f:
            rows.append(format_quest(json.loads(row)))
    return rows


def parse_response(input: str) -> dict | None:
    try:
        return json.loads(input)
    except:
        try:
            return json.loads(input[input.index("{") : input.rindex("}") + 1])
        except json.JSONDecodeError:
            logger.error(f"Error parsing {input}")
            return None


async def generate_desc(
    quest: str,
    semaphore: asyncio.Semaphore,
    prompts: Prompts,
    file,
    limiter: RateLimiter,
    write_lock: asyncio.Lock,
):
    async with semaphore:
        try:
            await limiter.wait()
            thinking, response = await respond(prompts.generate_setting(), quest)
            parsed = parse_response(response)
            parsed["quest"] = quest
        except Exception as e:
            logger.error(f"API Error: {e}")
            parsed = None
    async with write_lock:
        file.write(json.dumps(parsed) + "\n")
    logger.info(f"Done: {parsed.get('name')}" if parsed else "FAILED")


async def main():
    quest_souce_file = Path(__file__).parent / "../../source/rpg-quests.jsonl"
    output_file = Path("./rpg-quests-desc.jsonl")

    prompts = Prompts()
    limiter = RateLimiter(30)
    semaphore = asyncio.Semaphore(29)

    quest = read_jsonl(quest_souce_file)[:2000]
    write_lock = asyncio.Lock()
    with open(output_file, "a") as f:
        tasks = [
            generate_desc(row, semaphore, prompts, f, limiter, write_lock)
            for row in quest
        ]
        await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
