import json
import random
from pathlib import Path

TURN_START = "<|turn>"
TURN_END = "<turn|>"

ROLE_MAP = {"system": "system", "user": "user", "assistant": "model"}


def format_conversation(messages: list[dict]) -> str:
    text = ""
    for msg in messages:
        role = ROLE_MAP.get(msg["role"], msg["role"])
        content = msg["content"].strip()
        text += f"{TURN_START}{role}\n{content}{TURN_END}\n"
    return text


def load_jsonl(path) -> list[dict]:
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def main():
    base = Path(__file__).parent
    random.seed(42)

    print("Loading datasets...")
    npc_data  = load_jsonl(base / "datasets/npc_dialogue_v2/npc_dialogue_system.jsonl")
    pippa_data = load_jsonl(base / "datasets/pippa/pippa_system.jsonl")
    rpg_data  = load_jsonl(base / "datasets/rpg_quests/rpg-quests-dialogue.jsonl")

    print(f"  npc_dialogue_v2: {len(npc_data)} entries")
    print(f"  pippa:           {len(pippa_data)} entries")
    print(f"  rpg_quests:      {len(rpg_data)} entries")

    npc_sample   = random.sample([x for x in npc_data  if "messages" in x], 300)
    pippa_sample = random.sample([x for x in pippa_data if "messages" in x], 250)
    rpg_sample   = random.sample([x for x in rpg_data  if "messages" in x], 300)

    calibration_texts = []
    for item in npc_sample + pippa_sample + rpg_sample:
        calibration_texts.append(format_conversation(item["messages"]))

    random.shuffle(calibration_texts)

    output_path = base / "source/imatrix.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(calibration_texts))

    total_chars = sum(len(t) for t in calibration_texts)
    print(f"\nWrote {len(calibration_texts)} conversations to {output_path}")
    print(f"Approximate tokens: ~{total_chars // 4:,}")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
