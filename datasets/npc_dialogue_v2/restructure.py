from pathlib import Path
import json

data = Path("../../source/npc_dialogue.jsonl")
with open(data, 'r') as f_in, open("npc_dialogue_system.jsonl", 'w') as f_out:
    for line in f_in:
        row = json.loads(line)
        messages = row["messages"]

        first = messages[0]
        system = first["content"]
        if system:
            new_msg = [{"role": "system", "content": system},
                       {"role": "user", "content": "Greetings"},
                       {"role": "assistant", "content": messages[1]["content"]},
                       ] + messages[2:]
            row["messages"] = new_msg

        f_out.write(json.dumps(row) + '\n')
    print("DONE")
