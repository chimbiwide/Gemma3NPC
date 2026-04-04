from pathlib import Path
import json

def extract_system(prompt: str):
    prefix = "system"
    suffix = "Now with the information provided"
    if prompt.startswith(prefix) and suffix in prompt:
        content = prompt[len(prefix):prompt.index(suffix)]
        return content.strip()
    return None;

data = Path("../../source/pippa_chatml.jsonl")
with open(data, 'r') as f_in, open("pippa_system.jsonl", 'w') as f_out:
    for line in f_in:
        row = json.loads(line)
        messages = row["messages"]

        first = messages[0]
        system = extract_system(first["content"])
        if system:
            new_msg = [{"role": "system", "content": system},
                       {"role": "user", "content": "Greetings"},
                       {"role": "assistant", "content": messages[1]["content"]},
                       ] + messages[2:]
            row["messages"] = new_msg

        f_out.write(json.dumps(row) + '\n')
    print("DONE")
