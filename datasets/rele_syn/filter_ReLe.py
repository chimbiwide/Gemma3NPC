import lmstudio as lms
import json

def read_file(filename: str) -> list:
    chat_history = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            chat_history.append(line.strip())
    return chat_history


def llm_instance(text: str):
    schema = {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "enum": [
                    "Yes",
                    "No"
                ]
            }
        },
        "required": ["answer"],
    }

    prompt = """You are tasked with classifying whether the given interactions between the model and the user has a lot of repeating goodbyes.
Here is an example
```
<|model|> Thanks man! I am glad you think so!
<|user|> You should thank ME though.
<|model|> Well you did make me so thanks to you too!
<|user|> Exactly! And we both know you couldn't have done it without me!
<|model|> That is true! We work together!
<|user|> We do.
<|model|> So let us continue our adventures and make new friends!
<|user|> Deal.
<|model|> Great! I love making new friends! Let's talk more later!
<|user|> Cool. Bye!
<|model|> Later!
<|user|> Later!
<|model|> Ok bye!
<|user|> Bye!
<|model|> Bye!
<|user|> Bye!
<|model|> Bye!
<|user|> Bye!
```

After thinking, response only in "Yes" or "No"

Here is the given conversation: \n""" + text

    model = lms.llm("qwen/qwen3-4b-thinking-2507", config={
        "contextLength": 8192,
        "gpu": {
            "ratio": 1.0,
        }
    })

    result = model.respond(prompt, response_format=schema, config={
        "temperature": 0.6,
        "topKSampling": 20,
        "minPSampling": 0,
        "topPSampling": 0.95,
    })

    answer = json.loads(result.content)
    print(answer.get("answer"))
    return answer.get("answer")

def write_info(conversation: list, filename:str = 'filtered_json_chat_history.txt'):
    with open(filename, 'w', encoding='utf-8') as f:
        for i, chat in enumerate(conversation, 1):
            chat = chat.replace("\n", " ")
            f.write(chat)
            f.write("\n")

    print(f"saved {len(conversation)}")

def to_boolean(text:str) -> bool | None:
    if text == "Yes":
        return True
    elif text == "No":
        return False
    return None


def main():
    history = read_file("json_chat_history.txt")
    filtered_history = []
    filtered = 0

    for i, chat in enumerate (history, 1):
        result = llm_instance(chat)
        print(f"[{i} / {len(history)}], total {filtered}\n")
        if not to_boolean(result):
            filtered_history.append(chat)
        else:
            filtered += 1
    print(f"\n\nThere are {len(filtered_history)} number of good conversations")
    write_info(filtered_history)

if __name__ == "__main__":
    main()