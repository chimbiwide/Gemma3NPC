import csv
import json
import ast
import re

def read_ReLe(filename):
    conversation = []

    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            conversation.append(row['chat_history'])

    return conversation

def format_dataset(conversation):
    fixed_lines = []
    for line in conversation:
        fixed_line = re.sub(r'}\s+{', '},{', line.strip())
        messages = ast.literal_eval(fixed_line)

        reordered_messages = [
            {'role': msg['role'], 'content': msg['content']}
            for msg in messages
        ]
        fixed_lines.append(json.dumps(reordered_messages))

    return fixed_lines

def write_info(conversation: list, filename: str = 'json_chat_history.txt'):
    with open(filename, 'w', encoding='utf-8') as f:
        for chat in conversation:
            f.write(chat)
            f.write('\n')

    print(f"saved {len(conversation)}")


def main():
    conversation = read_ReLe("../rele.csv")
    conversation = format_dataset(conversation)
    write_info(conversation, "../formatted_chat.txt")

if __name__ == "__main__":
    main()