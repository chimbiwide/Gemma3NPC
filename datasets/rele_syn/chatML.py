import json

def to_chatML(filename:str):
    chatML = []
    with open(filename, 'r') as f:
        for lines in f:
            line = lines.strip()
            record = json.loads(line)
            chatMLRecord = {
                "messages": record,
            }
            chatML.append(chatMLRecord)

    write_jsonl_to_file(chatML, "../ReLe.jsonl")

def write_jsonl_to_file(records, output_filename):
    with open(output_filename, 'w', encoding='utf-8') as output_file:
        for record in records:
            json_line = json.dumps(record, ensure_ascii=False)
            output_file.write(json_line + '\n')

if __name__ == "__main__":
    to_chatML("../formatted_chat.txt")