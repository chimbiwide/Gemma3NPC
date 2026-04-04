import csv
import json


def process_jsonl_line_by_line(source_file, append_file):

    records_processed = 0
    with open(source_file, 'a', encoding='utf-8') as source:
        with open(append_file, 'r', encoding='utf-8') as append:
            source.write('\n')
            for line_number, line in enumerate(append, 1):
                # Strip whitespace and skip empty lines
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                    json_line = json.dumps(record, ensure_ascii=False)
                    source.write(json_line + '\n')
                    source.flush()
                    records_processed += 1

                except json.JSONDecodeError as e:
                    print(f"Error parsing line {line_number}: {e}")
                    continue

                if (line_number % 1000) == 0:
                    print(f"Processed {line_number} lines")
    print(f"successfully processed {records_processed} lines")

if __name__ == '__main__':
    process_jsonl_line_by_line('NPC.jsonl', 'game_dataset.jsonl')