import sys

import ijson
import json


def process_jsonl_line_by_line(filename, output_filename=None):

    cleaned_records = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line_number, line in enumerate(file, 1):
            # Strip whitespace and skip empty lines
            line = line.strip()
            if not line:
                continue

            try:
                # Parse each line as a complete JSON object
                record = json.loads(line)

                # Process the record immediately
                cleaned_record = process_single_record(record, line_number)

                if output_filename:
                    cleaned_records.append(cleaned_record)

            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_number}: {e}")
                continue

            if (line_number % 1000) == 0:
                print(f"Processed {line_number} lines")

    if output_filename and cleaned_records:
        write_jsonl_to_file(cleaned_records, output_filename)
        print(print(f"Wrote {len(cleaned_records)} cleaned records to {output_filename}"))


def process_single_record(record, line_number):
    # Example: Extract specific fields and validate them
    cleaned_record = {
        'bot_definitions': record.get('bot_definitions'),
    }

    return cleaned_record

def write_jsonl_to_file(records, output_filename):
    with open(output_filename, 'w', encoding='utf-8') as output_file:
        for record in records:
            json_line = json.dumps(record, ensure_ascii=False)
            output_file.write(json_line + '\n')

if __name__ == '__main__':
    process_jsonl_line_by_line('../cleaned_pippa.jsonl', 'Original PIPPA/definition.jsonl')