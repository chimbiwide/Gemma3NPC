import json
import random
import re

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
        print(f"Wrote {len(cleaned_records)} cleaned records to {output_filename}")


def process_single_record(record, line_number):
    categories = record.get('categories', [])
    bot_name = record.get('bot_name', '') or ''
    bot_greeting = record.get('bot_greeting', '') or ''
    bot_description = record.get('bot_description', ' ') or ''
    bot_definitions = record.get('bot_definitions', '') or ''
    conversation = record.get('conversation', '') or ''

    user_name = get_ran_name()

    #process bot_definition
    bot_definitions = bot_definitions.replace('{{char]}', bot_name)
    bot_definitions = replace_random_user_patterns(bot_definitions)
    bot_definitions = bot_definitions.replace('{{users}}', user_name)

    #process bot_description (just in case)
    bot_description = bot_description.replace('{{char]}', bot_name)
    bot_description = replace_random_user_patterns(bot_description)
    bot_description = bot_description.replace('{{users}}', user_name)

    #process conversation
    conversation_str = json.dumps(conversation)
    conversation_str = conversation_str.replace('{{char]}', bot_name)
    conversation_str = replace_random_user_patterns(conversation_str)
    conversation_str = conversation_str.replace('{{users}}', user_name)
    conversation = json.loads(conversation_str)

    #process greeting
    bot_greeting = bot_greeting.replace('{{char]}', bot_name)
    bot_greeting = replace_random_user_patterns(bot_greeting)
    bot_greeting = bot_greeting.replace('{{users}}', user_name)

    cleaned_record = {
        'categories': categories,
        'bot_name': bot_name,
        'bot_greeting': bot_greeting,
        'bot_definitions': bot_definitions,
        'bot_description': bot_description,
        'conversation': conversation
    }

    return cleaned_record

def get_ran_name(names_file='first-names.json'):
    try:
        with open(names_file, 'r', encoding='utf-8') as f:
            names = json.load(f)
            return random.choice(names)
    except FileNotFoundError:
        default_names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry"]
        return random.choice(default_names)

def replace_random_user_patterns(text):
    if not isinstance(text, str):
        return text
    name_mapping = {}
    pattern = r'\{\{random_user_\d+\}'
    def replacer(match):
        matched_pattern = match.group(0)
        if matched_pattern in name_mapping:
            return name_mapping[matched_pattern]
        new_name = get_ran_name()
        name_mapping[matched_pattern] = new_name
        return new_name
    return re.sub(pattern, replacer, text)

def write_jsonl_to_file(records, output_filename):
    with open(output_filename, 'w', encoding='utf-8') as output_file:
        for record in records:
            json_line = json.dumps(record, ensure_ascii=False)
            output_file.write(json_line + '\n')

if __name__ == '__main__':
    #process_jsonl_line_by_line('test_pippa.jsonl', 'test_pippa_shareGPT.jsonl')
    process_jsonl_line_by_line('../Other Datasets/processed_pippa.jsonl', 'Filtered PIPPA/processed_pippa.jsonl')