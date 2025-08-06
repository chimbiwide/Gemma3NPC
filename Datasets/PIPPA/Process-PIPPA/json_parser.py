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
        print(f"Wrote {len(cleaned_records)} cleaned records to {output_filename}")


def process_single_record(record, line_number):

    categories = record.get('categories', []) or []
    bot_name = record.get('bot_name', '')
    bot_greeting = record.get('bot_greeting', '')
    bot_description = record.get('bot_description', '')
    bot_definitions = record.get('bot_definitions', '')
    conversation = record.get('conversation', '')

    if categories is None or categories == "null" or categories == [] or not categories:
        categories = "none"
    elif isinstance(categories, list):
        seperator = ","
        categories = seperator.join(categories)
    elif isinstance(categories, str):
        categories = categories
    else:
        categories = str(categories)


    SystemPrompt = f"""system 
    Enter Roleplay Mode. You are roleplaying as {bot_name}. You must always stay in character.
    Your goal is to create an immersive, fun, creative roleplaying experience for the user. You must respond in a way that drives the conversation forward.
    Character Persona: 
    Name: {bot_name}
    Category of your character: {categories}
    Description of your character: {bot_description}
    Definition of your character (contains example chats so that you can better roleplay as the character): {bot_definitions}
    
    Now with the information provided, generate {bot_name}'s greeting to the user: 
    """

    chatml_messages = []

    chatml_messages.append({
        "role": "user",
        "content": SystemPrompt.strip()
    })

    for i, turn in enumerate(conversation):
        message = turn['message']
        ishuman = turn['is_human']
        role = "user" if ishuman else "assistant"

        chatml_messages.append({
            "role": role,
            "content": message
        })

    chatml_record = {
        "messages": chatml_messages,
    }

    return chatml_record

def write_jsonl_to_file(records, output_filename):
    with open(output_filename, 'w', encoding='utf-8') as output_file:
        for record in records:
            json_line = json.dumps(record, ensure_ascii=False)
            output_file.write(json_line + '\n')

if __name__ == '__main__':
    process_jsonl_line_by_line('Filtered PIPPA/processed_pippa.jsonl', 'Filtered PIPPA/pippa_chatml.jsonl')