import csv
import json
import os
import time
from typing import List, Dict, Any
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# Fix: Use consistent environment variable name
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


def read_prompts_txt(filename: str):
    prompts = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            prompts.append(line)
    return prompts


def create_game_system_prompt(system_prompt):
    """Create a more detailed system prompt for a game-centric dataset."""
    return f"""Generate a complete RPG conversation dataset entry following this exact format.

**TASK:** Create one realistic fantasy RPG conversation between a player and NPC character.

**NPC CHARACTER:** The first user message contains the NPC's system prompt defining their personality, background, and behavior. Use this to roleplay the NPC consistently throughout the conversation.

**CONVERSATION STRUCTURE:**
- Exactly 16 alternating messages (user/assistant pattern)  
- Message 1: NPC system prompt (provided)
- Message 2: NPC's opening greeting/response
- Messages 3-16: Natural back-and-forth dialogue (7 more exchanges)
- Player messages: 1-2 sentences, conversational, can sometimes be creative
- NPC messages: 2-4 sentences, immersive and character-driven

**QUALITY STANDARDS:**
- NPC stays completely in character based on the system prompt
- Dialogue feels natural and advances meaningfully
- Include emotional reactions, personality quirks, background references
- Avoid modern language or breaking immersion
- Create engaging interaction (quest, trade, lore, social exchange)

**CRITICAL:** Output ONLY valid JSON. No explanations, markdown, or extra text.

Generate exactly 16 messages following this JSON structure:
{{
 "messages": [
 {{"role": "user", "content": "{system_prompt}"}},
 {{"role": "assistant", "content": "NPC greeting based on their character"}},
 {{"role": "user", "content": "Player's response"}},
 {{"role": "assistant", "content": "NPC continues in character"}},
 {{"role": "user", "content": "Player response"}},
 {{"role": "assistant", "content": "NPC stays in character"}},
 {{"role": "user", "content": "Player response"}},
 {{"role": "assistant", "content": "NPC maintains character"}},
 {{"role": "user", "content": "Player response"}},
 {{"role": "assistant", "content": "NPC consistent character"}},
 {{"role": "user", "content": "Player response"}},
 {{"role": "assistant", "content": "NPC stays in character"}},
 {{"role": "user", "content": "Player response"}},
 {{"role": "assistant", "content": "NPC maintains character"}},
 {{"role": "user", "content": "Player response"}},
 {{"role": "assistant", "content": "NPC final response in character"}}
 ]
}}"""


def generate_conversation(system_prompt: str, max_retries: int = 7):
    """Generate a conversation using Gemini 2.0 Flash."""
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model='gemini-2.0-flash-001',
                contents=system_prompt,
                config=types.GenerateContentConfig(
                    temperature=1.0 + (attempt * 0.1),  # Increase temperature with each retry
                    top_p=0.95,
                    max_output_tokens=3000,
                    response_mime_type="application/json",
                    # CORRECTED SAFETY SETTINGS FORMAT
                    safety_settings=[
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                            threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH
                        ),
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                            threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH
                        ),
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                            threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH
                        ),
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                            threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH
                        )
                    ]
                )
            )

            # Parse the response - should be pure JSON now with response_mime_type
            response_text = response.text.strip()
            
            # Try direct parsing first (since we're using response_mime_type)
            try:
                conversation_data = json.loads(response_text)
            except json.JSONDecodeError:
                # Fallback: extract JSON manually and fix common issues
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                
                if start_idx != -1 and end_idx > start_idx:
                    json_str = response_text[start_idx:end_idx]
                    
                    # Fix common JSON issues
                    json_str = json_str.replace(',]', ']')  # Remove trailing commas in arrays
                    json_str = json_str.replace(',}', '}')  # Remove trailing commas in objects
                    
                    conversation_data = json.loads(json_str)
                else:
                    raise json.JSONDecodeError("No valid JSON found", response_text, 0)

            # Validate the conversation structure - check for 'messages' format
            if 'messages' in conversation_data:
                msgs = conversation_data['messages']
                if len(msgs) == 16:  # Expecting exactly 16 turns
                    # Check if alternating user/assistant pattern
                    valid_pattern = True
                    for i, turn in enumerate(msgs):
                        expected_role = "user" if i % 2 == 0 else "assistant"
                        if turn.get('role') != expected_role:
                            valid_pattern = False
                            break

                    if valid_pattern:
                        return conversation_data

            print(f"Attempt {attempt + 1} failed: Invalid format or structure")

        except json.JSONDecodeError as e:
            print(f"Attempt {attempt + 1} failed: JSON parsing error - {str(e)}")
        except Exception as e:
            print(f"Attempt {attempt + 1} failed with error: {str(e)}")

        if attempt < max_retries - 1:

            time.sleep(0.2)  # Wait before retry

    # Return a default conversation if all attempts fail
    print("All attempts failed, using default conversation")
    return " "


def create_datasets(prompts: List[str], limit: int = None):
    """Create both original and game-centric datasets."""
    npc_dataset = []

    # Limit the number of prompts to process (for testing)
    prompts_to_process = prompts[:limit] if limit else prompts

    print(f"Processing {len(prompts_to_process)} prompts...")

    for i, system_prompt in enumerate(prompts_to_process):
        print(f"  [{i+1}/{len(prompts_to_process)}] - Generating NPC conversation...")
        game_prompt = create_game_system_prompt(system_prompt)
        game_conv = generate_conversation(game_prompt)
        npc_dataset.append(game_conv)

        # Small delay to avoid rate limiting
        time.sleep(0.3)

    return npc_dataset


def save_datasets(game_dataset: List):
    """Save datasets to JSON file."""
    # Filter out failed generations (empty strings)
    valid_conversations = [c for c in game_dataset if c != " "]
    
    with open('game_dataset.json', 'w', encoding='utf-8') as f:
        json.dump(valid_conversations, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(valid_conversations)} valid conversations to game_dataset.json")

def append_conversation(conversation: Dict):
    """Append a single conversation to JSONL file."""
    with open('game_dataset.jsonl', 'a', encoding='utf-8') as f:
        json.dump(conversation, f, ensure_ascii=False)
        f.write('\n')


def main():
    """Main function to orchestrate the data generation."""
    # Check if API key is set (FIXED: now matches the variable used above)
    if not os.environ.get('GOOGLE_API_KEY'):
        print("Error: Please set your GOOGLE_API_KEY environment variable")
        print("Run: export GOOGLE_API_KEY=your_api_key_here")
        return

    # Read prompts from CSV
    print("Reading prompts from CSV...")
    prompts = read_prompts_txt('system_prompts.txt')
    print(f"Found {len(prompts)} prompts")

    # For testing, you can limit the number of prompts to process
    # Change this to None to process all prompts
    LIMIT = None # Process only first 10 prompts for testing

    # Create datasets
    NPC_dialogue = create_datasets(prompts, limit=LIMIT)

    # Save datasets
    save_datasets(NPC_dialogue)

    print("\nData generation complete!")
    print(f"Generated {len(NPC_dialogue)} conversations for each dataset")


if __name__ == "__main__":
    main()