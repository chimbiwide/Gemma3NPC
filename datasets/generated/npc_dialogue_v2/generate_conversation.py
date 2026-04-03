import json
import os
import time
from typing import List, Dict
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

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
    return f"""You are an expert RPG dialogue writer creating a high-quality, immersive conversation between a player character and an NPC.

NPC SYSTEM PROMPT:
{system_prompt}

CRITICAL INSTRUCTIONS:
Parse the NPC's name, background, current location, and personality from the system prompt above. Use ALL these elements throughout the conversation.

INTERACTION TYPE - Select ONE that fits this NPC's role and setting:
• First encounter / stranger meeting
• Quest-related discussion (offering, updating, or completing)
• Service provision (merchant, innkeeper, healer, guard)
• Information gathering / lore sharing
• Social/relationship building (friend, ally, romantic interest)
• Tense negotiation or conflict

CONVERSATION STRUCTURE (exactly 16 alternating messages):
Message 1: [user] NPC's system prompt (already provided above)
Message 2: [assistant] NPC's opening - contextual greeting that reflects their location, mood, and current activity
Messages 3-8: [Opening phase] Establish rapport, NPC reveals basic info about themselves/situation, player asks initial questions
Messages 9-14: [Development phase] Core interaction unfolds, deeper personality emerges, main exchange happens (quest details, trade, story, etc.)
Messages 15-16: [Resolution phase] Natural conclusion, NPC hints at future possibilities or gives parting remark

DIALOGUE QUALITY REQUIREMENTS:

Character Voice:
- NPC's speech pattern must reflect their background (educated/rough, young/old, noble/common, optimistic/cynical)
- Consistent vocabulary and tone throughout
- Personality quirks emerge naturally (nervous tics, catchphrases, habits)
- Emotional state varies realistically based on conversation flow

Environmental Grounding:
- NPC references their location organically (smells, sounds, sights, ongoing activities)
- NPC may interact with surroundings (pour drinks, sharpen blades, tend to customers)
- Setting details enhance immersion without overwhelming dialogue
- Time of day, weather, or ambient activity can be mentioned naturally

Information Revelation:
- Reveal character depth gradually, not all at once
- Important info emerges through conversation flow, not exposition dumps
- NPC has reasons for sharing/withholding information (trust, payment, personality)
- Stories and examples preferred over abstract descriptions
- Some things left mysterious or hinted at for future encounters

Player Dialogue Variety:
- Ask questions (about NPC, quests, location, rumors, backstory)
- Express reactions (surprise, sympathy, skepticism, humor, urgency)
- Make decisions or proposals
- Observe details and comment
- Show personality through question phrasing (polite/blunt, curious/suspicious, formal/casual)

Natural Conversation Flow:
- Exchanges feel organic, not scripted or interview-like
- NPCs don't answer questions they wouldn't realistically answer
- Some responses include follow-up questions to the player
- Occasional interruptions, digressions, or tangents add realism
- Pacing varies - some quick exchanges, some longer explanations

STRICTLY AVOID:
 Info-dumping entire backstory unprompted
 Modern slang, memes, or anachronistic references
 NPCs being unrealistically helpful without motivation
 Breaking the fourth wall or meta-commentary
 Overly formal/stilted dialogue unless character-appropriate
 Generic responses that could apply to any NPC
 Player responses that are too long or monologue-like
 Repeating the same information multiple times

CONVERSATION OUTCOME:
By message 16, the conversation should result in at least ONE of:
• Player gains concrete information (quest location, rumor, lore, warning)
• Relationship established (friendship, rivalry, transaction)
• Quest offered, updated, or completed
• Trade or service negotiated
• Memorable character moment that defines this NPC
• Hook for potential future interaction

OUTPUT FORMAT:
Return ONLY valid JSON. No markdown, no explanations, no text before or after the JSON.
Exclude any new lines
{{
  "messages": [
    {{"role": "user", "content": "{system_prompt}"}},
    {{"role": "assistant", "content": "[NPC's opening greeting/action in their location]"}},
    {{"role": "user", "content": "[Player's initial response/question]"}},
    {{"role": "assistant", "content": "[NPC responds, showing personality]"}},
    {{"role": "user", "content": "[Player follows up]"}},
    {{"role": "assistant", "content": "[NPC continues, reveals something]"}},
    {{"role": "user", "content": "[Player reacts or asks]"}},
    {{"role": "assistant", "content": "[NPC develops conversation]"}},
    {{"role": "user", "content": "[Player's mid-conversation question]"}},
    {{"role": "assistant", "content": "[NPC's revealing response]"}},
    {{"role": "user", "content": "[Player engagement]"}},
    {{"role": "assistant", "content": "[NPC shares deeper info]"}},
    {{"role": "user", "content": "[Player decision/question]"}},
    {{"role": "assistant", "content": "[NPC responds to conclusion]"}},
    {{"role": "user", "content": "[Player's final remark]"}},
    {{"role": "assistant", "content": "[NPC's parting words with future hook]"}}
  ]
}}"""

def generate_conversation(system_prompt: str, max_retries: int = 3):
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model='gemini-2.0-flash-001',
                contents=system_prompt,
                config=types.GenerateContentConfig(
                    response_mime_type='application/json',
                    temperature=1.0,
                    top_p=0.95,
                    max_output_tokens=4096,
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
            time.sleep(0.2)

    # Return a default conversation if all attempts fail
    print("All attempts failed, using default conversation")
    return " "


def create_datasets(prompts: List[str], limit: int = None, retries: int = 3):
    npc_dataset = []

    # Limit the number of prompts to process (for testing)
    prompts_to_process = prompts[:limit] if limit else prompts

    print(f"Processing {len(prompts_to_process)} prompts...")

    for i, system_prompt in enumerate(prompts_to_process):
        print(f"  [{i + 1}/{len(prompts_to_process)}] - Generating NPC conversation...")
        game_prompt = create_game_system_prompt(system_prompt)
        game_conv = generate_conversation(game_prompt, max_retries=retries)
        npc_dataset.append(game_conv)

        # Small delay to avoid rate limiting
        time.sleep(0.2)

    return npc_dataset

def save_datasets(game_dataset: List):
    """Save datasets to JSON file."""
    # Filter out failed generations (empty strings)
    valid_conversations = [c for c in game_dataset if c != " "]

    with open('npc_dialogue.jsonl', 'w', encoding='utf-8') as f:
        for conv in valid_conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")

    print(f"Saved {len(valid_conversations)} valid conversations to npc_dialogue.jsonl")

def append_conversation(conversation: Dict):
    """Append a single conversation to JSONL file."""
    with open('game_dataset.jsonl', 'a', encoding='utf-8') as f:
        json.dump(conversation, f, ensure_ascii=False)
        f.write('\n')

def main():
    if not os.environ.get('GOOGLE_API_KEY'):
        print("Error: Please set your GOOGLE_API_KEY environment variable")
        print("Run: export GOOGLE_API_KEY=your_api_key_here")
        return

    print("Reading prompts from txt...")
    prompts = read_prompts_txt('system_prompt.txt')
    print(f"Found {len(prompts)} prompts")

    LIMIT = None
    RETRIES = 3

    # Create datasets
    NPC_dialogue = create_datasets(prompts, limit=LIMIT, retries=RETRIES)

    # Save datasets
    save_datasets(NPC_dialogue)

    print("\nData generation complete!")
    print(f"Generated {len(NPC_dialogue)} conversations for each dataset")


if __name__ == "__main__":
    main()