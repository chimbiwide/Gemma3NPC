import csv
import json
import os
import time
from typing import List, Dict, Any
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


def create_game_setting(info):
    """Create a more detailed system prompt for a game-centric dataset."""
    return f"""You are an expert writer in creating RPG game NPC biographies.
               You will be given a character name, their brief biography, sample dialogue and emotion.
               From the information given, you should write a detailed description around 3-4 sentences
               with a more detailed NPC biography for roleplaying purposes. 
               Only return the written description, exclude all comments, emojis and new lines
               Exclude New lines.
               Character Profile: {info}
"""


def generate_setting(system_prompt: str):
    """Generate a conversation using Gemini 2.0 Flash."""
    response = client.models.generate_content(
        model='gemini-2.0-flash-001',
        contents=system_prompt,
        config=types.GenerateContentConfig(
            temperature=1.0,  # Increase temperature with each retry
            top_p=0.95,
            max_output_tokens=1024,
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
    response_text = response.text.strip()

    return response_text


def create_datasets(prompts: List[str], limit: int = None):
    npc_dataset = []

    prompts_to_process = prompts[:limit] if limit else prompts

    print(f"Processing {len(prompts_to_process)} prompts...")

    for i, system_prompt in enumerate(prompts_to_process):
        print(f"  [{i + 1}/{len(prompts_to_process)}] - Generating NPC conversation...")
        game_prompt = create_game_setting(system_prompt)
        game_setting = generate_setting(game_prompt)
        npc_dataset.append(game_setting)

        # Small delay to avoid rate limiting
        time.sleep(0.2)

    return npc_dataset


def save_datasets(game_dataset: List):
    with open('char_bio.txt', 'w', encoding='utf-8') as f:
        for i, setting in enumerate(game_dataset,1):
            f.write(setting)
            f.write("\n")
    print(f"Saved {len(game_dataset)} valid conversations")


def main():
    """Main function to orchestrate the data generation."""
    if not os.environ.get('GOOGLE_API_KEY'):
        print("Error: Please set your GOOGLE_API_KEY environment variable")
        print("Run: export GOOGLE_API_KEY=your_api_key_here")
        return

    print("Reading prompts from txt...")
    prompts = read_prompts_txt('info.txt')
    print(f"Found {len(prompts)} prompts")

    # Change this to None to process all prompts
    LIMIT = None

    # Create datasets
    NPC_setting = create_datasets(prompts, limit=LIMIT)

    # Save datasets
    save_datasets(NPC_setting)

    print("\nData generation complete!")
    print(f"Generated {len(NPC_setting)} conversations for each dataset")


if __name__ == "__main__":
    main()