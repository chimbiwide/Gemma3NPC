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


def read_prompts_csv(filename: str) -> List[Dict[str, str]]:
    """Read prompts from CSV file."""
    prompts = []
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompts.append({
                'name': row['Name'],
                'biography': row['Biography'],
                'Query': row['Query'],
                'Response': row['Response'],
            })
    return prompts


def create_system_prompt(name: str, biography: str, Query: str, Response: str) -> str:
    return f"System: Enter Roleplay Mode. You are {name}, a character living in a fantasy RPG world. You must stay completely in character at all times. CHARACTER PROFILE: Name: {name} Background: {biography} CONVERSATION EXAMPLE: Player: \"{Query}\" {name}: \"{Response}\" ROLEPLAY GUIDELINES: - Speak naturally as {name} would, using appropriate tone and vocabulary - Reference your background and motivations in responses - Show personality through your word choices and reactions - Keep responses conversational (2-4 sentences typically) - React authentically to what the player says - You don't know you're in a game - this is your real world CURRENT SITUATION: The player approaches you for the first time. Greet them as {name} would naturally react to meeting a stranger in your world."

def save_datasets(system_prompts: List[str], filename: str = 'system_prompts.txt'):
    """Save system prompts to a text file."""
    with open(filename, 'w', encoding='utf-8') as f:
        for i, prompt in enumerate(system_prompts, 1):
            f.write(prompt)
            f.write("\n")
    print(f"Saved {len(system_prompts)} system prompts to {filename}")


def main():
    # Read prompts from CSV
    print("Reading prompts from CSV...")
    prompts = read_prompts_csv('npc.csv')
    print(f"Found {len(prompts)} prompts")

    # Generate system prompts for each NPC
    system_prompts = []
    for prompt_data in prompts:
        system_prompt = create_system_prompt(
            prompt_data['name'],
            prompt_data['biography'], 
            prompt_data['Query'],
            prompt_data['Response']
        )
        system_prompts.append(system_prompt)

    # Save system prompts to txt file
    save_datasets(system_prompts)

    print("\nSystem prompt generation complete!")
    print(f"Generated {len(system_prompts)} system prompts")


if __name__ == "__main__":
    main()