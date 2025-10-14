import csv
import os
from typing import List, Dict
from google import genai
from dotenv import load_dotenv

load_dotenv()

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
                'Emotion': row['Emotion'],
            })
    return prompts


def create_char_info(name: str, biography: str, Query: str, Response: str, Emotion: str) -> str:
    return f"""Character Name: {name}, Character Biography: {biography}  Sample Question: {Query} {name}'s response: {Response} Current Emotion: {Emotion}"""

def save_info(system_prompts: List[str], filename: str = 'info.txt'):
    """Save system prompts to a text file."""
    with open(filename, 'w', encoding='utf-8') as f:
        for i, prompt in enumerate(system_prompts, 1):
            f.write(prompt)
            f.write("\n")
    print(f"Saved {len(system_prompts)} system prompts to {filename}")


def main():
    # Read prompts from CSV
    print("Reading prompts from CSV...")
    prompts = read_prompts_csv('npc_dialogue.csv')
    print(f"Found {len(prompts)} prompts")

    # Generate system prompts for each NPC
    system_prompts = []
    for prompt_data in prompts:
        system_prompt = create_char_info(
            prompt_data['name'],
            prompt_data['biography'],
            prompt_data['Query'],
            prompt_data['Response'],
            prompt_data['Emotion']
        )
        system_prompts.append(system_prompt)

    # Save system prompts to txt file
    save_info(system_prompts)

    print("\nSystem prompt generation complete!")
    print(f"Generated {len(system_prompts)} system prompts")


if __name__ == "__main__":
    main()