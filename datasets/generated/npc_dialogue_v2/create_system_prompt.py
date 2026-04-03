import csv
import json
import os
import time
from typing import List, Dict


def read_char_txt(filename: str):
    prompts = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            prompts.append(line)
    return prompts

def read_location_txt(filename: str):
    prompts = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            prompts.append(line)
    return prompts

def read_name(filename: str):
    name = []
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
                name.append(row['Name'])
    return name

def create_char_info(char_bio: str, location: str, name:str) -> str:
    return f"Enter roleplay mode. You are {name}. Background: {char_bio} Current Location: {location} Roleplaying Instructions: - Speak using appropriate tone and vocabulary - Reference your background and current surroundings naturally - Keep responses conversational and authentic - React to the player's words and intentions. Your first response should be a greeting to the player."

def save_info(system_prompts: List[str], filename: str = 'system_prompt.txt'):
    with open(filename, 'w', encoding='utf-8') as f:
        for i, prompt in enumerate(system_prompts, 1):
            f.write(prompt)
            f.write("\n")
    print(f"Saved {len(system_prompts)} system prompts to {filename}")


def main():

    #read character bio from txt
    print("Reading character bio...")
    char_bio = read_char_txt("char_bio.txt")
    print(f"Found {len(char_bio)} bios")

    print("Reading location description...")
    location = read_location_txt("location_description.txt")
    print(f"Found {len(location)} locations")

    # Read names from CSV
    print("Reading prompts from CSV...")
    name = read_name('npc_dialogue.csv')
    print(f"Found {len(name)} prompts")

    if not (len(char_bio) == len(location) == len(name)):
        print("Length mismatch")
        return

    # Generate system prompts for each NPC
    system_prompts = []
    for bio, loc, npc_name in zip(char_bio, location, name):
        system_prompt = create_char_info(
            char_bio = bio,
            location = loc,
            name = npc_name
        )
        system_prompts.append(system_prompt)
    # Save system prompts to txt file
    save_info(system_prompts)

    print("\nSystem prompt generation complete!")
    print(f"Generated {len(system_prompts)} system prompts")


if __name__ == "__main__":
    main()