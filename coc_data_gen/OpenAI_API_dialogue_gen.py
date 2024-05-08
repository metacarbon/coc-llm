import os
import json
import time
import random
import itertools
import requests
import openai

# Your API key
os.environ["OPENAI_API_KEY"] = "sk-XXXX"

# Load your API key from an environment variable or secret management service
openai.api_key = os.getenv("OPENAI_API_KEY")

# Your organization key
openai.organization = "org-XXXX"


# Read the scene card JSON file
with open('scene_cards.json', 'r', encoding="utf-8") as file:
    scene_cards = json.load(file)

# Read the character card JSON file
with open('characters.json', 'r', encoding="utf-8") as file:
    character_cards = json.load(file)

# Read the sample dialogue JSON file
with open('sample_dialogues.json', 'r', encoding="utf-8") as file:
    sample_dialogues = json.load(file)

for character_card in character_cards:
    del character_card['id']


# Function to find the dialogue for a pair of characters
def find_dialogue(character_names):
    for dialogue in sample_dialogues:
        if set(dialogue["characters"]) == set(character_names):
            return dialogue["dialogue"]
    return None

def generate_dialogue(prompt, max_retries=3, wait_time=60):
    retries = 0
    while retries < max_retries:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a dialogue generation language model."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2048
            )
            return response.choices[0].message.content
        except (openai.error.RateLimitError, openai.error.APIError, requests.exceptions.ConnectionError) as e:
            if retries < max_retries - 1:
                print(f"Error encountered: {str(e)}. Retrying in {wait_time} seconds... (attempt {retries + 1}/{max_retries})")
                time.sleep(wait_time)
            retries += 1
    raise Exception("Failed to generate dialogue after multiple attempts due to API errors.")


last_time_scene_id = -1
first_entry = True if last_time_scene_id < 0 else False
randomly_choose = True

if first_entry:
    # Create an empty file or clear the existing file
    with open('generated_dialogues.json', 'w', encoding="utf-8") as outfile:
        outfile.write('[')

for scene_card in scene_cards:
    scene_id = scene_card.pop("id")
    if scene_id <= last_time_scene_id: 
        continue

    character_card_combinations = list(itertools.combinations(character_cards, 2))

    if randomly_choose == True:
        # Randomly choose 4 character combinations
        selected_combinations = random.sample(character_card_combinations, 4)

        for char_combination in selected_combinations:
            character1_card, character2_card = char_combination
            char_names = [character1_card["name"], character2_card["name"]]
            sample_dialogue = find_dialogue(char_names)

            # Construct the prompt with the provided information
            prompt = f"""
            Sample Dialogue:
            {sample_dialogue}

            Generate a dialogue between two characters in a Call of Cthulhu setting based on the provided character cards and scene card. Keep the characters' personalities, skills, and speaking styles consistent with their respective cards. Make sure to showcase their unique traits, skills, and quirks in the dialogue, and have the characters engage with the situation and each other according to their personalities and abilities.

            Scene card:
            {scene_card}

            Character 1 card:
            {character1_card}

            Character 2 card:
            {character2_card}

            ### Dialogue:
            By explicitly asking for the unique traits, skills, and quirks to be showcased in the dialogue, the generated content will be more likely to highlight the aspects of the characters that make them distinctive and align with their character cards.

            """

            generated_dialogue = generate_dialogue(prompt)

            # Save the generated dialogue to the JSON file
            with open('generated_dialogues.json', 'a', encoding="utf-8") as outfile:
                if not first_entry:
                    outfile.write(',')
                else:
                    first_entry = False

                json.dump({
                    "scene": scene_id,
                    "characters": char_names,
                    "dialogue": generated_dialogue
                }, outfile, ensure_ascii=False, indent=4)

    else:
        for char_combination in character_card_combinations:
            character1_card, character2_card = char_combination
            char_names = [character1_card["name"], character2_card["name"]]
            sample_dialogue = find_dialogue(char_names)

            # Construct the prompt with the provided information
            prompt = f"""
            Sample Dialogue:
            {sample_dialogue}

            Generate a dialogue between two characters in a Call of Cthulhu setting based on the provided character cards and scene card. Keep the characters' personalities, skills, and speaking styles consistent with their respective cards. Make sure to showcase their unique traits, skills, and quirks in the dialogue, and have the characters engage with the situation and each other according to their personalities and abilities.

            Scene card:
            {scene_card}

            Character 1 card:
            {character1_card}

            Character 2 card:
            {character2_card}

            ### Dialogue:
            By explicitly asking for the unique traits, skills, and quirks to be showcased in the dialogue, the generated content will be more likely to highlight the aspects of the characters that make them distinctive and align with their character cards.

            """

            generated_dialogue = generate_dialogue(prompt)

            # Save the generated dialogue to the JSON file
            with open('generated_dialogues.json', 'a', encoding="utf-8") as outfile:
                if not first_entry:
                    outfile.write(',')
                else:
                    first_entry = False

                json.dump({
                    "scene": scene_id,
                    "characters": char_names,
                    "dialogue": generated_dialogue
                }, outfile, ensure_ascii=False, indent=4)

# Close the JSON array in the file
with open('generated_dialogues.json', 'a', encoding="utf-8") as outfile:
    outfile.write(']')