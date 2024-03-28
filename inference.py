import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


FT_PATH = 'saved_sft/llama2/'

tokenizer = AutoTokenizer.from_pretrained(FT_PATH, local_files_only=True, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(FT_PATH, device_map="auto", local_files_only=True, trust_remote_code=True)

def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.read().strip()

system = read_file('samples/system.txt')
scene_card = read_file('samples/scene_card.txt')
first_character_card = read_file('samples/first_character_card.txt')
second_character_card = read_file('samples/second_character_card.txt')

prompt = system + scene_card + first_character_card + second_character_card

inputs = tokenizer(prompt, return_tensors='pt')
inputs = inputs.to(model.device)
pred = model.generate(input_ids=inputs.input_ids, max_new_tokens=2048,repetition_penalty=1.1)

print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))