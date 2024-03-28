import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed, DummyOptim
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import logging
import json
from tqdm import tqdm


logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)

IGNORE_TOKEN_ID = -100


def safe_ids(ids, max_value, pad_id):
    return [i if i < max_value else pad_id for i in ids]


def tokenize(messages, tokenizer):
    input_ids = []
    labels = []

    system = "Generate a dialogue based on the information provided in the Scene Card and Character Cards. You are partaking in a role-play dialogue set in the style of Call of Cthulhu." \
            "The Scene Card will set the context of the dialogue, and the Character Cards will define the personas of the characters involved. Your dialogue should adhere to the character's speaking style, personality traits, and quirks as described in the Character Cards." \
            "Remember, the dialogue should be driven by the character's persona."

    scene_card = messages["Scene Card"]
    first_character_card = messages["First Character Card"]
    second_character_card = messages["Second Character Card"]
    response = messages["Response"] + '</s>'
    
    system_ids = tokenizer.encode(system, add_special_tokens=False)
    scene_card_ids = tokenizer.encode(scene_card, add_special_tokens=False)
    first_character_card_ids = tokenizer.encode(first_character_card, add_special_tokens=False)
    second_character_card_ids = tokenizer.encode(second_character_card, add_special_tokens=False)
    response_ids = tokenizer.encode(response, add_special_tokens=False)

    # Append all the sections to the input_ids
    input_ids += system_ids + scene_card_ids + first_character_card_ids + second_character_card_ids + response_ids
    
    # The labels should ignore the scene_card, first_character_card, and second_character_card (set to IGNORE_TOKEN_ID),
    # and should match the response_ids for the response part
    labels += [IGNORE_TOKEN_ID] * len(system_ids) + [IGNORE_TOKEN_ID] * len(scene_card_ids) + [IGNORE_TOKEN_ID] * len(first_character_card_ids) + [IGNORE_TOKEN_ID] * len(second_character_card_ids) + response_ids
    
    # Ensure lengths do not exceed model's max length
    input_ids = input_ids[:tokenizer.model_max_length]
    labels = labels[:tokenizer.model_max_length]
    
    input_ids = safe_ids(input_ids, tokenizer.vocab_size, tokenizer.pad_token_id)
    labels = safe_ids(labels, tokenizer.vocab_size, IGNORE_TOKEN_ID)
    return input_ids, labels


class CoCData(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        item = self.data[item]
        input_ids, labels = tokenize(item, self.tokenizer)
        return torch.tensor(input_ids), torch.tensor(labels)

    def collate_fn(self, data):
        input_ids, labels = zip(*data)
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_TOKEN_ID)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        features = {
            'input_ids': input_ids.long(),
            'labels': labels.long(),
            'attention_mask': attention_mask.long(),
        }
        return features


def main():
    parser = argparse.ArgumentParser(description='Fine-tuning LLM')
    parser.add_argument('--model_path', type=str, default='/sata/data/LLMs/Llama-2-7b-hf/', help='Path to the pre-trained model')
    parser.add_argument('--save_path', type=str, default='out/Llama-coc-7b', help='Path to save the fine-tuned model')
    args = parser.parse_args()
    model_path = args.model_path
    save_path = args.save_path
    
    set_seed(42)
    accelerator = Accelerator()
    batch_size = 4

    logger.info('Initializing tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, padding_side="right", model_max_length=4096, local_files_only=True, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.unk_token

    logger.info('Initializing model...')
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    

    dataset = CoCData(json.load(open('data/coc_dialogue.json')), tokenizer)

    data_loader = torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn,
                                              batch_size=batch_size, num_workers=0, shuffle=True)

    dummy_optimizer = DummyOptim(model.parameters())

    logger.info('accelerator preparing...')
    model, optimizer, data_loader = accelerator.prepare(model, dummy_optimizer, data_loader)

    for epoch in range(3):
        logger.info('=' * 10 + f'Start training {save_path} epoch {epoch + 1}' + '=' * 10)
        accelerator.wait_for_everyone()
        model.train()
        pbar = tqdm(enumerate(data_loader), total=len(data_loader), disable=(not accelerator.is_local_main_process))
        loss_report = []
        with accelerator.accumulate(model):
            for i, batch in pbar:
                out = model(**batch)
                loss = out.loss

                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                optimizer.zero_grad()

                loss_report.append(accelerator.gather(loss).mean().item())
                pbar.set_description(f"epoch {epoch + 1} iter {i}: train loss {sum(loss_report[-100:]) / len(loss_report[-100:]):.5f}.")
                
        accelerator.wait_for_everyone()
        # save model states
        model.save_checkpoint(f'{save_path}/{epoch}')
        logger.info(f'model for epoch {epoch + 1} is saved...')
        

if __name__ == '__main__':
    main()
