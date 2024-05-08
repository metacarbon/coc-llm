# Call of Cthulhu (CoC) Dialogue Generation

This project utilizes fine-tuned language models to generate dialogues based on the "Call of Cthulhu" role-playing game.

## Installation Steps

1. **Create and activate a new Conda environment:**
   ```bash
   conda create -n coc python=3.8
   conda activate coc
   ```
2. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Set up Accelerate with DeepSpeed:**
   ```bash
   accelerate config
   ```

## Configuration

### DeepSpeed Settings

Adjust the DeepSpeed settings to meet your requirements by modifying the configuration file located at `config/deepspeed.json`.

## Dataset

### CoC Dialogue Dataset

The dataset is located at `data/coc_dialogue.json` and is specifically crafted for fine-tuning the language model on CoC dialogue generation.

### Dataset Generation

The folder `coc_data_gen/` details the process of dataset generation using the GPT-3.5-turbo API.

## Model Weights

Download the supervised fine-tuned weights for Baichuan-7B and Llama2-7B from the following link:
- [Baichuan-7B and Llama2-7B SFT Weights](https://pan.quark.cn/s/b6ce9a5fa2ae)

## Training

To train the language model, use the following command which leverages Accelerate with DeepSpeed:
```bash
ACCELERATE_USE_DEEPSPEED=true CUDA_VISIBLE_DEVICES="0,1" accelerate launch finetuning.py
```
Adjust the `CUDA_VISIBLE_DEVICES` parameter as necessary to specify the GPUs you intend to use.

## Inference

Generate dialogues with the trained model by running:
```bash
python inference.py
```
This script takes prompts and generates dialogues using the fine-tuned model.

## Directory Structure

- `coc_data_gen/`: Scripts for CoC dataset generation using the GPT-3.5-turbo API.
- `config/`: DeepSpeed configuration file.
- `data/`: CoC dialogue dataset and additional evaluation data.
- `samples/`: Sample prompts for inference.
- `finetuning.py`: Training script for the language model.
- `inference.py`: Inference script for dialogue generation.
- `requirements.txt`: Required Python dependencies.

## License

This project is released under the MIT License.

