# Call of Cthulhu (CoC) Dialogue Generation

This project aims to generate dialogues based on the Call of Cthulhu (CoC) role-playing game using fine-tuned language models.

## Installation

1. Create a new conda environment with Python 3.8:\
conda create -n coc python=3.8
2. Activate the environment:\
conda activate coc
3. Install the required dependencies:\
pip install -r requirements.txt
4. Set up Accelerate to use the DeepSpeed configuration:\
accelerate config

## DeepSpeed Settings

The DeepSpeed configuration file is located at `config/deepspeed.json`. Modify this file to adjust the DeepSpeed settings according to your requirements.

## CoC Dataset

The Call of Cthulhu dialogue dataset is located at `data/coc_dialogue.json`. This dataset is used for fine-tuning the language model.

## Training

To train the language model, run the following command:\
ACCELERATE_USE_DEEPSPEED=true CUDA_VISIBLE_DEVICES="0,1" accelerate launch finetuning.py\
This command launches the training script using Accelerate with DeepSpeed enabled. It will use two GPUs for training. Adjust the `CUDA_VISIBLE_DEVICES` parameter to use the GPUs of your choice.

## Inference

To generate dialogues using the trained model, run the following command:\
python inference.py\
This command executes the inference script, which generates dialogues based on the provided prompts and the trained model.

## Directory Structure

- `config/`: Contains the DeepSpeed configuration file.
- `data/`: Contains the CoC dialogue dataset.
- `finetuning.py`: The training script for fine-tuning the language model.
- `inference.py`: The inference script for generating dialogues.
- `requirements.txt`: The list of required Python dependencies.
- `README.md`: This readme file.

## License

This project is licensed under the MIT License.
