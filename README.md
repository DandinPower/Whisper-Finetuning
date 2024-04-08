# Whisper-Finetuning

## Installation

1. Following the instructions
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    pip install -U "huggingface_hub[cli]"
    huggingface-cli login
    ```

## Training

1. Modify config of `run.sh`

2. Run the following command to train the model
    ```bash
    bash run.sh
    ```

## Reference

1. [Whisper Finetuning event](https://github.com/huggingface/community-events/tree/main/whisper-fine-tuning-event)