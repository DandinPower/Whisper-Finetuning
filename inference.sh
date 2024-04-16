DATASET_NAME="DandinPower/Taiwanese_ASR"
MODEL_NAME="DandinPower/whisper-large-taiwanese-asr-v2"
OUTPUT_PATH="submission/whisper-large-taiwanese-asr-v2.csv"
INFERENCE_SPLIT="test"
SAMPLE_RATE=16000

python inference.py \
    --model_name $MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --output_path $OUTPUT_PATH \
    --inference_split $INFERENCE_SPLIT \
    --sample_rate $SAMPLE_RATE