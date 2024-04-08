MODEL_NAME_OR_PATH="openai/whisper-tiny"
DATASET_NAME="mozilla-foundation/common_voice_11_0"
DATASET_CONFIG_NAME="zh-TW"
LANGUAGE="chinese"
TRAIN_SPLIT_NAME="train+validation"
EVAL_SPLIT_NAME="test"
MODEL_INDEX_NAME=Whisper_Finetuning
MAX_STEPS="5000"
PER_DEVICE_TRAIN_BATCH_SIZE="8"
PER_DEVICE_EVAL_BATCH_SIZE="1"
LEARNING_RATE="1e-5"
WARMUP_STEPS="500"
GENERATION_MAX_LENGTH="225"

python run_speech_recognition_seq2seq_streaming.py \
	--model_index_name $MODEL_INDEX_NAME \
	--model_name_or_path $MODEL_NAME_OR_PATH --dataset_name $DATASET_NAME --dataset_config_name $DATASET_CONFIG_NAME --language $LANGUAGE \
	--train_split_name $TRAIN_SPLIT_NAME --eval_split_name $EVAL_SPLIT_NAME \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE --per_device_eval_batch_size $PER_DEVICE_EVAL_BATCH_SIZE \
    --max_steps $MAX_STEPS --learning_rate $LEARNING_RATE --warmup_steps $WARMUP_STEPS --generation_max_length $GENERATION_MAX_LENGTH \
	--output_dir="./runs" \
	--logging_steps="25" \
	--evaluation_strategy="steps" \
	--eval_steps="1000" \
	--save_strategy="steps" \
	--save_steps="1000" \
	--length_column_name="input_length" \
	--max_duration_in_seconds="30" \
	--text_column_name="sentence" \
	--freeze_feature_encoder="False" \
	--report_to="tensorboard" \
	--metric_for_best_model="wer" \
	--greater_is_better="False" \
	--load_best_model_at_end \
	--fp16 \
	--overwrite_output_dir \
	--do_train \
	--do_eval \
	--predict_with_generate \
	--do_normalize_eval \
	--use_auth_token \
	# --gradient_checkpointing \
	# --streaming \
	# --push_to_hub