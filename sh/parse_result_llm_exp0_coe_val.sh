#!/bin/bash

export PYTHONPATH="."
export HF_HOME=./hf_cache

MODEL_PATHS=(
  # "Llama-3.2-1B-Instruct"
  "Qwen2.5-1.5B-Instruct"
  "Qwen2.5-7B-Instruct"
  "Llama-3.1-8B-Instruct"
  "Meditron3-8B"
)

MAX_INPUT_LEN="4k"  # please fix to 8k for this experiment
DATASET="EHRSHOT"
if [[ "$DATASET" == "MIMICIV" ]]; then
  TASKS=("icu_phenotyping" "icu_mortality")
elif [[ "$DATASET" == "EHRSHOT" ]]; then
  TASKS=("new_pancan" "guo_readmission")
fi
PE_METHODS=("eb" "coe" "cot")
RESPONSE_DIR="./log/CoE_val"
LOG_DIR="./log/CoE_val/metric"
DATA_FORMAT="nl" 

for MODEL_PATH in "${MODEL_PATHS[@]}"; do
  for TASK in "${TASKS[@]}"; do
    for PE_METHOD in "${PE_METHODS[@]}"; do
        # Run the LLM inference in sequence
        python parse_result_llm.py \
              --pe_method $PE_METHOD \
              --data_format $DATA_FORMAT \
              --task $TASK \
              --llm_id $MODEL_PATH \
              --max_input_len $MAX_INPUT_LEN \
              --response_dir $RESPONSE_DIR \
              --dataset $DATASET \
              --log_dir $LOG_DIR 
    done
  done
done