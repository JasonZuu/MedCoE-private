export PYTHONPATH="."
export HF_DATASETS_CACHE=./hf_cache

MODEL_PATHS=(
  # "data/hf_models/meta-llama--Llama-3.2-1B-Instruct"
  # "data/hf_models/Qwen--Qwen2.5-1.5B-Instruct"
  "data/hf_models/Qwen--Qwen2.5-7B-Instruct"
  "data/hf_models/meta-llama--Llama-3.1-8B-Instruct"
)

DATASET="EHRSHOT"
if [[ "$DATASET" == "MIMICIV" ]]; then
  TASKS=("icu_phenotyping" "icu_mortality")
elif [[ "$DATASET" == "EHRSHOT" ]]; then
  TASKS=("new_pancan" "guo_readmission")
fi

GPU_UTIL=0.9
NUM_RESPONSES=1
DATA_FORMAT="nl"
PE_METHODS=("raw" "eb" "coe" "cot")
LOG_DIR="./log/CoE_val"
MAX_INPUT_LEN="4k"

for MODEL_PATH in "${MODEL_PATHS[@]}"; do
  for TASK in "${TASKS[@]}"; do
    for PE_METHOD in "${PE_METHODS[@]}"; do
      echo "Running inference for model: $MODEL_PATH, task: $TASK, PE method: $PE_METHOD"
      python infer_llm.py \
                --pe_method $PE_METHOD \
                --gpu_util $GPU_UTIL \
                --data_format $DATA_FORMAT\
                --task $TASK \
                --gpu_id 0 \
                --dataset $DATASET \
                --num_responses $NUM_RESPONSES \
                --model_path $MODEL_PATH \
                --max_input_len $MAX_INPUT_LEN \
                --log_dir $LOG_DIR 
    done
  done
done

