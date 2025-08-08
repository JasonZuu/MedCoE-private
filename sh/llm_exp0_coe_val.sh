export PYTHONPATH="."
export HF_DATASETS_CACHE=./hf_cache

MODEL_PATHS=(
  "data/hf_models/Qwen--Qwen2.5-7B-Instruct"
  "data/hf_models/meta-llama--Llama-3.1-8B-Instruct"
  "data/hf_models/OpenMeditron--Meditron3-8B"
)

DATASET="MIMICIV"
if [[ "$DATASET" == "MIMICIV" ]]; then
  TASKS=("icu_phenotyping" "icu_mortality")
elif [[ "$DATASET" == "EHRSHOT" ]]; then
  TASKS=("new_pancan" "guo_readmission")
fi

GPU_UTIL=0.93
NUM_RESPONSES=1
DATA_FORMAT="nl"
PE_METHODS=("cot" "coe")
LOG_DIR="./log/CoE_val"
MAX_INPUT_LEN="8k"
DATA_FORMAT="nl"  

for MODEL_PATH in "${MODEL_PATHS[@]}"; do
  for TASK in "${TASKS[@]}"; do
    for PE_METHOD in "${PE_METHODS[@]}"; do
      # Run the LLM inference in sequence
      python info_cons_infer_llm.py \
                --pe_method $PE_METHOD \
                --gpu_util $GPU_UTIL \
                --data_format $DATA_FORMAT\
                --task $TASK \
                --gpu_id 1 \
                --num_responses $NUM_RESPONSES \
                --model_path $MODEL_PATH \
                --max_input_len $MAX_INPUT_LEN \
                --log_dir $LOG_DIR 
    done
  done
done

