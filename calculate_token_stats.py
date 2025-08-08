import polars as pl
from transformers import AutoTokenizer
from pathlib import Path
import json

from run_fn.token_stats import calc_time_token_stats
from vocab import TPETokenizerFast

dataset = "MIMICIV"
if dataset == "MIMICIV":
    tasks = ["icu_phenotyping", "icu_mortality"]
elif dataset == "EHRSHOT":
    tasks = ["guo_readmission", "new_pancan"]

sft_model_dir = "data/tpe_tokenizers"
model_fps   = ["data/hf_models/Qwen--Qwen2.5-1.5B-Instruct", 
               "data/hf_models/meta-llama--Llama-3.2-1B-Instruct"]
tokeniser_types = ["tpe-sft", "bpe"]
data_split = "held_out"
max_n_list = [2, 3, 4, 5]
max_m_list = [1000, 2000, 5000, 10000]

for task in tasks:
    parquet_fp = f"data/EHR_QA/{dataset}/{task}/nl/{data_split}.parquet"
    out_dir    = Path(f"log/token_stats/{dataset}_{data_split}_{task}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for model_fp in model_fps:
        for tokeniser_type in tokeniser_types:
            print(f"Processing model: {model_fp}")
            model_name = model_fp.split("/")[-1].split("--")[-1]
            df  = pl.read_parquet(parquet_fp)
            if tokeniser_type == "tpe-sft":
                for max_n in max_n_list:
                    for max_m in max_m_list:
                        log_fp = out_dir / f"{model_name}_token-stats_{tokeniser_type}_maxN-{max_n}_maxM-{max_m}.json"
                        _model_fp = f"{sft_model_dir}/{model_name}_task-{task}_maxN-{max_n}_maxM-{max_m}"
                        tok = TPETokenizerFast.from_pretrained(_model_fp)

                        # check if the log file already exists
                        if log_fp.exists():
                            print(f"Log file {log_fp} already exists, skipping.")
                            continue

                        # if not exists, calculate token stats
                        final_stats = calc_time_token_stats(
                            df, tok, instruction="", data_prefix="## Data\n"
                        )
                        with open(log_fp, "w") as f:
                            json.dump(final_stats, f, indent=2)

            elif tokeniser_type == "bpe":
                log_fp = out_dir / f"{model_name}_token-stats_{tokeniser_type}.json"
                tok = AutoTokenizer.from_pretrained(model_fp)

                # check if the log file already exists
                if log_fp.exists():
                    print(f"Log file {log_fp} already exists, skipping.")
                    continue
                
                # calculate token stats
                final_stats = calc_time_token_stats(
                    df, tok, instruction="", data_prefix="## Data\n"
                )

                # log
                with open(log_fp, "w") as f:
                    json.dump(final_stats, f, indent=2)
