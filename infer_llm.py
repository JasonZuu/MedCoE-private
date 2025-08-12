import os
import argparse
from pathlib import Path
from vllm import LLM, SamplingParams
import os
from transformers import AutoTokenizer, AutoConfig
import time
import json
os.environ["POLARS_ALLOW_FORKING_THREAD"] = "1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

from utils.misc import set_random_seed, get_task_id, get_log_fname, get_sft_model_dir
from config.algo_config import LLMConfig
from dataset.load_fn import load_hf_dataset
from run_fn.infer_fn import infer_llm_on_dataset
from vocab.tpe_tokenizer_fast import TPETokenizerFast


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="data/hf_models/Qwen--Qwen2.5-1.5B-Instruct", help="LLM model path.")
    parser.add_argument("--data_dir", type=str, default="data/EHR_QA_v2", help="Data directory.")
    parser.add_argument("--log_dir", type=str, help="Output file path.")
    parser.add_argument("--tpe_model_dir", type=str, default="data/tpe_models", help="TPE model directory.")
    parser.add_argument("--sft_model_dir", type=str, default="data/sft_models", help="SFT model directory.")
    parser.add_argument("--dataset", type=str, default="EHRSHOT",
                        choices=['MIMICIV', "EHRSHOT"], 
                        help="Dataset to use for inference.")
    parser.add_argument("--task", type=str, default="guo_readmission", #
                        choices=["icu_mortality", "icu_phenotyping", 
                                 "guo_readmission", "new_pancan"], 
                        help="Task name.")
    parser.add_argument("--data_format", type=str, default="nl",
                        choices=["nl", "json", "yaml", "xml"], help="Data format.")
    parser.add_argument("--pe_method", type=str, default="eb", 
                        choices=["raw", "eb", 'cot', "coe", "medcoe"], help="Algorithm to use for inference.")
    parser.add_argument("--data_split", type=str, default="held_out", 
                        choices=["train", "tuning", "held_out"], help="Set name.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for inference.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument("--num_responses", type=int, default=1, help="number of responses to generate dataset.")
    parser.add_argument("--gpu_ids",  type=lambda s: [int(item) for item in s.split(',')], default=[0], help="Comma-separated list of GPU IDs to use for inference.")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of workers to use for data loading.")
    parser.add_argument("--gpu_util", type=float, default=0.9, help="Number of workers to use for data loading.")
    parser.add_argument("--max_input_len", type=str, default="8k", choices=["500", "1k", "2k", "4k", "6k", "8k", "12k"],
                        help="Max input length.")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in args.gpu_ids])
    print(f"GPU IDs: {args.gpu_ids}")
    set_random_seed(args.seed)

    log_dir = Path(args.log_dir) / args.dataset
    log_dir.mkdir(parents=True, exist_ok=True)
    log_llm_name = args.model_path.split("/")[-1].split("--")[-1]  # e.g., "Qwen-7B"
    print(f"Evaluating LLM: {log_llm_name}")
    print(f"Log dir: {log_dir}")
    print(f"data dir: {args.data_dir}")

    # check if the results already exist
    log_fname = get_log_fname(task=args.task, 
                              pe_method=args.pe_method, 
                              max_input_len=args.max_input_len, 
                              llm_name=log_llm_name,
                              data_format=args.data_format,
                              n_response=args.num_responses)
    output_fpath = log_dir / f"{log_fname}.parquet"
    print("will save to ", output_fpath)
    if output_fpath.exists():
        print(f"Results already exist at: {output_fpath}")
        exit()

    # Initialize the LLM config
    algo_config = LLMConfig()
    algo_config.llm_name = args.model_path
    algo_config.log_dir = log_dir
    if args.max_input_len == "500":
        algo_config.max_input_len = 500
    elif args.max_input_len == "1k":
        algo_config.max_input_len = 1*1024
    elif args.max_input_len == "2k":
        algo_config.max_input_len = 2*1024
    elif args.max_input_len == "4k":
        algo_config.max_input_len = 4*1024
    elif args.max_input_len == "6k":
        algo_config.max_input_len = 6*1024
    elif args.max_input_len == "8k":
        algo_config.max_input_len = 8*1024
    elif args.max_input_len == "12k":
        algo_config.max_input_len = 12*1024

    # set the max_input_len and output_token_len
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    default_config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    max_model_len = min(default_config.max_position_embeddings, tokenizer.model_max_length)
    print(f"Model's available max model len: {max_model_len}")
    if max_model_len - algo_config.max_input_len >= 4*1024:
        algo_config.output_token_len = 4*1024
        algo_config.max_model_len = algo_config.max_input_len + algo_config.output_token_len
    else:
        raise ValueError(f"Model's max model len is too small: {max_model_len}")

    print(f"max input length: {algo_config.max_input_len}")
    print(f"output token length: {algo_config.output_token_len}")

    algo_config.device = args.device

    # init the model and sampling params
    print("Model Path:", args.model_path)
    llm = LLM(model=args.model_path, 
                dtype="bfloat16", 
                max_model_len=algo_config.max_model_len,
                enforce_eager=True, 
                gpu_memory_utilization=args.gpu_util, 
                tensor_parallel_size=len(args.gpu_ids),
                trust_remote_code=True)
    
    sampling_params = SamplingParams(
        n=args.num_responses,    
        temperature=algo_config.temperature, top_p=1, top_k=-1,
        max_tokens=algo_config.output_token_len,  
    )

    # Initialize the dataset
    task_id = get_task_id(dataset=args.dataset, task=args.task,
                         data_format=args.data_format)
    data_dir = Path(args.data_dir) / task_id 
    data_files = {'train': str(data_dir / f"train.parquet"),
                  'tuning': str(data_dir / f"tuning.parquet"),
                  'held_out': str(data_dir / f"held_out.parquet"),}
    print(f"Loading dataset from: {data_dir}")
    for data_fpath in data_files.values():
        data_fpath = Path(data_fpath)
        if not data_fpath.exists():
            raise ValueError(f"Dataset file does not exist at: {data_fpath}")

    dataset = load_hf_dataset(data_files, tokenizer, 
                                data_split=args.data_split,
                                input_max_length=algo_config.max_input_len,
                                pe_method=args.pe_method, num_workers=args.num_workers)

    # run the inference
    print("Running inference...")
    running_info = {}
    start_time = time.time()
    result_df = infer_llm_on_dataset(llm, dataset, sampling_params, algo_config=algo_config)
    end_time = time.time()
    running_info['inference_time'] = end_time - start_time
    
    # Save the results
    with open(log_dir / f"running_info_{log_fname}.json", "w") as f:
        json.dump(running_info, f, indent=4)
    result_df.write_parquet(output_fpath)
 