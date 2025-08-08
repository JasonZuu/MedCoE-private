import os
import argparse
from pathlib import Path
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoConfig
os.environ["POLARS_ALLOW_FORKING_THREAD"] = "1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

from utils.misc import set_random_seed, get_task_id, get_sft_qa_fname, get_sft_dataset_dir
from config.algo_config import LLMConfig
from dataset.load_fn import load_hf_dataset
from run_fn.infer_fn import infer_llm_on_dataset


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="data/hf_models/Qwen--Qwen2.5-1.5B-Instruct", help="LLM model path.")
    parser.add_argument("--data_dir", type=str, default="data/EHR_QA", help="Data directory.")
    parser.add_argument("--output_dir", type=str, default="data/SFT_QA", help="Output file path.")
    parser.add_argument("--dataset", type=str, default="EHRSHOT",
                        choices=['MIMICIV', "EHRSHOT"], help="Dataset to use for inference.")
    parser.add_argument("--task", type=str, default="guo_readmission", #
                        choices=["icu_mortality", "icu_phenotyping", 
                                 "guo_readmission", "new_pancan"], help="Task name.")
    parser.add_argument("--data_format", type=str, default="nl",
                        choices=["nl", "json", "yaml", "xml"], help="Data format.")
    parser.add_argument("--data_split", type=str, default="held_out", 
                        choices=["train", "tuning", "held_out"], help="Set name.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for inference.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument("--num_responses", type=int, default=1, help="number of responses to generate.")
    parser.add_argument("--gpu_ids",  type=lambda s: [int(item) for item in s.split(',')], default=[1], help="Comma-separated list of GPU IDs to use for inference.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers to use for data loading.")
    parser.add_argument("--gpu_util", type=float, default=0.6, help="Number of workers to use for data loading.")
    parser.add_argument("--max_input_len", type=str, default="6k", choices=["2k", "4k", "6k", "8k"],
                        help="Max input length.")
    parser.add_argument("--max_output_len", type=str, default="2k", choices=["2k", "4k"],
                        help="Max output length.")
    return parser.parse_args()



if __name__ == "__main__":
    args = get_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in args.gpu_ids])
    print(f"GPU IDs: {args.gpu_ids}")
    set_random_seed(args.seed)

    _dataset_dir = get_sft_dataset_dir(
        dataset=args.dataset, 
        max_input_len=args.max_input_len, 
        max_output_len=args.max_output_len
    )
    output_dir = Path(args.output_dir) / _dataset_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    log_llm_name = args.model_path.split("/")[-1].split("--")[-1]  # e.g., "Qwen-7B"
    print(f"Evaluating LLM: {log_llm_name}")
    print(f"Log dir: {output_dir}")
    print(f"data dir: {args.data_dir}")

    # check if the results already exist
    sft_qa_fname = get_sft_qa_fname(task=args.task,
                                    data_format=args.data_format, 
                                    data_split=args.data_split, 
                                    llm_name=log_llm_name)
    output_fpath = output_dir / (sft_qa_fname + ".parquet")
    print("will save to ", output_fpath)
    if output_fpath.exists():
        print(f"Results already exist at: {output_fpath}")
        exit()

    # Initialize the LLM config
    algo_config = LLMConfig()
    algo_config.llm_name = args.model_path
    algo_config.log_dir = output_dir
    algo_config.data_split = args.data_split

    # set the max_input_len and output_token_len
    if args.max_input_len == "2k":
        algo_config.max_input_len = 2*1024
    elif args.max_input_len == "4k":
        algo_config.max_input_len = 4*1024
    elif args.max_input_len == "6k":
        algo_config.max_input_len = 6*1024
    elif args.max_input_len == "8k":
        algo_config.max_input_len = 8*1024
    
    if args.max_output_len == "2k":
        algo_config.output_token_len = 2*1024
    elif args.max_output_len == "4k":
        algo_config.output_token_len = 4*1024
    algo_config.max_model_len = algo_config.max_input_len + algo_config.output_token_len

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    default_config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    max_model_len = min(default_config.max_position_embeddings, tokenizer.model_max_length)
    print(f"Model's available max model len: {max_model_len}")
    print(f"max input length: {algo_config.max_input_len}")
    print(f"output token length: {algo_config.output_token_len}")
    assert algo_config.max_model_len <= max_model_len, \
        f"Max model length {algo_config.max_model_len} exceeds model's max position embeddings {max_model_len}. " 
    algo_config.device = args.device

    # init the model and sampling params
    llm =LLM(model=algo_config.llm_name, dtype="bfloat16", max_model_len=algo_config.max_model_len,
                enforce_eager=True, gpu_memory_utilization=args.gpu_util, tensor_parallel_size=len(args.gpu_ids), 
                trust_remote_code=True)
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        n=args.num_responses,    
        temperature=algo_config.temperature, top_p=1, top_k=-1,
        max_tokens=algo_config.output_token_len, 
        skip_special_tokens=True, # remove special tokens from the output
    )

    # Initialize the dataset
    task_id = get_task_id(dataset=args.dataset, task=args.task,
                         data_format=args.data_format)
    data_dir = Path(args.data_dir) / task_id 
    data_files = {'train': str(data_dir / "train.parquet"),
                  'tuning': str(data_dir / "tuning.parquet"),
                  'held_out': str(data_dir / "held_out.parquet"),}
    print(f"Loading dataset from: {data_dir}")
    for data_fpath in data_files.values():
        data_fpath = Path(data_fpath)
        if not data_fpath.exists():
            raise ValueError(f"Dataset file does not exist at: {data_fpath}")

    dataset = load_hf_dataset(data_files, tokenizer, 
                                input_max_length=algo_config.max_input_len,
                                data_split=args.data_split,
                                pe_method="raw", num_workers=args.num_workers)
    # run the inference
    result_df = infer_llm_on_dataset(llm, dataset, sampling_params, algo_config=algo_config, n_samples=None)
    result_df.write_parquet(output_fpath)
 