import argparse
from pathlib import Path
import json

from utils.misc import set_random_seed, get_log_fname
from run_fn.test_fn import llm_test_fn


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_id", type=str, default='Qwen2.5-7B-Instruct',
                        choices=['Qwen2.5-7B-Instruct', 
                                 'Meditron3-Qwen2.5-7B', 
                                 'DeepSeek-R1-Distill-Qwen-7B',
                                 'Llama-3.1-8B-Instruct', "Meditron3-8B", 
                                 "DeepSeek-R1-Distill-Llama-8B",
                                 "Qwen2.5-1.5B-Instruct", 
                                 "DeepSeek-R1-Distill-Qwen-1.5B",
                                 "Llama-3.2-1B-Instruct",
                                 ],
                        help="LLM model ID.")
    parser.add_argument("--response_dir", type=str, default="log/TPE/EHRSHOT-info_cons_vocab",
                        help="LLM Output directory.")
    parser.add_argument("--log_dir", type=str, default="log/TPE/metrics", help="Output file path.")
    parser.add_argument("--dataset", type=str, default="EHRSHOT",
                        choices=['MIMICIV', "EHRSHOT"],
                        help="Dataset to use for inference.")
    parser.add_argument("--task", type=str, default="guo_readmission",
                        choices=["icu_mortality", "icu_phenotyping", 
                                 "guo_readmission", "new_pancan"],
                        help="Task name.")
    parser.add_argument("--data_format", type=str, default="nl",
                        choices=["nl", "json", "yaml", "xml"],
                        help="Data format.")
    parser.add_argument("--pe_method", type=str, default="cot",
                        choices=["raw", "eb", 'cot', "coe", "medcoe"], help="Algorithm to use for inference.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument("--max_input_len", type=str, default="8k",
                        choices=["500", "1k", "2k", "4k", "6k", "8k", "12k", "16k", "24k"],
                        help="Max input length.")
    parser.add_argument("--n_bootstrap", type=int, default=1000,
                        help="Number of bootstrap times.")
    parser.add_argument("--avg_method", type=str, default="macro",
                        choices=["binary", "macro", "micro"],
                        help="Method to use for averaging.")
    parser.add_argument('--n_response', type=int, default=1,
                        help="Number of responses to consider for evaluation.")
    parser.add_argument('--task_specific_pe', action='store_true',
                        help="Whether task-specific prompt engineering was used.")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    set_random_seed(args.seed)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # deal with the log file name.
    log_fname = get_log_fname(task=args.task,
                             data_format=args.data_format,
                             max_input_len=args.max_input_len,
                             llm_name=args.llm_id,
                             pe_method=args.pe_method,
                             n_response=args.n_response,
                             task_specific_pe=args.task_specific_pe)
    
    # check if the results already exist
    output_fpath = log_dir /f"{log_fname}.json"
    print("will save to ", output_fpath)
    if output_fpath.exists():
        print(f"Results already exist at: {output_fpath}")
        exit()

    # Check if the LLM output file exists. If not, there is no data to evaluate.
    response_file = Path(args.response_dir)/ args.dataset /f"{log_fname}.parquet"
    running_info_file = Path(args.response_dir)/ args.dataset /f"running_info_{log_fname}.json"
    if not response_file.exists():
        print(f"Results do not exist at: {response_file}")
        exit()

    # Determine if the task is multi-class
    if args.task == "icu_mortality":
        is_multi_label = False
        num_classes = 2
    elif args.task == "icu_phenotyping":
        is_multi_label = True
        num_classes = 25
    elif args.task == "guo_readmission":
        is_multi_label = False
        num_classes = 2
    elif args.task == "new_pancan":
        is_multi_label = False
        num_classes = 2
    else:
        raise ValueError(f"Unknown task: {args.task}")

    # Run evaluation
    test_metrics = llm_test_fn(str(response_file), 
                               num_classes=num_classes, 
                               n_response=args.n_response,
                               avg_method=args.avg_method, 
                               n_bootstrap=args.n_bootstrap,
                               random_seed=args.seed, 
                               use_bootstrap=True,
                               is_multi_label=is_multi_label)
   
    # add metadata
    test_metrics["metadata"]["task"] = args.task
    test_metrics["metadata"]["data_format"] = args.data_format
    test_metrics["metadata"]["pe_method"] = args.pe_method
    test_metrics["metadata"]["max_input_len"] = args.max_input_len
    test_metrics["metadata"]["llm_id"] = args.llm_id
    test_metrics["metadata"]["seed"] = args.seed

    # form the recorded results
    test_metrics["results"] = {}
    f1_mean = test_metrics["bootstrap_metrics"]["f1"]["mean"]
    f1_std = test_metrics["bootstrap_metrics"]["f1"]["std"]
    fcr = test_metrics["metadata"]["valid_ratio"]
    test_metrics["results"]["f1"] = f"{f1_mean:.3f} ({f1_std:.3f})"
    test_metrics["results"]["FCR"] = f"{fcr:.3f}"

    # add running info if available, care inference time
    if running_info_file.exists():
        with open(running_info_file, 'r') as f:
            running_info = json.load(f)
        infer_time_min = running_info["inference_time"]/60
        test_metrics["results"]["inference_time"] = f"{infer_time_min:.3f} min"
    
    # Save metrics as JSON
    with open(output_fpath, 'w') as f:
        json.dump(test_metrics, f, indent=4)
