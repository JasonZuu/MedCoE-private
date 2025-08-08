import wandb
import argparse
from functools import partial
import torch
import os

from config.sweep_config import regular_baseline_config
from run_fn.objectives import regular_baseline_objective_fn
from utils.misc import get_study_name, check_sweep_status, get_sweep_id_by_name


def get_args():
    parser = argparse.ArgumentParser(description='Train the model on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--entity", type=str, default="mingchengzhu250",
                        help="wandb entity")
    parser.add_argument('--project', type=str, default="event-llm",
                        help='wandb project name')
    parser.add_argument('--data_dir', type=str, default='data/EHR_regular', help='Data directory')
    parser.add_argument("--log_dir", type=str, default='log', help="Directory to save logs and model checkpoints.")
    parser.add_argument("--dataset", type=str, default="MIMICIV",
                        choices=['MIMICIV'], help="Dataset to use for inference.")
    parser.add_argument("--task", type=str, default="icu_phenotyping", 
                        choices=["icu_mortality", "icu_phenotyping"], help="Task name.")
    parser.add_argument("--model", type=str, default="gru", 
                        choices=["gru", 'retain'], help="Algorithm name.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device for training.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument("--time_resolution", type=str, default="1h", choices=["30m", "1h", "2h"],
                        help="Time resolution for the dataset.")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    study_name = get_study_name(
        dataset=args.dataset,
        task=args.task,
        model=args.model,
        time_resolution=args.time_resolution,
    )

    sweep_id = get_sweep_id_by_name(
        entity=args.entity,
        project=args.project,
        study_name=study_name
    )

    # If no sweep loaded, create a new Bayesian sweep
    if sweep_id is None:
        sweep_config = regular_baseline_config
        sweep_config["name"] = study_name
        sweep_id = wandb.sweep(sweep_config, project=args.project)
        print(f"No provided sweep_id. Created new Bayesian sweep_id: {sweep_id}")
    else:
        print(f"Loaded sweep_id: {sweep_id}")
    
        
    objective_fn = partial(
        regular_baseline_objective_fn,
        args=args
    )

    # Check sweep status and run agent if not finished
    sweep_status = check_sweep_status(
        entity=args.entity,
        project=args.project,
        sweep_id=sweep_id
    )

    if sweep_status == "finished":
        print(f"Sweep {sweep_id} is finished. Please check the results.")
    else:
        wandb.agent(
            sweep_id,
            project=args.project,
            function=objective_fn,
        )