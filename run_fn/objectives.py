import wandb
from pathlib import Path
from torch.utils.data import DataLoader

from config.algo_config import RegularAlgoConfig
from config.data_config import RegularMIMICConfig
from utils.misc import set_random_seed, get_task_id
from dataset.regular_dataset import RegularSampleDataset
from models.gru import GRUBasedModel
from models.retain import RETAINBasedModel
from run_fn.regular_fn import regular_train_fn
from run_fn.test_fn import regular_test_fn


def regular_baseline_objective_fn(args):
    """
    Objective function for the baseline model training and evaluation.
    args: Command line arguments containing configuration parameters. Input using partial
    """
    set_random_seed(args.seed)

    # config dataset_config
    dataset_config = RegularMIMICConfig()
    task_id = get_task_id(dataset=args.dataset, task=args.task)
    dataset_config.data_dir = f"{args.data_dir}/{task_id}-{args.time_resolution}"
    dataset_config.time_resolution = args.time_resolution
    dataset_config.task = args.task
    if args.task == "icu_mortality":
        dataset_config.num_classes = 2
        is_multi_label = False
    elif args.task == "icu_phenotyping":
        dataset_config.num_classes = 25
        is_multi_label = True
    else:
        raise ValueError(f"Invalid task name: {args.task}")

    # load datasets
    train_dataset = RegularSampleDataset(dataset_config, set_name='train')
    tuning_dataset = RegularSampleDataset(dataset_config, set_name='tuning')
    held_out_dataset = RegularSampleDataset(dataset_config, set_name='held_out')

    # # set config
    algo_config = RegularAlgoConfig()
    algo_config.task = args.task
    algo_config.entity = args.entity
    algo_config.project = args.project
    algo_config.task_id = task_id
    algo_config.model_name = args.model

    # init model    
    if args.model == "gru":
        model = GRUBasedModel(
            num_classes=dataset_config.num_classes,
            demo_dims=train_dataset.demo_dims,
            ts_dims=train_dataset.ts_dims
        )
    elif args.model == "retain":
        model = RETAINBasedModel(
            num_classes=dataset_config.num_classes,
            demo_dims=train_dataset.demo_dims,
            ts_dims=train_dataset.ts_dims
        )
    else:
        raise ValueError(f"Invalid algorithm name: {args.model}")
    model.to(args.device)

    
    with wandb.init(project=args.project, entity=args.entity) as run:
        algo_config.learning_rate = run.config.learning_rate
        algo_config.batch_size = run.config.batch_size
    
        # Train model
        val_result = regular_train_fn(
            algo_config, 
            model, 
            train_dataset, 
            tuning_dataset, 
            write_log=False
        )

        # test model
        held_out_loader = DataLoader(
            held_out_dataset, 
            batch_size=1024, 
            shuffle=False, 
            drop_last=False
        )
        test_result = regular_test_fn(
            model, 
            held_out_loader, 
            args.device, 
            avg_method=algo_config.avg_method,
            is_multi_label=is_multi_label, 
            use_bootstrap=False
        )
        
        # log results
        val_result = {f"val_{k}": v for k, v in val_result.items()}
        test_result = {f"test_{k}": v for k, v in test_result.items()}
        run.log(val_result)
        run.log(test_result)


# How to run the sweep:
# sweep_id = wandb.sweep(sweep_config, project="your_project")
# wandb.agent(sweep_id, function=baseline_objective)