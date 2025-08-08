import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb
import torch
from torch import optim
from tqdm import tqdm
from pathlib import Path

from config.algo_config import RegularAlgoConfig
from models.gru import GRUBasedModel
from run_fn.test_fn import regular_test_fn
from models.tracker import PerformanceTracker


def regular_train_fn(config: RegularAlgoConfig,
                      model: GRUBasedModel,
                      train_dataset: torch.utils.data.Dataset,
                      tuning_dataset: torch.utils.data.Dataset,
                      write_log: bool = True):
    optimizer = optim.Adam(model.parameters(), config.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=config.lr_decay_steps,
                                                gamma=config.lr_decay_gamma)

    tracker = PerformanceTracker(early_stop_epochs=config.early_stop_epochs,
                                 metric=config.select_metric, 
                                 direction=config.select_metric_direction)
    if write_log:
        run = wandb.init(project=config.project, group=config.task_id, name=f"{config.task_id}//{config.model_name}")
    else:
        run = None

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    tuning_loader = DataLoader(tuning_dataset, batch_size=1024, shuffle=False, drop_last=False)

    for i_epoch in range(config.max_num_epochs):
        _regular_train_loop(model=model, train_loader=train_loader, optimizer=optimizer,
                             device=config.device, run=run, i_epoch=i_epoch)
        if (i_epoch + 1) > config.lr_warmup_epochs:
            scheduler.step()
        
        # calculate validation metric
        if config.task == "icu_mortality":
             val_metric = regular_test_fn(model=model, test_loader=tuning_loader, device=config.device, avg_method=config.avg_method,
                                          is_multi_label=False)
        elif config.task == "icu_phenotyping":
            val_metric = regular_test_fn(model=model, test_loader=tuning_loader, device=config.device, avg_method=config.avg_method,
                                          is_multi_label=True)

        if write_log:
            val_metric["epoch"] = i_epoch
            run.log(val_metric)
        state_dict = {"model": model.state_dict()}
        early_stop_flag = tracker.update(val_metric, state_dict)
        if early_stop_flag:
            break

    best_model_state_dict = tracker.export_best_model_state_dict()
    best_val_metric_dict = tracker.export_best_metric_dict()
    model.load_state_dict(best_model_state_dict["model"])

    if write_log:
        torch.save(best_model_state_dict, config.log_dir / "model.pth")
        run.finish()

    return best_val_metric_dict


def _regular_train_loop(model, train_loader, optimizer, device:str,
                         run, i_epoch):
    """
    function to train the net
    """
    model.train()
    i_step = i_epoch*len(train_loader)
    pbar = tqdm(total=len(train_loader), desc=f'Regular Baseline Training ({i_epoch+1} epoch)', unit='batch')

    for data in train_loader:
        demo, ts, label = data['demo'], data['ts'], data['label']
        demo, ts, label = demo.to(device), ts.to(device), label.to(device)

        optimizer.zero_grad()
        y_scores = model(demo, ts)

        # losses
        loss = F.cross_entropy(y_scores, label)
        loss.backward()
        optimizer.step()

        # display loss
        pbar.set_postfix(**{'loss(batch)': loss.item()})

        lr = optimizer.param_groups[0]['lr']
        log_dict = {"lr": lr, "loss": loss.item()}

        if run is not None:
            run.log(log_dict, step=i_step)

        i_step += 1
        pbar.update()
