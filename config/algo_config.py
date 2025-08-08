from dataclasses import dataclass


@dataclass
class LLMConfig:
    log_dir = 'log/'
    device = 'cuda'
    llm_name = 'meta-llama/Llama-3.1-8B-Instruct'
    max_model_len = None # depends on the model
    max_input_len = 4*1024  # 24k
    output_token_len = 8*1024  # 8k
    temperature=0.7


@dataclass
class RegularAlgoConfig:
    log_dir = 'log/regular'
    model_name = None
    study_name = None
    device = 'cuda'
    avg_method = "micro"
    task = "icu_mortality"

    max_num_epochs = 100
    lr_decay_steps = 1
    lr_decay_gamma = 0.95
    lr_warmup_epochs = 10
    early_stop_epochs = 10
    task_id = None
    select_metric = "loss"
    select_metric_direction = "minimize"
    
    # hyper-parameters
    batch_size = 128
    lr = 1e-4
    