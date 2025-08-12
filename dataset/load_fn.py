from datasets import load_dataset
from dataset.map_fn import sample_mapping_fn
from dataset.info_cons_format_map_fn import info_constrained_sample_mapping_fn, load_multi_format_dataset
import os


def load_hf_dataset(data_files: dict, 
                    llm_tokenizer, 
                    input_max_length, 
                    data_split:str,
                    pe_method: str = "raw", 
                    num_workers=None):
    instruction = "Now make your prediction."
    if pe_method == "raw":
        pass
    elif pe_method == "eb": # evidence based
        instruction += " Generate the answer with evidence and explanation."
    elif pe_method == "cot":
        instruction += " Think step by step."
    elif pe_method == "coe":
        instruction += " Think step-by-step and generate the answer with evidence and explanation."
    elif pe_method == "medcoe":
        prompt = \
"""Output sections (keep each within 2 lines):
1. Problem Representation - one sentence, disease-agnostic (demographics, key risk factors, tempo, defining measurement patterns).
2. Top Scripts (3-5) - for each: expected features vs contradictors from the provided data.
3. Evidence Weighting - for the top 2 scripts, state whether key findings raise (↑), are neutral (→), or lower (↓) the likelihood; name those findings.
4. Structured Reflection - one threat to the top dx and one alternative you'd consider; say what would change your mind.
5. Prediction - output JSON only: {"answer": "Your Answer”}.
Rules: 
- Use only the EHR text provided here; do not add outside facts.
- Be concise; prefer explicit measurements, trends, thresholds, and time windows.
- The final line of your response must be the JSON object and nothing else."""
        instruction += prompt
    else:
        raise ValueError(f"Invalid pe_method: {pe_method}")
    
    dataset = load_dataset(
        "parquet",
        data_files=data_files,
        columns=['label', "data", "question"],
        cache_dir=os.path.dirname(list(data_files.values())[0],),
        split=data_split
    )

    dataset = dataset.map(
        sample_mapping_fn,
        batched=True, 
        fn_kwargs={
            "instruction": instruction,
            "llm_tokenizer": llm_tokenizer,
            "max_input_length": input_max_length,
        },
        num_proc=num_workers,
        remove_columns=['data', 'question']  # Remove unnecessary columns
    )
    return dataset


def load_info_constrained_hf_dataset(data_files: dict, llm_tokenizer, input_max_length,
                                    pe_method: str = "raw", 
                                    data_format: str = "nl",
                                    num_workers=None):
    instruction = "Now make your prediction."
    if pe_method == "raw":
        pass
    elif pe_method == "cot":
        instruction += " Let's think step by step."
    else:
        raise ValueError(f"Invalid pe_method: {pe_method}")
    
    dataset = load_multi_format_dataset(data_files=data_files)

    dataset = dataset.map(
        info_constrained_sample_mapping_fn,
        fn_kwargs={
            "instruction": instruction,
            "llm_tokenizer": llm_tokenizer,
            "max_input_length": input_max_length,
            "data_format": data_format
        },
        num_proc=num_workers
    )
    return dataset

