from transformers import AutoTokenizer
import polars as pl
from datasets import Dataset, DatasetDict
from collections import defaultdict
import json


def load_multi_format_dataset(data_files):
    """
    Memory-efficient merging of multiple Parquet files
    """
    # Phase 1: Process files sequentially to reduce memory pressure
    all_datasets = {}
    
    for split, files_dict in data_files.items():
        key_to_info = defaultdict(lambda: {'data_formats': [], 'rows': {}})
        
        for data_format, path in files_dict.items():
            # Standard Polars data loading
            df = pl.read_parquet(path, columns=['meta_data', 'label', 'question', 'data'])
            df = df.with_columns(
                pl.col("meta_data").str.json_decode(),
            )
            df = df.rename({'data': f'{data_format}-data'})
            
            # Process all rows at once (more efficient for most datasets)
            for row in df.iter_rows(named=True):
                key = (
                    row['meta_data']['subject_id'],
                    row['meta_data']['hadm_id'],
                    row['meta_data']['icustay_id']
                )
                if key not in key_to_info or data_format not in key_to_info[key]['rows']:
                    key_to_info[key]['data_formats'].append(data_format)
                    key_to_info[key]['rows'][data_format] = {
                        'label': row.get('label'),
                        'question': row.get('question'),
                        f'{data_format}-data': row[f'{data_format}-data']
                    }


        # Phase 2: Stream records to avoid full list in memory
        all_data_formats = list(files_dict.keys())
        
        def generate_records():
            for key, info in key_to_info.items():
                subject_id, hadm_id, icustay_id = key
                yield {
                    'subject_id': subject_id,
                    'hadm_id': hadm_id,
                    'icustay_id': icustay_id,
                    'source_data_formats': info['data_formats'],
                    'label': next(
                        (r['label'] for r in info['rows'].values() 
                         if r and r.get('label') is not None), 
                        None
                    ),
                    'question': next(
                        (r['question'] for r in info['rows'].values() 
                         if r and r.get('question') is not None), 
                        None
                    ),
                    **{
                        f'{fmt}-data': info['rows'].get(fmt, {}).get(f'{fmt}-data')
                        for fmt in all_data_formats
                    }
                }

        # Phase 3: Create dataset from generator
        all_datasets[split] = Dataset.from_generator(generate_records)
        
        # Cleanup
        del key_to_info
    
    return DatasetDict(all_datasets)


def _turn_data_dict_to_info_constrained_str(
    sample: dict,
    max_data_tokens_length: int,
    llm_tokenizer,
    data_format: str = "nl",
    data_prefix: str = "## Data\n",
    reversed_time: bool = True
):
    """
    Advanced multi-format processor with time direction control
    
    Args:
        reversed_time: If True, processes time points from newest to oldest
    """
    # Calculate base token usage
    prefix_tokens = llm_tokenizer.encode(data_prefix, add_special_tokens=False)
    remaining_tokens = max_data_tokens_length - len(prefix_tokens)
    
    # Detect available dynamic formats
    available_formats = [
        key.split('-')[0] for key in sample.keys() 
        if '-' in key and key.endswith('-data')
    ]
    if not available_formats:
        raise ValueError("No format-specific data found in sample")
    if data_format not in available_formats:
        raise ValueError(f"Requested format {data_format} not available")
    
    # Phase 1: For each format, find maximum includable time points
    format_analyses = {}
    all_time_points = set()
    
    for fmt in available_formats:
        temporal_data = sample.get(f"{fmt}-data", "{}")
        temporal_data = json.loads(temporal_data)
        _static_sentences = temporal_data.pop("Static", [])
        _static_sentence = "".join(_static_sentences) if _static_sentences else ""
        static_tokens = llm_tokenizer.encode(_static_sentence, add_special_tokens=False) if _static_sentence else []

        if not isinstance(temporal_data, dict):
            continue
            
        # Collect all time points
        time_entries = []
        for time, content in temporal_data.items():
            if not content:
                continue
                
            content_str = content if isinstance(content, str) else "".join(content)
            tokens = llm_tokenizer.encode(content_str, add_special_tokens=False)
            time_entries.append({
                "time": time,
                "content": content_str,
                "tokens": tokens,
                "token_count": len(tokens)
            })
        
        # Sort based on reversed_time parameter
        time_entries.sort(key=lambda x: x["time"], reverse=reversed_time)
        
        # Calculate includable points
        cumulative_tokens = 0
        max_includable = 0
        
        for entry in time_entries:
            if cumulative_tokens + entry["token_count"] > remaining_tokens-len(static_tokens):
                break
            cumulative_tokens += entry["token_count"]
            max_includable += 1
            all_time_points.add(entry["time"])
        
        format_analyses[fmt] = {
            "max_includable": max_includable,
            "time_entries": time_entries
        }
    
    # Phase 2: Find minimal time range
    if not all_time_points:
        selected_times = []
    else:
        # Sort based on direction
        sorted_times = sorted(all_time_points, reverse=reversed_time)
        min_coverage = min(
            analysis["max_includable"] for analysis in format_analyses.values()
        )
        selected_times = sorted_times[:min_coverage]
    
    # Phase 3: Build data str as output
    target_data = sample.get(f"{data_format}-data", {})
    target_data = json.loads(target_data)
    static_sentences = target_data.pop("Static", [])
    static_sentence = "".join(static_sentences) if static_sentences else ""
    
    # Collect content in the desired time order
    truncated_content = []
    time_points = sorted(selected_times, reverse=False)
    # time_points = sorted(selected_times, reverse=reversed_time)
    
    for time in time_points:
        content = target_data.get(time)
        if content:
            truncated_content.append(
                content if isinstance(content, str) else "".join(content)
            )
    
    return data_prefix + static_sentence + "".join(truncated_content)


def info_constrained_sample_mapping_fn(sample, instruction, llm_tokenizer, max_input_length,
                                       data_format="nl"):
    """
    Batch processing version of sample_mapping_fn.
    Args:
        sample (dict): A batch of samples.
        instruction (str): The instruction to add to the prompt.
        llm_tokenizer: The tokenizer used to encode the data.
        max_input_length (int): The maximum input length for the model.
    Returns:
        dict: A dictionary containing the processed messages and labels.
    """
    labels = sample['label']
    question = sample['question']

    # Process prompt
    nodata_messages = [{"role": "user", "content": question + instruction}]
    nodata_tokens = llm_tokenizer.apply_chat_template(nodata_messages)
    max_data_tokens_length = max_input_length - len(nodata_tokens)
    data_str = _turn_data_dict_to_info_constrained_str(sample, max_data_tokens_length, llm_tokenizer, data_format=data_format, reversed_time=True)
    prompt = data_str + question + instruction
    message = [{"role": "user", "content": prompt}]

    return {"message": message, "label": labels}


def map_gender_to_index(gender:str):
    if gender in ["F"]:
        return 0
    elif gender in ["M"]:
        return 1
    else:
        raise ValueError(f"invalid gender: {gender}")
    

if __name__ == "__main__":
    data_files = {"held_out": {'nl': 'data/EHR_QA/MIMICIV/icu_phenotyping/nl/held_out.parquet',
                               "json": 'data/EHR_QA/MIMICIV/icu_phenotyping/nl/held_out.parquet',
                               }}
    dataset = load_multi_format_dataset(data_files)

    llm_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    dataset = dataset.map(info_constrained_sample_mapping_fn, 
                          fn_kwargs={"instruction": "Now make your prediction.",
                                    "llm_tokenizer": llm_tokenizer, "max_input_length": 16*1024-8*1024, "data_format": "nl"},)
    
    print(dataset["held_out"]["message"][2])

