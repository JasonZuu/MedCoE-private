import polars as pl
from torch.utils import data
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path
import json


class BaseEventDataset(data.Dataset):
    def __init__(self, llm_tokenizer, parquet_fpath, 
                 max_token_length=32*1024, output_token_length=2*1024,):
        """
        Args:
            llm_tokenizer (class): The tokenizer object to be used for tokenization.
            parquet_fpath (str): Path to the Parquet file containing the data.
            instruction (str, optional): Additional instruction to be appended to the prompt.
            max_token_length (int, optional): Maximum token length of the model.
            output_token_length (int, optional): Maximum token length of the output. The token length is reserved for the output.
        """
        self.llm_tokenizer = llm_tokenizer
        self.parquet_fpath = parquet_fpath
        self.instruction = None
        self.max_input_length = max_token_length - output_token_length
        self.data_prefix = "## Data\n"
        self.data_prefix_tokens = self.llm_tokenizer.tokenize(self.data_prefix)
        self.data_df = pl.read_parquet(self.parquet_fpath)

        # get times, read from the parquet
        datas = self.data_df.get_column("data") # list of dict
        if len(datas) == 0:
            raise ValueError("The data is empty. Please check the Parquet file.")
        data_dicts = [json.loads(data_str) for data_str in datas]
        # get all times from all data
        all_times = set()
        for data_dict in data_dicts:
            for time in data_dict.keys():
                all_times.add(time)
        # sort times
        self.all_times = sorted(list(all_times))
        # self.all_times = [f"Day0 {i:02d}Hour {j:02d}Minute" for i in range(24) for j in range(60)] + ["Day1 00Hour 00Minute"]

        self.chat_message = []

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data_df)

    def __getitem__(self, idx):
        """Return the prompt and label for the sample at the given index."""
        assert self.instruction is not None, "Not getting any instructions."
        row = self.data_df.row(idx, named=True)
        label = row['label']
        data = row['data']
        question = row['question']

        # process prompt
        nodata_messages = self.chat_message + [{"role": "user", "content": question+self.instruction}]
        nodata_tokens = self.llm_tokenizer.apply_chat_template(nodata_messages)
        max_data_tokens_length = self.max_input_length - len(nodata_tokens)
        data_str = self._turn_data_dict_to_str(data, max_data_tokens_length)
        prompt = data_str + question + self.instruction
        message = self.chat_message + [{"role": "user", "content": prompt}]
        return message, label
    
    def _turn_data_dict_to_tokens(self, data_dict:dict, max_data_tokens_length:int):
        """
        Convert a dictionary to tokens. 
        Warning: This is not the default behavior to get the data. This is only used for the calculation of token lengths of the whole data.

        Args:
            data_dict (dict): The dictionary to be converted to tokens.
            max_data_tokens_length (int): The maximum available data tokens length.

        Returns:
            list: The list of tokens.
        """
        # Extract the static events
        static_value = data_dict.pop("Static")
        if static_value is not None:
            static_sentence = "".join(static_value)
            static_tokens = self.llm_tokenizer.tokenize(static_sentence)
        else:
            static_tokens = []
        
        max_data_tokens_length -= len(static_tokens) + len(self.data_prefix_tokens)

        tokens = []
        for time, str_events in data_dict.items():
            if str_events is None:
                continue
            time_sentence = "".join(str_events)
            time_tokens = self.llm_tokenizer.tokenize(time_sentence)
            if len(tokens) + len(time_tokens) > max_data_tokens_length:
                break
            # Extend time_tokens at the beginning of the tokens list
            tokens = time_tokens + tokens
        return self.data_prefix_tokens + static_tokens + tokens
    
    def _turn_data_dict_to_str(self, data_dict:dict, max_data_tokens_length:int):
        """
        Convert a dictionary to a string. 
        This function will calculate the maximum length of the data and return the string representation of the data.
        Args:
            data_dict (dict): The dictionary to be converted to a string.
            max_data_tokens_length (int): The maximum available data tokens length.
        Returns:
            str: The string representation of the dictionary.
        """
        # Extract the static events
        static_sentences = data_dict.pop("Static")
        if static_sentences is not None:
            static_sentence = "".join(static_sentences)
            static_tokens = self.llm_tokenizer.tokenize(static_sentence)
        else:
            static_tokens = []
        
        max_data_tokens_length -= len(static_tokens) + len(self.data_prefix_tokens)

        sentences = []
        tokens = []

        for time, str_events in data_dict.items():
            if str_events is None:
                continue
            time_sentence = "".join(str_events)
            time_tokens = self.llm_tokenizer.tokenize(time_sentence)
            if len(tokens) + len(time_tokens) > max_data_tokens_length:
                break
            # Extend time_tokens at the beginning of the tokens list
            tokens = time_tokens + tokens
            sentences = str_events + sentences
        
        sentences = [self.data_prefix, static_sentence] + sentences
        return "".join(sentences)
    
    def calculate_time2event_accumulation(self):
        result_dict = defaultdict(list)
        raw_datas = self.data_df.get_column("data")

        pbar = tqdm(total=len(raw_datas), desc="Calculating time-event accumulation")
        for raw_data_str in raw_datas:
            raw_data_dict = json.loads(raw_data_str)
            event_accumulation = 0
            time2event_accum = {}
            
            for time in self.all_times:
                events = raw_data_dict.get(time, None)
                if events is not None:
                    event_accumulation += len(events)
                time2event_accum[time] = event_accumulation

            for time, event_accum in time2event_accum.items():
                result_dict[time].append(event_accum)
            pbar.update(1)

        pbar.close()
        result_dict_stats = {}
        for time, event_accum in result_dict.items():
            result_dict_stats[time] = {"mean": np.mean(event_accum), "std": np.std(event_accum), "median": np.median(event_accum),
                                       "min": np.min(event_accum), "max": np.max(event_accum), "3quartile": np.percentile(event_accum, 75),
                                       "1quartile": np.percentile(event_accum, 25)}
        rows = []
        for time, stats in result_dict_stats.items():
            row = {"time": time}  # Add the time column
            row.update(stats)     # Add the statistics columns
            rows.append(row)
        stats_df = pl.DataFrame(rows)
        stats_df = stats_df.sort("time")
        return stats_df, result_dict
    
    def calculate_time2token_accumulation(self):
        times          = self.all_times               # 假设已排序、长度 n_times
        n_times        = len(times)
        n_rows         = len(self.data_df)
        data_col       = self.data_df.get_column("data").to_list()

        # 1⃣ 事件级 token 缓存
        token_len_cache: dict[str, int] = {}

        # 2⃣ 预分配二维矩阵：shape = (n_rows, n_times), dtype 可选 int32/64
        accum_mat = np.zeros((n_rows, n_times), dtype=np.int32)

        pbar = tqdm(total=n_rows, desc="Calculating time-token accumulation")

        for row_idx, data_str in enumerate(data_col):
            data_dict   = json.loads(data_str)        # O(#events) 而非 O(#times)
            row_counts  = np.zeros(n_times, dtype=np.int32)

            # ⽤一次 list-lookup 替代内层 1 000 次 dict.get
            # 如果 times 很多，可先把 data_dict 转成 dict->array 更快
            for col_idx, t in enumerate(times):
                events = data_dict.get(t)
                if events:
                    total_len = 0
                    for ev in events:
                        cached = token_len_cache.get(ev)
                        if cached is None:
                            cached = len(self.llm_tokenizer.tokenize(ev))
                            token_len_cache[ev] = cached
                        total_len += cached
                    row_counts[col_idx] = total_len

            accum_mat[row_idx] = np.cumsum(row_counts, dtype=np.int32)
            pbar.update(1)

        pbar.close()

        # 3⃣ 用 NumPy 一次性生成统计量
        stats = {
            "mean":       accum_mat.mean(axis=0),
            "std":        accum_mat.std(axis=0),
            "median":     np.median(accum_mat, axis=0),
            "min":        accum_mat.min(axis=0),
            "max":        accum_mat.max(axis=0),
            "1quartile":  np.percentile(accum_mat, 25, axis=0),
            "3quartile":  np.percentile(accum_mat, 75, axis=0),
        }

        # 4⃣ 拼成 Polars DataFrame（按需要也可直接返回 NumPy）
        rows = [
            {"time": t, **{k: v[i] for k, v in stats.items()}}
            for i, t in enumerate(times)
        ]
        stats_df = pl.DataFrame(rows).sort("time")
        return stats_df, accum_mat   

    
class ICU_Mortality_Dataset(BaseEventDataset):
    def __init__(self, llm_tokenizer, parquet_fpath, max_length=32*1024, 
                 output_token_length=2*1024, pe_method="raw"):
        super().__init__(llm_tokenizer, parquet_fpath, max_length, output_token_length)
        self.pe_method = pe_method
        self.instruction = "Now make your prediction."
        if self.pe_method == "raw":
            pass # do nothing
        elif self.pe_method == "cot":
            self.instruction += " Let's think step by step."
        else:
            raise ValueError("Invalid positional encoding method. Please choose from 'raw' or 'cot'.")


class ICU_Phenotyping_Dataset(BaseEventDataset):
    def __init__(self, llm_tokenizer, parquet_fpath, max_length=32*1024, 
                 output_token_length=2*1024, pe_method="raw"):
        super().__init__(llm_tokenizer, parquet_fpath, max_length, output_token_length)
        self.pe_method = pe_method
        self.instruction = "Now make your prediction."
        if self.pe_method == "raw":
            pass # do nothing
        elif self.pe_method == "cot":
            self.instruction += " Let's think step by step."
        else:
            raise ValueError("Invalid positional encoding method. Please choose from 'raw' or 'cot'.")


if __name__ == "__main__":
    from vocab.tpe_tokenizer_fast import TPETokenizerFast
    from create_tpe_tokenizer import get_tpe_tokenizer_dir

    tokenizer_ids = ["orig"]
    dataset_id = "EHRSHOT"
    log_dir = f"log/stats/{dataset_id}"
    task = "guo_readmission"
    data_format = "nl"
    # max_m_list = [100, 500, 1000, 2000, 4000, 5000, 8000, 10000, 20000]
    # max_m_list = [100, 1000, 2000, 4000, 5000, 10000, 20000]
    max_n_list = [2, 3, 4, 5]
    max_m_list = [1000, 2000, 5000, 10000]
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    model_paths = ["data/hf_models/Qwen--Qwen2.5-1.5B-Instruct",
                   "data/hf_models/meta-llama--Llama-3.2-1B-Instruct"]
    for model_path in model_paths:
        model_name = model_path.split("/")[-1].split("--")[-1]  # e.g., "Qwen-7B"
        print(f"Processing model: {model_name}")
        tpe_tokenizers_dir = "data/tpe_tokenizers"
        fparquet_fpath = f'data/EHR_QA/{dataset_id}/{task}/{data_format}/all.parquet'

        for tokenizer_type in tokenizer_ids:
            if tokenizer_type == "orig":
                token_stats_log_fpath = log_dir / f"{model_name}_{tokenizer_type}_{task}_{data_format}_token_stats.csv"
                if token_stats_log_fpath.exists():
                    print(f"{model_name} {tokenizer_type} {task} {data_format} already exists. Skip.")
                    continue

                llm_tokenizer = AutoTokenizer.from_pretrained(model_path)
                dataset = ICU_Mortality_Dataset(llm_tokenizer=llm_tokenizer, 
                                                parquet_fpath=fparquet_fpath, 
                                                pe_method="raw")
            
                # Calculate the time-event and time-token accumulation
                token_accum_stats_df, token_accum_results = dataset.calculate_time2token_accumulation()
            
                # Save the results
                token_accum_stats_df.write_csv(token_stats_log_fpath)
                print(f"{data_format} results saved successfully.")

            else:
                for max_n in max_n_list:
                    for max_m in max_m_list:
                        token_stats_log_fpath = log_dir / f"{model_name}_{tokenizer_type}_{task}_{data_format}_M-{max_m}_N-{max_n}_token_stats.csv"
                        # token_accum_log_fpath = log_dir / f"{model_name}_{tokenizer_type}_{task}_{data_format}_M-{max_m}_N-{max_n}_token_accum.json"
                        if token_stats_log_fpath.exists():
                            print(f"{model_name} {tokenizer_type} {task} {data_format} already exists. Skip.")
                            continue

                        tokenizer_dir = get_tpe_tokenizer_dir(tpe_tokenizers_dir, 
                                                            model_name=model_name,
                                                            task=task, 
                                                            max_n=max_n,
                                                            max_m=max_m)
                        llm_tokenizer = TPETokenizerFast.from_pretrained(str(tokenizer_dir))
                        dataset = ICU_Mortality_Dataset(llm_tokenizer=llm_tokenizer, 
                                                        parquet_fpath=f'data/EHR_QA/{dataset_id}/{task}/{data_format}/all.parquet', 
                                                        pe_method="raw")
                        
                        # Calculate the time-event and time-token accumulation
                        token_accum_stats_df, token_accum_results = dataset.calculate_time2token_accumulation()
                
                        # Save the results
                        token_accum_stats_df.write_csv(token_stats_log_fpath)
                        # with open(token_accum_log_fpath, "w") as f:
                        #     json.dump(token_accum_results, f, indent=4)
                        print(f"{data_format} results saved successfully.")
