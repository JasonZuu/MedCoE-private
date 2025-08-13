from datasets import load_dataset
import json
import re
import numpy as np
from typing import List, Dict, Any
from transformers import PreTrainedTokenizerBase
from llmlingua import PromptCompressor


def compress_with_lingua2_map_fn(sample,
                                 compress_rate: float):
    """
    The samples is a list of {
        "message": messages_batch,
        "label":   labels_batch
    }
    This function compresses the messages in the samples using LLMLingua2.
    return the same sample list with compressed messages.
    """
    llm_lingua = PromptCompressor(
        model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
        use_llmlingua2=True, # Whether to use llmlingua-2
    )
    rate = 1 - compress_rate # LLMLingua2 uses rate as the compression rate
    
    # Extract messages from samples
    prompts = []
    for message in sample["message"]:
        if isinstance(message, list):
            prompts.append(message[0]["content"])
        else:
            prompts.append(message["content"])

    # Compress prompts using LLMLingua2 by batch
    compressed_prompts = []
    for prompt in prompts:
        compressed_prompt = llm_lingua.compress_prompt(prompt,
                                                        rate=rate,
                                                        force_tokens=['\n', ':'])
        compressed_prompts.append(compressed_prompt["compressed_prompt"])

    message_batch = [
        [{"role": "user", "content": prompt}] for prompt in compressed_prompts
    ]
    labels_batch = sample["label"]
    return {
        "message": message_batch,
        "label":   labels_batch
    }


def sample_mapping_fn(sample, instruction, llm_tokenizer, max_input_length):
    """
    Batch-processing version of sample_mapping_fn.

    Args:
        sample (dict): Batch of samples, e.g.
            {
              "data":     [json_str1, json_str2, ...],
              "question": [q1, q2, ...],
              "label":    [lbl1, lbl2, ...]
            }
        instruction (str): The instruction to append.
        llm_tokenizer: Tokenizer with apply_chat_template().
        max_input_length (int): Max tokens for the model input.

    Returns:
        dict: {
            "message": [ [ {role,content}, ... ],  # one list per example
                         ... ],
            "label":   [ lbl1, lbl2, ... ]
        }
    """
    # 计算剩余可用 token 长度
    #    先算仅 question+instruction 模板需要多少 token
    question = sample["question"]
    if isinstance(question, list):
        question = question[0]
    nodata_messages = [
        {"role": "user", "content": question + instruction}
    ]
    nodata_tokens = llm_tokenizer.apply_chat_template(nodata_messages)
    max_data_tokens = max_input_length - len(nodata_tokens)

    messages_batch = []
    labels_batch = []

    # 逐条处理
    for data_json, label in zip(
        sample["data"], sample["label"]
    ):
        # 1) 载入 data 字典
        data_dict = json.loads(data_json)

        # 2) 把 data_dict 转成字符串片段，截断到 max_data_tokens
        data_str = _turn_data_dict_to_str(
            data_dict, max_data_tokens, llm_tokenizer
        )

        # 3) 拼接最终 prompt
        prompt = data_str + question + instruction

        # 4) 按 chat 格式包装成 list-of-messages
        messages_batch.append(
            [{"role": "user", "content": prompt}]
        )

        # 5) 保存标签
        labels_batch.append(label)

    return {
        "message": messages_batch,
        "label":   labels_batch
    }

def _turn_data_dict_to_tokens(data_dict:dict, max_data_tokens_length:int, llm_tokenizer,
                              data_prefix="## Data\n"):
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
        static_tokens = llm_tokenizer.encode(static_sentence, add_special_tokens=False)
    else:
        static_tokens = []
    data_prefix_tokens = llm_tokenizer.encode(data_prefix, add_special_tokens=False)
    max_data_tokens_length -= len(static_tokens) + len(data_prefix_tokens)

    tokens = []
    for time, str_events in reversed(data_dict.items()):
        if str_events is None:
            continue
        time_sentence = "".join(str_events)
        time_tokens = llm_tokenizer.encode(time_sentence, add_special_tokens=False)
        if len(tokens) + len(time_tokens) > max_data_tokens_length:
            break
        # Extend time_tokens at the beginning of the tokens list
        tokens = time_tokens + tokens
    return data_prefix_tokens + static_tokens + tokens


def _turn_data_dict_to_str(data_dict: dict, max_data_tokens_length: int, llm_tokenizer, data_prefix: str = "## Data\n"):
    """
    Convert a dictionary to a string.
    This function will calculate the maximum length of the data and return the string representation of the data.
    Args:
        data_dict (dict): The dictionary to be converted to a string.
        max_data_tokens_length (int): The maximum available data tokens length.
        llm_tokenizer: The tokenizer used to encode the data.
        data_prefix (str): The prefix to add before the data.
    Returns:
        str: The string representation of the dictionary.
    """
    if type(data_dict) is dict:
        # Extract the static events
        static_sentences = data_dict.pop("Static", [])
        static_sentence = "".join(static_sentences) if static_sentences else ""
        static_tokens = llm_tokenizer.encode(static_sentence, add_special_tokens=False) if static_sentence else []
        data_prefix_tokens = llm_tokenizer.encode(data_prefix, add_special_tokens=False)
        max_data_tokens_length -= len(static_tokens) + len(data_prefix_tokens)

        sentences = []
        tokens = []

        for time, str_events in reversed(data_dict.items()):
            if not str_events:
                continue
            if type(str_events) is str:
                str_events = [str_events]
            time_sentence = "".join(str_events)
            time_tokens = llm_tokenizer.encode(time_sentence, add_special_tokens=False)
            if len(tokens) + len(time_tokens) > max_data_tokens_length:
                break
            # Extend time_tokens at the beginning of the tokens list
            tokens = time_tokens + tokens
            sentences = str_events + sentences

        sentences = [data_prefix, static_sentence] + sentences
        input_prompt = "".join(sentences)
    elif type(data_dict) is str:
        data_prefix_tokens = llm_tokenizer.encode(data_prefix, add_special_tokens=False)
        max_data_tokens_length -= len(data_prefix_tokens)
        data_tokens = llm_tokenizer.encode(data_dict, add_special_tokens=False)
        if len(data_tokens) > max_data_tokens_length:
            data_tokens = data_tokens[-max_data_tokens_length:]
            input_prompt = llm_tokenizer.decode(data_tokens)
        else:
            input_prompt = data_dict
    return input_prompt



def _find_subseq(sequence: List[int], pattern: List[int]) -> int:
    """Return start index of *pattern* in *sequence* or -1 if not found."""
    pat_len = len(pattern)
    if pat_len == 0 or pat_len > len(sequence):
        return -1
    # simple sliding-window search (list.index on tuple slice is also OK)
    for i in range(len(sequence) - pat_len + 1):
        if sequence[i : i + pat_len] == pattern:
            return i
    return -1


def sft_map_prefix_fn(
    batch: Dict[str, List[Any]],
    tokenizer,
    max_seq_length: int,
    generation_prefix: str = "<|im_start|>assistant\n",
) -> Dict[str, np.ndarray]:
    """Batched ``map`` function for SFT with robust *response* alignment.

    Parameters
    ----------
    batch : dict
        HuggingFace Datasets batch containing ``message`` (list-of-dicts) and
        ``generated_text_0`` (assistant outputs).
    tokenizer : PreTrainedTokenizerBase
        Tokenizer with a chat template configured.
    max_seq_length : int
        Desired sequence length after padding/truncation.
    generation_prefix : str, optional
        String that *always* precedes the assistant response in the applied
        chat template (defaults to Qwen-2 multimodal prefix
        ``"<|im_start|>assistant\n"``).  The first occurrence of its token
        sequence is used to mark the response start when building ``labels``.

    Notes
    -----
    All tokens *before* the first ``generation_prefix`` plus the prefix itself
    are masked out with ``-100`` in ``labels``; only the actual assistant
    answer (and its trailing EOS) contributes to loss.
    """

    messages_batch: List[List[Dict[str, str]]] = batch["message"]
    outputs_batch: List[str] = batch["generated_text_0"]

    eos_id: int = tokenizer.eos_token_id
    prefix_ids: List[int] = tokenizer(
        generation_prefix, add_special_tokens=False, return_attention_mask=False
    )["input_ids"]

    all_ids, all_masks, all_labels = [], [], []

    for msgs, out in zip(messages_batch, outputs_batch):
        # 1. 复制消息并追加助手回复
        conv = msgs + [{"role": "assistant", "content": out}]

        # 2. 应用模板得到完整对话串（不含 generation prompt，因为我们手动追加回复）
        chat_str: str = tokenizer.apply_chat_template(
            conv,
            tokenize=False,
            add_generation_prompt=False,
        )

        # 3. 分词（关闭 special tokens，因为模板已含全部控制符）
        ids: List[int] = tokenizer(
            chat_str, add_special_tokens=False, return_attention_mask=False
        )["input_ids"]

        # 4. 末尾确保 EOS
        # if not ids or ids[-1] != eos_id:
        #     ids.append(eos_id)

        # 6. Attention mask
        attn = [1] * len(ids)

        # 7. Padding to max_len (右侧 pad) using EOS as pad token
        pad_len = max_seq_length - len(ids)
        if pad_len > 0:
            ids.extend([eos_id] * pad_len)
            attn.extend([0] * pad_len)

        # 8. 定位回复起点：查找 generation_prefix token 序列
        start = _find_subseq(ids, prefix_ids)
        if start == -1:
            # Fallback: mask 到最后一个 EOS 之前
            start = len(ids) - pad_len - 1
        else:
            start += len(prefix_ids)  # 跳过 prefix 本身

        # 9. 构造 labels：start 之前全部 -100，之后保留
        labels = [-100] * start + ids[start:]
        if pad_len > 0:
            labels[-pad_len:] = [-100] * pad_len

        all_ids.append(ids)
        all_masks.append(attn)
        all_labels.append(labels)

    input_ids = np.array(all_ids, dtype=np.int32)
    attention_mask = np.array(all_masks, dtype=np.int8)
    labels = np.array(all_labels, dtype=np.int32)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def sft_map_postfix_fn(
    batch: Dict[str, List[Any]],
    tokenizer: PreTrainedTokenizerBase,
    max_seq_length: int,
    generation_postfix: str,
) -> Dict[str, np.ndarray]:
    """Vectorised batched *map* for SFT.

    * **Batch‑encode** history prompts via ``apply_chat_template`` once.
    * **Batch‑encode** answers ("output + postfix") once.
    * 对每个样本在 Python 层做拼接 / 截断 / padding，避免多次 ``tokenizer`` 调用。
    """

    messages_batch: List[List[Dict[str, str]]] = batch["message"]
    outputs_batch:  List[str]                  = batch["generated_text_0"]
    eos_id: int = tokenizer.eos_token_id

    # 1. ----- 历史 prompt (带助手前缀) 批量处理 ----- #
    prompt_strs: List[str] = tokenizer.apply_chat_template(
        messages_batch,                       # list[list[dict]]
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_enc = tokenizer(
        prompt_strs,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_attention_mask=False,
    )
    prompt_ids_batch: List[List[int]] = prompt_enc["input_ids"]

    # 2. ----- 回答 + postfix 批量编码 ----- #
    answer_strs = [out + generation_postfix for out in outputs_batch]
    answer_enc = tokenizer(
        answer_strs,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_attention_mask=False,
    )
    answer_ids_batch: List[List[int]] = answer_enc["input_ids"]

    # 3. ----- 逐样本拼接 / 截断 / pad / labels ----- #
    all_ids, all_masks, all_labels = [], [], []

    for prompt_ids, ans_ids in zip(prompt_ids_batch, answer_ids_batch):
        seq: List[int] = prompt_ids + ans_ids

        # truncate if too long
        if len(seq) > max_seq_length:
            seq = seq[:max_seq_length]

        label_start = len(prompt_ids)  # labels 起始位置：prompt 长度

        # attention mask
        attn = [1] * len(seq)

        # 右侧 padding
        pad_len = max_seq_length - len(seq)
        if pad_len > 0:
            seq.extend([eos_id] * pad_len)
            attn.extend([0] * pad_len)

        # labels：prompt 区域及 padding 设为 -100
        labels = [-100] * label_start + seq[label_start:]
        if pad_len > 0:
            labels[-pad_len:] = [-100] * pad_len

        # debug 断言
        assert len(seq) == max_seq_length
        assert len(attn) == max_seq_length
        assert len(labels) == max_seq_length

        all_ids.append(seq)
        all_masks.append(attn)
        all_labels.append(labels)

    return {
        "input_ids": np.asarray(all_ids, dtype=np.int32),
        "attention_mask": np.asarray(all_masks, dtype=np.int8),
        "labels": np.asarray(all_labels, dtype=np.int32),
    }




def tpe_sample_batch_mapping_fn(samples):
    """
    Batch processing version of sample_mapping_fn.
    Args:
        sample (dict): A batch of samples.
    """
    messages = samples['message']
    labels = samples["generated_text_0"]

    # Process prompt
    messages = [message[0]["content"] for message in messages]
    inputs = [message+label for message, label in zip(messages, labels)]

    return {"input": inputs, "label": labels}


def dataset_to_Dtok_batch_mapping_fn(samples, tokenizer, filter_numbers:bool=True):
    """
    Map the dataset to Dtok format.
    Args:
        sample (dict): A batch of samples.
        tokenizer: The tokenizer used to encode the data.

    Returns:
        dict: A dictionary containing the processed input and label.
    """
    messages = samples['input']
    if filter_numbers:
        cleaned_messages = [re.sub(r'\d+', '', text) for text in messages]
    else:
        cleaned_messages = messages
    tokens = [tokenizer.tokenize(text) for text in cleaned_messages]
    return {"token": tokens}


def map_gender_to_index(gender:str):
    if gender in ["F"]:
        return 0
    elif gender in ["M"]:
        return 1
    else:
        raise ValueError(f"invalid gender: {gender}")
    

def map_race_to_index(race:str):
    """
    map string race into index
    needed to be updatec
    """
    race = race.lower()
    if "white" in race:
        return 0
    elif "black" in race:
        return 1
    elif "hispanic" in race:
        return 2
    elif "asian" in race:
        return 3
    elif "other" in race or "hawaiian" in race or "south american" in race:
        return 4
    else: # unknown case
        return 5
    