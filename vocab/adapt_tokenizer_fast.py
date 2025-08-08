import json
from typing import Dict, List, Union
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast, BatchEncoding
import os
from transformers.tokenization_utils_base import PaddingStrategy, TruncationStrategy
from collections import defaultdict


class AdaptTokenizerFast(PreTrainedTokenizerFast):
    def __init__(self, original_tokenizer, vocab_new: Dict[str, int], max_n: int = 3):
        """
        Args:
            original_tokenizer: Original tokenizer instance
            vocab_new: New vocabulary dictionary {token_str: token_id}
            max_n: Maximum n-gram matching length
        """

        super().__init__(
            tokenizer_object=original_tokenizer.backend_tokenizer,
            vocab=original_tokenizer.vocab,
            merges=original_tokenizer.merges if hasattr(original_tokenizer, "merges") else [],
            **original_tokenizer.init_kwargs  # 传递其他必要参数
        )
        # self.original_tokenizer = original_tokenizer
        self.encode_special_tokens = original_tokenizer.backend_tokenizer.encode_special_tokens
        self.original_tokenizer = original_tokenizer
        setattr(self.original_tokenizer, "truncation", self.original_tokenizer.backend_tokenizer.truncation)
        setattr(self.original_tokenizer, "padding", self.original_tokenizer.backend_tokenizer.padding)
        self.vocab_new = vocab_new
        self.max_n = max_n
        self.id_to_token = {v: k for k, v in vocab_new.items()}  # Reverse mapping
        
        # Load merge table
        self.merge_table = self._load_merge_table(original_tokenizer)

        # inherit special tokens
        self._inherit_special_tokens(original_tokenizer)

    def _inherit_special_tokens(self, src_tokenizer):
        """继承特殊token配置"""
        special_tokens_map = src_tokenizer.special_tokens_map
        self.special_tokens_ids = []
        
        for attr, token in special_tokens_map.items():
            if attr != "additional_special_tokens":
                token_id = self.vocab_new.get(token, None)
                assert token_id is not None, f"Token '{token}' not found in new vocab"
                setattr(self, attr, token)
                setattr(self, f"{attr}_id", token_id)
                self.special_tokens_ids.append(token_id)
            else:
                for t in token:
                    token_id = self.vocab_new.get(t, None)
                    if token_id is not None:
                        self.special_tokens_ids.append(token_id)
        
        # add unk token
        if not hasattr(self, "unk_token"):
            self.unk_token = self.pad_token
            self.unk_token_id = self.pad_token_id
            self.special_tokens_ids.append(self.pad_token_id)

        self.special_tokens_ids = list(set(self.special_tokens_ids))

        
    def _load_merge_table(self, tokenizer) -> Dict[str, tuple]:
        """从原始tokenizer加载merge规则"""
        # 获取tokenizer的合并规则
        if hasattr(tokenizer, "get_merges"):
            merges = tokenizer.get_merges()
        else:
            # 尝试从tokenizer.json加载
            tokenizer_config = getattr(tokenizer, "backing_kwargs", {}).get("tokenizer_file")
            if tokenizer_config:
                with open(tokenizer_config, "r", encoding="utf-8") as f:
                    config = json.load(f)
                    merges = config.get("model", {}).get("merges", [])
            else:
                merges = []
        
        # 构建merge table
        merge_table = {}
        for merge in merges:
            t1, t2 = merge.split(" ")
            merged = t1 + t2
            merge_table[merged] = (t1, t2)
        return merge_table
    

    def _decompose_token(self, t: str) -> List[str]:
        """递归分解token直到所有子token在新词汇表中存在"""
        if t in self.vocab_new:
            return [t]
        if t in self.merge_table:
            t1, t2 = self.merge_table[t]
            return self._decompose_token(t1) + self._decompose_token(t2)
        else:
            # 如果无法分解且不在新词汇表中，保留原样
            return [t]
    
    def tokenize(self, text: str, **kwargs) -> List[str]:
        """核心分词方法"""
        # 1. 用原始tokenizer分词
        original_tokens = self.original_tokenizer.tokenize(text, **kwargs)
        
        # 2. 分解为基本单元
        base_tokens = []
        for t in original_tokens:
            base_tokens.extend(self._decompose_token(t))
        
        # 3. 使用最大前缀匹配重新组合
        tokens_new = []
        i = 0
        while i < len(base_tokens):
            matched = False
            # 从最大可能长度开始尝试匹配
            for k in range(min(self.max_n, len(base_tokens) - i), 0, -1):
                candidate = "".join(base_tokens[i:i+k])
                if candidate in self.vocab_new:
                    tokens_new.append(candidate)
                    i += k
                    matched = True
                    break
            if not matched:
                # 如果没有匹配，添加当前子词
                tokens_new.append(base_tokens[i])
                i += 1
        return tokens_new
    
    def tokenize_original(self, text: str, **kwargs) -> List[str]:
        """使用原始tokenizer分词"""
        return self.original_tokenizer.tokenize(text, **kwargs)
    
    def _batch_encode_plus(self, 
                           batch_text_or_text_pairs, 
                           add_special_tokens = True, 
                           padding_strategy = PaddingStrategy.DO_NOT_PAD, 
                           truncation_strategy = TruncationStrategy.DO_NOT_TRUNCATE, 
                           max_length = None, 
                           stride = 0, 
                           is_split_into_words = False, 
                           pad_to_multiple_of = None, 
                           padding_side = None, 
                           return_tensors = None, 
                           return_token_type_ids = None, 
                           return_attention_mask = None, 
                           return_overflowing_tokens = False, 
                           return_special_tokens_mask = False, 
                           return_offsets_mapping = False, 
                           return_length = False, 
                           verbose = True, 
                           split_special_tokens = False):
        """批量编码文本"""
        encode_data = defaultdict(list)
        for i, text in enumerate(batch_text_or_text_pairs):
            tokens = self.tokenize(text)
            input_ids = self.convert_tokens_to_ids(tokens)

            # add special tokens
            if add_special_tokens:
                input_ids = [self.cls_token_id] + input_ids + [self.sep_token_id]

            # truncate and padding
            if max_length is not None and len(input_ids) >= max_length:
                input_ids = input_ids[:max_length]
                attention_mask = [1] * max_length
            elif max_length is not None and len(input_ids) < max_length:
                input_ids += [self.pad_token_id] * (max_length - len(input_ids))
                attention_mask = [1] * len(input_ids) + [0] * (max_length - len(input_ids))
            else:
                # input_ids = input_ids
                attention_mask = [1] * len(input_ids)
            
            encode_data["input_ids"].append(input_ids)
            if return_attention_mask:
                encode_data["attention_mask"].append(attention_mask)
            if return_special_tokens_mask:
                special_tokens_mask = [1 if token in self.encode_special_tokens else 0 for token in tokens]
                encode_data["special_tokens_mask"].append(special_tokens_mask)
            if return_token_type_ids:
                token_type_ids = [0] * len(input_ids)
                encode_data["token_type_ids"].append(token_type_ids)
            
        batch_encoding = BatchEncoding(data=encode_data)
        return batch_encoding
    
    def __len__(self):
        """获取词汇表大小"""
        return len(self.vocab_new)
    
    def _convert_token_to_id(self, token: str) -> int:
        """将token转换为ID"""
        return self.vocab_new.get(token, self.pad_token_id)
    
    def _convert_id_to_token(self, index: int) -> str:
        """将ID转换为token"""
        return self.id_to_token.get(index, self.pad_token)
    
    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return self._convert_token_to_id(tokens)
        elif isinstance(tokens, list):
            return [self._convert_token_to_id(t) for t in tokens]
        
    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, int):
            return self._convert_id_to_token(ids)
        elif isinstance(ids, list):
            return [self._convert_id_to_token(i) for i in ids]
    
    def get_vocab(self) -> Dict[str, int]:
        """获取词汇表"""
        return self.vocab_new
    
    def save_pretrained(self, save_directory, **kwargs):
        """保存tokenizer"""
        # 保存原始tokenizer配置
        self.original_tokenizer.save_pretrained(save_directory)
        
        # 添加自定义词汇表信息
        import os
        with open(os.path.join(save_directory, "vocab_new.json"), "w", encoding="utf-8") as f:
            json.dump(self.vocab_new, f)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """从保存的目录加载"""
        original_tokenizer = PreTrainedTokenizerFast.from_pretrained(pretrained_model_name_or_path)
        
        # 加载自定义词汇表
        new_vocab_path = os.path.join(pretrained_model_name_or_path, "vocab_new.json")
        if os.path.exists(new_vocab_path):
            with open(new_vocab_path, "r", encoding="utf-8") as f:
                vocab_new = json.load(f)
        else:
            raise ValueError(f"Custom vocab file not found at {new_vocab_path}")
        
        return cls(original_tokenizer, vocab_new, **kwargs)
    
    # 代理其他必要方法
    def __getattr__(self, name):
        # 直接访问实例字典，避免触发 __getattr__
        if name == "tokenize":
            return self._tokenize
        original_tokenizer = self.__dict__.get("original_tokenizer")
        if original_tokenizer is not None:
            return getattr(original_tokenizer, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    


if __name__ == "__main__":
    from vocab.vocab_modify_bpe import vocabulary_modify_fn
    from dataset.map_fn import sft_sample_mapping_fn

    original_tokenizer = PreTrainedTokenizerFast.from_pretrained("data/hf_models/meta-llama--Llama-3.1-8B-Instruct")
    print("tokenizer_type: ", type(original_tokenizer))
    vocab_orig = original_tokenizer.get_vocab()
    test_text = """The event types are as follows:
    1. ED_REGISTRATION: Registration to an emergency department.
    2. ED_OUT: Discharge or exit from the emergency department.
    3. HOSPITAL_ADMISSION: Admission to a hospital.
    4. HOSPITAL_DISCHARGE: Discharge from a hospital.
    5. DIAGNOSIS: A disease diagnosis for a patient.
    6. DRG: Assigning a Diagnosis-Related Group (DRG) code for billing and classification purposes.
    7. MEDICATION: Administering or prescribing medication to a patient.
    8. HCPCS: Recording Healthcare Common Procedure Coding System (HCPCS) codes for procedures and services.
    9. LAB: Recording laboratory test results.
    10. OMR: Recording Objective Medical Records (OMR), such as structured clinical data.
    11. GENDER: Recording a patient's gender.
    16. PROCEDURE: A medical procedure.
    17. TRANSFER_TO: Transferring a patient to another department or facility.
    18. ICU_ADMISSION: Admission to an Intensive Care Unit (ICU).
    19. ICU_DISCHARGE: Discharge from an ICU.
    20. VITALS: Recording vital signs of a patient.
    21. INFUSION_START: Starting an infusion.
    22. INFUSION_END: Ending an infusion.
    23. SUBJECT_WEIGHT_AT_INFUSION: Recording a patient's weight at the time of infusion.
    24. SUBJECT_FLUID_OUTPUT: Recording a patient's fluid output."""

    # get the vocab_new
    data_files = {'held_out': 'log/QA/MIMICIV/icu_phenotyping_nl_2k_DeepSeek-R1-Distill-Llama-8B_raw.parquet'}
    max_n = 3 # max n-gram size
    dataset = load_dataset("parquet", data_files=data_files, columns=['message', 'generated_text_0'])
    dataset = dataset.map(sft_sample_mapping_fn)
    vocab_new, info = vocabulary_modify_fn(original_tokenizer, dataset, max_num_new_tokens=10000, max_n=max_n, dataset_split="held_out")

    # create the adapted tokenizer
    adapted_tokenizer = AdaptTokenizerFast(original_tokenizer=original_tokenizer, vocab_new=vocab_new, max_n=max_n)
    tokens_new = adapted_tokenizer.tokenize(test_text, add_special_tokens=True)
    tokens_orig = adapted_tokenizer.tokenize_original(test_text)
    encodings = adapted_tokenizer(test_text, add_special_tokens=True, return_attention_mask=True, return_token_type_ids=True)
    print("Original tokens: ", tokens_orig)
    print("Adapted tokens: ", tokens_new)
    print("Original vocab size: ", len(vocab_orig))
    print("New vocab size: ", len(vocab_new))
    print("added tokens size: ", len(info))
    print("Length of original tokens: ", len(tokens_orig))
    print("Length of adapted tokens: ", len(tokens_new))
    