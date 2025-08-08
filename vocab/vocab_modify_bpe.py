from collections import defaultdict, Counter, OrderedDict
import heapq
from tqdm import tqdm
from typing import Optional, Dict, List, Tuple
from datasets import DatasetDict
from transformers import PreTrainedTokenizerFast
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import bisect


from dataset.map_fn import dataset_to_Dtok_batch_mapping_fn


# 全局变量（子进程里也能访问）
_shared_vocab = None
_shared_merges = None
_shared_merged = None

def _init_worker(vocab, merges, merged):
    """
    在每个子进程启动时调用一次，把大结构赋值给进程全局变量
    """
    global _shared_vocab, _shared_merges, _shared_merged
    _shared_vocab = vocab
    _shared_merges = merges
    _shared_merged = merged


def prepare_n_tokens(Dtok, max_n, min_n=2):
    """Generate all contiguous n-grams (as strings) with length from min_n to max_n
    
    Args:
        Dtok: Tokenized documents (list of lists of tokens)
        max_n: Maximum n-gram length
        min_n: Minimum n-gram length (default=2)
    
    Returns:
        List of unique n-gram strings
    """
    assert max_n >= min_n, f"max_n ({max_n}) must be >= min_n ({min_n})"
    
    n_tokens = set()
    pbar = tqdm(total=len(Dtok), desc="Generating n-grams", unit="doc")
    for doc in Dtok:
        tokens = doc
        length = len(tokens)
        for i in range(length):
            # Start from min_n instead of 1
            for j in range(min_n, max_n + 1):
                if i + j <= length:
                    ngram = ' '.join(tokens[i:i+j])
                    n_tokens.add(ngram)
        pbar.update(1)
    pbar.close()
    return list(n_tokens)


def cnt_freqs(Dtok, n_tokens, max_n):
    """Count the frequency of each n-gram in the corpus"""
    freq = defaultdict(int)
    ngram_set = set(n_tokens)
    pbar = tqdm(total=len(Dtok), desc="Counting n-gram frequencies", unit="doc")
    for doc in Dtok:
        tokens = doc
        doc_len = len(tokens)
        for i in range(doc_len):
            for j in range(1, max_n + 1):
                if i + j <= doc_len:
                    ngram = ' '.join(tokens[i:i+j])
                    if ngram in ngram_set:
                        freq[ngram] += 1
        pbar.update(1)
    pbar.close()
    return freq


def cnt_overlaps(n_tokens, Fn_tok, max_n):
    """
    Count the overlaps between n-grams and their prefixes.
    """
    overlaps = defaultdict(lambda: defaultdict(int))
    prefix_to_targets = defaultdict(list)

    # Build prefix-to-n-gram (t' -> list of its prefixes)
    for t_prime in n_tokens:
        tokens = t_prime.split()
        for k in range(1, min(len(tokens), max_n)):
            prefix = ' '.join(tokens[:k])
            prefix_to_targets[prefix].append(t_prime)

    # Match each n-gram t to all t' it is a prefix of
    for t in n_tokens:
        for t_prime in prefix_to_targets.get(t, []):
            overlaps[t][t_prime] = Fn_tok[t_prime]

    return overlaps


def count_old_token_frequencies(
    tokenized_docs: List[List[str]], 
    vocab_old: Dict[str, int],
    disable_progress: bool = False
) -> Dict[str, int]:
    """
    Efficiently counts frequencies of tokens present in original vocabulary
    
    Args:
        tokenized_docs: List of tokenized documents (list of lists)
        vocab_old: Original vocabulary {token: id}
        disable_progress: Whether to hide progress bar
    
    Returns:
        Dictionary of {token: count} for tokens in original vocabulary
    """
    # Precompute set for faster lookups
    vocab_set = set(vocab_old.keys())
    freq_counts = defaultdict(int)
    
    # Use manual progress bar control for better performance
    with tqdm(total=len(tokenized_docs), 
             desc="Counting token frequencies",
             unit="doc",
             disable=disable_progress) as pbar:
        
        # Process documents in batches for better memory efficiency
        batch_size = 1024
        for i in range(0, len(tokenized_docs), batch_size):
            batch = tokenized_docs[i:i+batch_size]
            
            # Flatten and count using Counter's C-optimized implementation
            batch_counts = Counter(
                token
                for doc in batch
                for token in doc
                if token in vocab_set
            )
            
            # Merge counts
            for token, count in batch_counts.items():
                freq_counts[token] += count
                
            pbar.update(len(batch))
    return freq_counts


def deduplicate_keep_order(lst):
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def insert_one_merge(
    merges: List[Tuple[str, str]],
    merge_to_insert: Tuple[str, str],
    position: int
) -> None:
    """
    Insert a single merge at the specified position if it does not already exist.
    Args:
        merges: List of existing merges (modified in place).
        merge_to_insert: The merge tuple (a, b) to insert.
        position: Index at which to insert the new merge.
    """
    if merge_to_insert in merges:
        return
    # Ensure position is within bounds
    pos = max(0, min(position, len(merges)))
    merges.insert(pos, merge_to_insert)


def validate_merge_path(merge_path, vocab):
    """Check if the merge path is valid in the vocabulary.
    which means all tokens in the path exist in the vocabulary.
    """
    for merge in merge_path:
        if len(merge) != 2:
            return False
        a, b = merge
        if a not in vocab or b not in vocab:
            return False
    return True


# 修改后的原子插入验证函数（返回操作列表）
def insert_merge_path_at_earliest(
    merges: List[Tuple[str, str]],
    merged_list: List[str],
    merge_path: List[Tuple[str, str]],
    vocab: Dict[str, int]
) -> Tuple[bool, List[Tuple[int, Tuple[str, str]]]]:
    """返回 (是否成功, 需要插入的[位置, token对]列表) """
    existing = set(merges)
    pos_map = {tok: idx for idx, tok in enumerate(merged_list)}
    ops = []
    
    for a, b in merge_path:
        if (a not in vocab) or (b not in vocab):
            return False, []
        if (a, b) in existing:
            continue
        if (len(a)>1 and a not in pos_map) or (len(b)>1 and b not in pos_map):
            return False, []
        
        idx = max(pos_map.get(a, 0), pos_map.get(b, 0)) + 1
        ops.append((idx, (a, b)))
        
        # 仅模拟不实际修改
        merged_tok = a + b
        pos_map[merged_tok] = idx
        existing.add((a, b))
    
    return True, ops

# 并行处理函数
def process_single_ngram(ngram_tok):
    sub_toks = ngram_tok.split()
    merged_token = "".join(sub_toks)
    
    if merged_token in _shared_merged:
        return [], ngram_tok
    
    merge_paths = get_all_merge_paths(sub_toks)
    for path in merge_paths:
        if not validate_merge_path(path, _shared_vocab):
            continue
        valid, ops = insert_merge_path_at_earliest(
            _shared_merges, _shared_merged, path, _shared_vocab
        )
        if valid:
            return ops, ngram_tok
    return [], ngram_tok


def parallel_insert_ngram_into_merges(ngram_tokens, merges, vocab, n_workers=8):
    initial_merges = merges.copy()
    initial_merged = [f"{a}{b}" for a, b in initial_merges]
    valid_ngram_tokens = []
    all_ops = []

    # 1) 提交所有任务，拆开参数
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(vocab, initial_merges, initial_merged)
    ) as executor:
        # 3) 这里只传小参数：每个任务只传一个 token 字符串
        futures = {executor.submit(process_single_ngram, tok): tok
                   for tok in ngram_tokens}

        # 4) 进度条 + 收集 ops
        for future in tqdm(as_completed(futures),
                           total=len(futures),
                           desc=f"Adding merges for n-grams (N_proc={n_workers})",
                           unit="ngram"):
            ops, ngram_tok = future.result()
            if ops:
                all_ops.extend(ops)
                valid_ngram_tokens.append(ngram_tok)

    # 3) 把所有 ops 排序、按原逻辑插入
    all_ops.sort(key=lambda x: x[0])
    insertion_records = []
    final_merges = initial_merges.copy()

    for original_idx, (a, b) in all_ops:
        # 计算实际插入位置
        insert_pos = original_idx
        for pos, count in insertion_records:
            if original_idx >= pos:
                insert_pos += count

        final_merges.insert(insert_pos, (a, b))

        # 记录这次插入
        bisect_idx = bisect.bisect_left([r[0] for r in insertion_records], original_idx)
        insertion_records.insert(bisect_idx, (original_idx, 1))

    # 4) 去重并返回
    final_merges = deduplicate_keep_order(final_merges)
    return final_merges, valid_ngram_tokens


def get_all_merge_paths(subtokens: List[str]) -> List[List[Tuple[str, str]]]:
    """
    Enumerate all full merge paths for a list of subtokens.

    Args:
        subtokens: List of subtokens, e.g. ["a", "b", "c", "d"]

    Returns:
        A list of merge-sequences. Each merge-sequence is a list of tuples,
        where each tuple is (left, right) tokens that were merged at that step.
        The sequence has length len(subtokens) - 1.
    """
    # 递归基：只有一个 token 时，不用 merge，返回包含一个空路径的列表
    if len(subtokens) <= 1:
        return [[]]

    all_merge_paths = []

    # 对每一个可能的相邻位置 i，进行一次 merge
    for i in range(len(subtokens) - 1):
        left = subtokens[i]
        right = subtokens[i + 1]
        merged = left + right

        # 构造下一层的 token 列表
        next_tokens = (
            subtokens[:i] +         # 左侧不变部分
            [merged] +              # 放入新合并的 token
            subtokens[i + 2:]       # 右侧不变部分
        )

        # 递归获取剩余的 merge-paths
        for suffix_path in get_all_merge_paths(next_tokens):
            # 把当前这一步 (left,right) prepend 到后续路径上
            all_merge_paths.append([(left, right)] + suffix_path)

    return all_merge_paths


def vocabulary_modify_fn(tokenizer: PreTrainedTokenizerFast, 
                        dataset_dict: DatasetDict,  # Changed to HF Dataset
                        max_num_new_tokens: int, 
                        max_n: int,
                        merges: list,
                        dataset_split: str="train",
                        min_freq: int=2) -> Tuple[Dict[str, int], List[Tuple[str, str]], Dict[int, Dict[str, str]]]:
    """
    Modify the vocabulary: replace up to max_num_new_tokens lowest-frequency old tokens with top max_num_new_tokens n-grams

    Key notes:
        1) skip special tokens and tokens with len(str) == 1 from replacement
        2) skip tokens in the merge paths in an recursive manner
        3) keeps all the old tokens that are not replaced by new tokens
    Args:
        tokenizer: PreTrainedTokenizerFast object
        dataset_dict: HF DatasetDict object
        max_num_new_tokens: Maximum number of new tokens to add
        max_n: Maximum n-gram length
        merges: List of tuples representing merges
        dataset_split: Dataset split to use (e.g., "train", "test")
        min_freq: Minimum frequency for n-grams to be considered (default=2)
    Returns:
        vocab_new: New vocabulary
        merges_new
        new_token_info: Information about new tokens added
    """
    # Initialize old and new vocabularies
    vocab_old = tokenizer.vocab
    vocab_new = {}
    
    # 1. Tokenize the corpus from HF Dataset
    dataset = dataset_dict[dataset_split]  # Extract text column
    dataset = dataset.map(dataset_to_Dtok_batch_mapping_fn,
                          fn_kwargs={"tokenizer": tokenizer},
                          batched=True,
                          desc="Tokenizing dataset",
                          batch_size=1000,
                          num_proc=8)  # Use multiple processes for efficiency
    Dtok = [item["token"] for item in dataset]  # Assuming 'data' is the text column
    
    # 2. Extract n-grams and calculate their score
    ngram_tokens = prepare_n_tokens(Dtok, max_n)
    Fn_tok = cnt_freqs(Dtok, ngram_tokens, max_n)
    
    # Calculate overlap relationships
    Foverlaps = cnt_overlaps(ngram_tokens, Fn_tok, max_n)
    
    # Calculate initial scores S = frequency × length
    _S = {t: Fn_tok[t] * len(t.split()) for t in ngram_tokens if Fn_tok[t] >= min_freq} # remove low freq n-grams
    S = OrderedDict(sorted(_S.items(), key=lambda x: x[1], reverse=True))  # Sort by score descending
    ngram_tokens = list(S.keys())

    # 3. Develop Merges Table for all n-grams
    # # Collect new merge paths for ngram
    merges, valid_ngram_tokens = parallel_insert_ngram_into_merges(
        ngram_tokens=ngram_tokens,
        merges=merges,
        vocab=vocab_old,
        n_workers=16
    )
    S = {k: S[k] for k in valid_ngram_tokens}  # Filter S to only include valid n-grams
    ngram_tokens = list(S.keys())

    # 4. Set token list that won't be replaced
    # Special tokens
    special_tokens = set()
    for key, token in tokenizer.special_tokens_map.items():
        if token is None:
            continue
        if key == "additional_special_tokens":
            special_tokens.update(set(token))
        else:
            special_tokens.add(token)
        
    # Reserve tokens on the merge paths for new toks
    merges_dict = {f"{parts[0]}{parts[1]}": parts for parts in merges}
    def _decompose_token(t: str, vocab) -> List[str]:
        """递归分解token直到所有子token在merges_dict存在"""   
        if t in merges_dict:
            t1, t2 = merges_dict[t]
            return [t] + _decompose_token(t1, vocab) + _decompose_token(t2, vocab)
        elif t in vocab:
            return [t]
        else:   
            raise ValueError(f"Token {t} not in vocab")
        
    subtokens = set()
    pbar = tqdm(total=len(ngram_tokens), desc="Decomposing tokens", unit="token")
    for tok in ngram_tokens:
        for sub_tok in tok.split():
            if sub_tok not in vocab_old: # make sure all sub-tokens are in vocab_old
                S.pop(tok)
                break
            subtokens.update(_decompose_token(sub_tok, vocab_old))
        pbar.update(1)
    pbar.close()

    # 4. create the new vocab by replacing old tokens with new tokens
    # Build min-heap (sorted by frequency ascending) of old tokens
    old_token_freq = count_old_token_frequencies(
        Dtok, vocab_old, disable_progress=False
    )
    heap = []

    for token_str, token_id in vocab_old.items():
        # Skip special tokens and tokens with length 1
        if token_str in special_tokens or len(token_str) == 1:
            continue
        elif token_str in subtokens:
            continue
        freq = old_token_freq.get(token_str, 0)
        heapq.heappush(heap, (freq, token_str, token_id))
    
    # Determine actual number of replacements
    if max_num_new_tokens == -1:
        actual_replacements = len(S)
    else:
        actual_replacements = min(max_num_new_tokens, len(S))
    new_token_info = {}
    
    # Replace tokens
    # dependent_toks = [] # toks that on the merge path of the new token
    # merged_list = [f"{parts[0]}{parts[1]}" for parts in merges]
    pbar = tqdm(total=actual_replacements, desc="Replacing tokens", unit="token")
    for _ in range(actual_replacements):
        if not heap or not S:
            break
        
        new_token_ngram = max(S, key=lambda k: S[k])
        sub_tokens = new_token_ngram.split()
        if not all(sub_token in vocab_old for sub_token in sub_tokens):
            print(f"Token {new_token_ngram} contains unknown sub-tokens. Skipping.")
            S.pop(new_token_ngram)
            pbar.update(1)
            continue

        sub_token_ids = [vocab_old[sub_token] for sub_token in sub_tokens]
        new_token_str = new_token_ngram.replace(" ", "")  # only drop the space caused by the split and keep the original space

        # Check if new token already exists in old vocab
        if new_token_str in vocab_old:
            print(f"Token {new_token_str} already exists in old vocab. Skipping.")
            S.pop(new_token_ngram)  # Already in old vocab
            pbar.update(1)
            continue

        # Pop the lowest frequency token from the heap of old tokens but skip the ones in dependent_toks
        freq, old_token_str, token_id = heapq.heappop(heap)

        # replace old token with new tok en
        vocab_old.pop(old_token_str)
        vocab_new[new_token_str] = token_id

        # add new token info
        new_token_info[token_id] = {
            "token_ngram": new_token_ngram,
            "token_str": new_token_str,
            "sub_token_ids": sub_token_ids,
            "sub_tokens": sub_tokens,
        }
        
        # Decrease the score of the n-gram with the new token as prefix
        for t_prime in Foverlaps.get(new_token_ngram, {}):
            if t_prime in S:
                S[t_prime] -= Foverlaps[new_token_ngram][t_prime] * len(new_token_ngram.split())
        
        S.pop(new_token_ngram)  # Remove the n-gram from S
        pbar.update(1)
    pbar.close()

    # Merge remaining vocabulary
    vocab_new.update(vocab_old)

    # 6. remove invalid merges for the new vocab
    updated_merges_valid = []
    for merge in merges:
        assert len(merge) == 2, f"Invalid merge: {merge}"
        if merge[0]+merge[1] in vocab_new and all(part in vocab_new and part != "" for part in merge):
            updated_merges_valid.append((merge[0], merge[1]))
    
    return vocab_new, updated_merges_valid, new_token_info

