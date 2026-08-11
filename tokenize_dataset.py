import math
import torch
import pandas as pd
from tqdm import tqdm
from const import CONST, CONFIG
from CustomDataset import getDataPath
from transformers import AutoTokenizer

def main():
    _tokenizer = AutoTokenizer.from_pretrained(CONST.model_name)
    _seg_num = 0
    for _num in range(CONST.DATA_FILE_COUNT):
        print(f"Handling chunk {_num + 1}/{CONST.DATA_FILE_COUNT}")
        _df = pd.read_parquet(getDataPath(_num))["text"]
        _tokens = []
        _attn_masks = []
        for _text in tqdm(_df):
            _tok = _tokenizer(_text)
            _token = _tok["input_ids"]
            _attn_mask = _tok["attention_mask"]
            _token = [_tokenizer.bos_token_id] + _token + [_tokenizer.eos_token_id]
            _attn_mask = [1] + _attn_mask + [1]
            _tokens.extend(_token)
            _attn_masks.extend(_attn_mask)

        del _token, _attn_mask, _df

        _tokens = torch.tensor(_tokens)
        _attn_masks = torch.tensor(_attn_masks)

        _total_length = len(_tokens)
        _lines = math.floor(_total_length / CONFIG.context_size)
        _trim_lines = (
            math.floor(_lines / CONFIG.buffer_size)
            * CONFIG.buffer_size
            * CONFIG.context_size
        )
        _tokens_block = _tokens[:_trim_lines]
        _tokens_remainder = _tokens[_trim_lines:]
        del _tokens
        _attn_masks_block = _attn_masks[:_trim_lines]
        _attn_masks_remainder = _attn_masks[_trim_lines:]
        del _attn_masks
        _remainder_length = _tokens_remainder.shape[0]
        _trim_length = (
            math.floor(_remainder_length / CONFIG.context_size) * CONFIG.context_size
        )
        _tokens_remainder = _tokens_remainder[:_trim_length]
        _attn_masks_remainder = _attn_masks_remainder[:_trim_length]
        _tokens_block = _tokens_block.view(-1, CONFIG.buffer_size, CONFIG.context_size)
        _attn_masks_block = _attn_masks_block.view(
            -1, CONFIG.buffer_size, CONFIG.context_size
        )
        _seg = _tokens_block.shape[0]
        for _i in range(_seg):
            _df = pd.DataFrame(
                {
                    "tokens": _tokens_block[_i].tolist(),
                    "attention_mask": _attn_masks_block[_i].tolist(),
                }
            )
            _df.to_parquet(f"{CONST.PROCESSED_PATH}{_seg_num:03d}_00000.parquet")
            _seg_num += 1
            del _df
        del _tokens_block, _attn_masks_block

        _tokens_remainder = _tokens_remainder.view(-1, CONFIG.context_size)
        _attn_masks_remainder = _attn_masks_remainder.view(-1, CONFIG.context_size)
        _df = pd.DataFrame(
            {
                "tokens": _tokens_remainder.tolist(),
                "attention_mask": _attn_masks_remainder.tolist(),
            }
        )
        _df.to_parquet(f"{CONST.PROCESSED_PATH}{_seg_num:03d}_00000.parquet")
        _seg_num += 1
        del _df
        del _tokens_remainder, _attn_masks_remainder
    print("============\n Please update the tokenized file count number in const.py\n ============")
    print("New value:", _seg_num)

if __name__ == '__main__':
    main()
