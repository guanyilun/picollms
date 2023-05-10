"""various data loading utilities in general"""
import jax
# jax.config.update('jax_platform_name', 'cpu')  # debug

import jax.numpy as np
from typing import NamedTuple
from tokenizers import Tokenizer as HFTokenizer
import json


class Tokenizer(NamedTuple):
    """Thin wrapper around HFTokenizer, mostly to get rid of the `.ids`"""
    tokenizer: HFTokenizer
    @classmethod
    def from_file(cls, tokenizer_file):
        tokenizer = HFTokenizer.from_file(tokenizer_file)
        return cls(tokenizer=tokenizer)
    def encode(self, text):
        return self.tokenizer.encode(text).ids
    def decode(self, ids):
        return self.tokenizer.decode(ids)


class JSONLoader(NamedTuple):
    """jsonl dataloader"""
    file: str
    tokenizer: Tokenizer
    text_field: str

    @classmethod
    def from_file(cls, json_file, tokenizer=None, text_field="text"):
        if tokenizer is None:
            tokenizer = Tokenizer.from_file("20B_tokenizer.json")
        return cls(file=json_file, tokenizer=tokenizer, text_field=text_field)

    def get_dataloader(self, batch_size=8, block_size=256):
        def data_generator():
            with open(self.file, 'r') as f:
                x_batch = []
                y_batch = []
                l_batch = []
                for line in f:
                    text = json.loads(line)[self.text_field]
                    tokens = self.tokenizer.encode(text.strip())
                    prev, next = tokens[:-1], tokens[1:]
                    x, l = fold_into_blocks(prev, block_size)
                    y, _ = fold_into_blocks(next, block_size)
                    for (x_, y_, l_) in zip(x, y, l):
                        x_batch.append(x_)
                        y_batch.append(y_)
                        l_batch.append(l_)
                        if len(x_batch) == batch_size:
                            yield np.array(x_batch), np.array(y_batch), np.array(l_batch)
                            x_batch = []
                            y_batch = []
                            l_batch = []
        return data_generator()

def save_txt_as_npy(txt: str, tokenizer: Tokenizer, npy_file: str):
    """save a text file as a numpy array of token ids"""
    with open(txt, 'r') as f:
        text = f.read()
    tokens = tokenizer.encode(text)
    np.save(npy_file, tokens)

def load_npy_as_dataloader(npy_file: str, batch_size: int, block_size: int):
    """load a numpy array of token ids as a dataloader"""
    tokens = np.load(npy_file)
    prev, next = tokens[:-1], tokens[1:]
    x, l = fold_into_blocks(prev, block_size)
    y, _ = fold_into_blocks(next, block_size)
    def data_generator():
        for i in range(0, len(x), batch_size):
            yield x[i:i+batch_size], y[i:i+batch_size], l[i:i+batch_size]
    return data_generator()

def fold_into_blocks(arr, block_size, pad_value=0):
    """fold a 1D array into blocks of size block_size, padding `pad_value` if necessary.
    returns a tuple of (folded_arr, lengths) where lengths is a 1D array of the lengths of each block before
    padding."""
    arr = np.array(arr)
    nrows = int(np.ceil(len(arr) / block_size))
    last_row_length = len(arr) % block_size if len(arr) % block_size != 0 else block_size
    padded_arr = np.pad(arr, (0, block_size-last_row_length), mode='constant', constant_values=pad_value)
    folded_arr = padded_arr.reshape(nrows, block_size)
    lengths = np.array([block_size] * (nrows-1) + [last_row_length])
    return folded_arr, lengths

# save_txt_as_npy("data/shakespeare.txt", Tokenizer.from_file("20B_tokenizer.json"), "data/shakespeare.npy")
# dataloader = load_npy_as_dataloader("data/shakespeare.npy", 8, 128)
