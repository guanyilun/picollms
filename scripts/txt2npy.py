"""convert txt to npy based on a given tokenizer"""
import argparse
from tokenizers import Tokenizer
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("ifiles", nargs="+")
parser.add_argument("--tokenizer", default="20B_tokenizer.json")

args = parser.parse_args()

tokenizer = Tokenizer.from_file(args.tokenizer)
for ifile in args.ifiles:
    with open(ifile, "r") as f:
        data = f.read()
    data = "\n".join([line for line in data.split("\n") if line])
    tokens = tokenizer.encode(data).ids
    ofile = ifile.replace(".txt", ".npy")
    print(f"writing {ofile}")
    with open(ofile, "wb") as f:
        np.save(f, np.array(tokens, dtype=np.uint16))