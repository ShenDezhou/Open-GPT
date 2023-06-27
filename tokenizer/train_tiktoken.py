from educational import *
from tiktoken.load import dump_tiktoken_bpe

gpt2_pattern = (
        r"""'s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}]+| ?[\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
with open('shakespeare_input.txt', "r") as f:
    data = f.read()

enc = SimpleBytePairEncoding.train(data, vocab_size=1000, pat_str=gpt2_pattern)

dump_tiktoken_bpe(enc.mergeable_ranks, "e1k_base/e1k_base.ticktoken")
