import tiktoken
from tiktoken.load import load_tiktoken_bpe

e1k={
    "name": "e1k_base",
    "explicit_n_vocab": 1001,
    "pat_str": r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
    "mergeable_ranks": load_tiktoken_bpe("e1k_base/e1k_base.ticktoken"),
    "special_tokens": {"<|endoftext|>": 1000},
}
enc = tiktoken.Encoding(**e1k)
ts = enc.encode("First, you know Caius Marcius is chief enemy to the people.<|endoftext|>", allowed_special="all")
print(ts)
for line in open('shakespeare_input.txt'):
    print(enc.encode(line, allowed_special="all"))
