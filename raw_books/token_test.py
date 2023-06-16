# This code validates all the characters in the text can be tokenized properly.
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("../generation/gpt2-private")

with open('daodejing.txt','r') as f:
    for line in f:
        tokens = tokenizer.tokenize(line)
        for tokenin in tokens:
            assert tokenizer.unk_token_id != tokenin

print('done')