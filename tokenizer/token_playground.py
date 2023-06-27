from transformers import GPT2Tokenizer, CpmTokenizer, XLNetTokenizer, AutoTokenizer, BertTokenizer
# loading tokenizer from the saved model path
tokenizer = BertTokenizer.from_pretrained("./wp_21k_mul")
# tokenizer = GPT2Tokenizer.from_pretrained("./gpt2-50k")
# tokenizer.add_special_tokens({
#   "eos_token": "</s>",
#   "bos_token": "<s>",
#   "unk_token": "<unk>",
#   "pad_token": "<pad>",
#   "mask_token": "<mask>"
# })

dao_1 = "道可道，⾮常道。名可名，⾮常名。⽆名天地之始；有名万物之⺟。故常⽆欲，以观其妙；常有欲，以观其徼。此两者，同出⽽异名，同谓之⽞。⽞之⼜⽞，衆妙之⻔。"
string_tokenized = tokenizer.encode(dao_1)
print(f"len of string: {len(dao_1)}, len of tokens: {len(string_tokenized)}")
print(tokenizer.convert_ids_to_tokens(string_tokenized))

en_1 = "<s>These types represent all the different kinds of input that a Tokenizer accepts when using encode_batch().</s>"
string_tokenized = tokenizer.encode(en_1)
print(f"len of string: {len(en_1)}, len of tokens: {len(string_tokenized)}")
print(tokenizer.convert_ids_to_tokens(string_tokenized))
