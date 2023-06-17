from tokenizers.models import BPE
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import Sequence, ByteLevel, UnicodeScripts, Whitespace
from tokenizers.trainers import BpeTrainer

class BPE_token(object):
    def __init__(self, vocab_size=50000):
        self.tokenizer = Tokenizer(BPE())
        self.tokenizer.normalizer = NFKC()
        self.tokenizer.pre_tokenizer = Sequence([UnicodeScripts(), ByteLevel()])
        self.tokenizer.decoder = ByteLevelDecoder()
        self.vocab_size = vocab_size

    def bpe_train(self, paths):

        trainer = BpeTrainer(vocab_size=self.vocab_size, show_progress=True, initial_alphabet=ByteLevel.alphabet(), special_tokens=[
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
            "<mask>"
        ])
        self.tokenizer.train(paths, trainer)

    def save_tokenizer(self, location, prefix=None):
        if not os.path.exists(location):
            os.makedirs(location)
        self.tokenizer.model.save(location, prefix)

if __name__ == "__main__":
    from pathlib import Path
    import os

    # the folder 'text' contains all the files
    paths = [str(x) for x in Path("./data/").glob("**/*.txt")]
    tokenizer = BPE_token()
    # train the tokenizer model
    tokenizer.bpe_train(paths)
    # saving the tokenized data in our specified folder
    save_path = 'gpt2'
    tokenizer.save_tokenizer(save_path)