import re
import pandas as pd
from configure import *
import numpy as np
import pickle

# https://arxiv.org/pdf/1711.04810.pdf
class Tokenizer(object):
    def __init__(self):
        self.stoi = {}
        self.itos = {}
        SMI_REGEX_PATTERN = r"""(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"""
        self.regex = re.compile(SMI_REGEX_PATTERN)

    def __len__(self):
        return len(self.stoi)

    def tokenizer(self, smiles_string):
        tokens = [token for token in self.regex.findall(smiles_string)]
        return tokens

    def make_one_hot(self, data, max_len=120):
        vocab = self.stoi
        data_one_hot = np.zeros((len(data), max_len, len(vocab)))
        for i, smiles in enumerate(data):
            smiles = tokenizer.text_to_sequence(smiles)
            smiles = smiles[:max_len] + [0] * (max_len - len(smiles))
            for j, sequence in enumerate(smiles):
                if sequence is not vocab['<UNK>']:
                    data_one_hot[i, j, sequence] = 1
                else:
                    data_one_hot[i, j, vocab['<UNK>']] = 1
        return data_one_hot

    def fit_on_texts(self, smiles):
        vocab_ = set()
        for item in smiles:
            for letter in self.tokenizer(item):
                vocab_.add(letter)
        vocab = {}
        vocab['<PAD>'] = 0
        vocab['<UNK>'] = 1
        for i, letter in enumerate(vocab_):
            vocab[letter] = i + 2
        for i, s in enumerate(vocab):
            self.stoi[s] = i
        self.itos = {item[1]: item[0] for item in self.stoi.items()}

    def text_to_sequence(self, text):
        sequence = []
        for s in self.tokenizer(text):
            sequence.append(self.stoi[s])
        return sequence

    def sequence_to_text(self, sequence):
        return ''.join(list(map(lambda i: self.itos[i], sequence)))

    def predict_caption(self, sequence):
        caption = ''
        for i in sequence:
            if i == self.stoi['<PAD>']:
                break
            caption += self.itos[i]
        return caption


if __name__ == '__main__':
    data_train = pd.read_csv(train_data_dir)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data_train["SMILES"])
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
