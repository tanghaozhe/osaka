import re
import pandas as pd

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

    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            sequence = self.text_to_sequence(text)
            sequences.append(sequence)
        return sequences

    def sequence_to_text(self, sequence):
        return ''.join(list(map(lambda i: self.itos[i], sequence)))

    def sequences_to_texts(self, sequences):
        texts = []
        for sequence in sequences:
            text = self.sequence_to_text(sequence)
            texts.append(text)
        return texts

    def predict_caption(self, sequence):
        caption = ''
        for i in sequence:
            if i == self.stoi['<PAD>']:
                break
            caption += self.itos[i]
        return caption

    def predict_captions(self, sequences):
        captions = []
        for sequence in sequences:
            caption = self.predict_caption(sequence)
            captions.append(caption)
        return captions

if __name__ == '__main__':
    data = pd.read_csv("./data/chembl_500k_train.csv")
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data["SMILES"].tolist())
    tokenizer.text_to_sequence("Cc1cc(ccc1C(=O)c2ccccc2Cl)N3N=CC(=O)NC3=O")