import os
import pickle
import nltk
import torch
import re


# def nltk_tokenizer(text):
#     # Use regular expressions to split text into words
#     tokens = re.findall(r'\b\w+\b', text.lower())
#     return tokens

def nltk_tokenizer(text):
    from nltk.tokenize import word_tokenize
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    return word_tokenize(text.lower())


class Vocabulary:
    def __init__(self, special_tokens=None, max_len=None):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self.max_len = max_len
        self.special_tokens = special_tokens or ["<unk>", "<pad>", "<sos>", "<eos>"]
        for token in self.special_tokens:
            self.add_word(token)

    def add_word(self, word):
        if self.max_len is None or len(self.word2idx) < self.max_len:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def __call__(self, words):
        if isinstance(words, list):
            return [self.word2idx.get(word, self.word2idx["<unk>"]) for word in words]
        return self.word2idx.get(words, self.word2idx["<unk>"])

    def add_words(self, words):
        for word in words:
            self.add_word(word)

    def __len__(self):
        return len(self.word2idx)


def build_vocab_from_dataset(dataset, vocab_file=None, label_first=False, max_len=None):
    if vocab_file and os.path.isfile(vocab_file):
        with open(vocab_file, 'rb') as f:
            vocab = pickle.load(f)
        return vocab
    vocab = Vocabulary(max_len=max_len)
    vocab.add_word('<unk>')
    for text, label in dataset:
        if label_first:
            label, text = text, label
        tokens = nltk_tokenizer(text)
        vocab.add_words(tokens)
    if vocab_file:
        with open(vocab_file, 'wb') as f:
            pickle.dump(vocab, f)
    return vocab


def load_glove_embeddings(path=None):
    if not path:
        path = 'data/glove/glove.6B.100d.txt'
    with open(path, 'r', encoding='utf-8') as f:
        embeddings = {}
        for line in f:
            values = line.split()
            word = values[0]
            vector = torch.tensor([float(val) for val in values[1:]], dtype=torch.float)
            embeddings[word] = vector
        return embeddings


def create_embedding_matrix(vocab, embedding_dim):
    glove_path = 'data/glove/glove.6B.{}d.txt'.format(embedding_dim)
    glove_embeddings = load_glove_embeddings(glove_path)
    embedding_matrix = torch.zeros(len(vocab.word2idx), embedding_dim)
    for word, idx in vocab.word2idx.items():
        if word in glove_embeddings:
            embedding_matrix[idx] = glove_embeddings[word]
        else:
            embedding_matrix[idx] = torch.randn(embedding_dim)
    return embedding_matrix
