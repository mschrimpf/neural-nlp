import logging
import os
import sys

import fire
import torch
import torch.nn.functional as F
from io import open
from pathlib import Path

from neural_nlp.models import model_pool
from neural_nlp.models.implementations import BrainModel


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        path = Path(path)
        self.test = self.tokenize(path / 'test.txt')

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ['<eos>']
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids


def language_modeling(model_identifier, data='wikitext-2', device='cpu', bptt=35):
    # cf. https://github.com/pytorch/examples/blob/90738a76837d04e6de1403962acd21df5fbb820c/word_language_model/main.py

    data_path = Path(__file__).parent.parent.parent / 'ressources' / 'ml' / data
    assert data_path.exists(), f"{data_path} does not exist"
    corpus = Corpus(data_path)

    def batchify(data, batch_size):
        # Work out how cleanly we can divide the dataset into batch_size parts.
        nbatch = data.size(0) // batch_size
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * batch_size)
        # Evenly divide the data across the batch_size batches.
        data = data.view(batch_size, -1).t().contiguous()
        return data.to(device)

    eval_batch_size = 10
    test_data = batchify(corpus.test, eval_batch_size)
    criterion = torch.nn.CrossEntropyLoss()

    def get_batch(source, i):
        seq_len = min(bptt, len(source) - 1 - i)
        data = source[i:i + seq_len]
        target = source[i + 1:i + 1 + seq_len].view(-1)
        return data, target

    model = model_pool[model_identifier]
    model.mode = BrainModel.Modes.language_modeling

    total_loss = 0.
    ntokens = len(corpus.dictionary)
    with torch.no_grad():
        for i in range(0, test_data.size(0) - 1, bptt):
            data, targets = get_batch(test_data, i)
            output = model(data)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    test_loss = total_loss / (len(test_data) - 1)
    print(test_loss)


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    fire.Fire()
