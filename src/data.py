import random
from string import ascii_uppercase, digits, punctuation

import pickle
import numpy

import torch
from torch.utils import data

VOCAB = " ·\t\n" + ascii_uppercase + digits + punctuation
VOCAB_DIC = {ch: i for i, ch in enumerate(VOCAB, 0)}


class MyDataset(data.Dataset):

    def __init__(self, dict_path="data/data.pkl", val_size=50):
        with open(dict_path, 'rb') as f:
            data = pickle.load(f)

        data_items = list(data.items())
        random.shuffle(data_items)
        self.data_dict = dict(data_items)

        self.maxlen = 1500

    def get_data(self, batch_size=8, device='cpu'):

        samples = random.sample(self.data_dict.keys(), batch_size)

        texts = [self.data_dict[k][0] for k in samples]
        labels = [self.data_dict[k][1] for k in samples]

        texts = [s.ljust(self.maxlen, " ") for s in texts]
        labels = [
            numpy.pad(a, (0, self.maxlen - len(a)), mode="constant", constant_values=0) for a in labels
        ]

        text_tensor = torch.zeros(self.maxlen, batch_size, dtype=torch.long)
        for i, text in enumerate(texts):
            text_upper = text.upper()
            text_tensor[:, i] = torch.LongTensor([VOCAB_DIC.get(c, 0) for c in text_upper])

        truth_tensor = torch.zeros(self.maxlen, batch_size, dtype=torch.long)
        for i, label in enumerate(labels):
            truth_tensor[:, i] = torch.LongTensor(label)

        return samples, text_tensor.to(device), truth_tensor.to(device)


def main():
    dataset = MyDataset()
    keys, texts, truths = dataset.get_data()
    print(keys)


if __name__ == "__main__":
    main()
