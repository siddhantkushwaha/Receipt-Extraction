import argparse

import torch

from data import VOCAB_DIC
from models import MyModel0

from test import predict

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--device", default="cpu")

args = parser.parse_args()
args.device = torch.device(args.device)


def check(model):
    with open('data/images/img1.txt', 'r') as f:
        text = f.read()

    predict(args.device, model, ['1'], [text])


if __name__ == "__main__":
    curr_model = MyModel0(len(VOCAB_DIC), 20, 512).to(args.device)
    curr_model.load_state_dict(torch.load('models/demo.pth', map_location=torch.device('cpu')))

    check(curr_model)
