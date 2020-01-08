import os
import regex
import json
import argparse
from pprint import pprint

import torch

from models import MyModel0
from data import VOCAB_DIC, MyDataset
from utils import color_print, get_text

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--device", default="cpu")

args = parser.parse_args()
args.device = torch.device(args.device)


def extract(text, label):
    data = ['', '', '', '']

    curr = ''
    curr_class = 0
    for i in range(len(text)):
        if label[i] > 0:
            curr += text[i]
            curr_class = label[i]
        else:
            if curr_class > 0 and len(data[curr_class - 1]) <= len(curr):
                data[curr_class - 1] = curr
            curr = ''
            curr_class = 0

    return {
        'company': data[0].strip(),
        'address': data[1].strip(),
        'date': data[2].strip(),
        'total': data[3].strip(),
    }


def validate(model, dataset, batch_size=1):
    model.eval()
    with torch.no_grad():
        keys, text, truth = dataset.get_data(batch_size=batch_size, device=args.device)
        pred = model(text)
        for i, key in enumerate(keys):
            print_text, label = dataset.data_dict[key]
            print_text_class = pred[:, i][: len(print_text)].cpu().numpy()

            color_print(print_text, print_text_class)

            extracted = extract(print_text, print_text_class)
            pprint(extracted)

            correct = extract(print_text, label)
            pprint(correct)

            print('\n')


def predict(device, model, fns, texts):
    maxlen = 1500

    texts = [s.strip().ljust(maxlen, " ") for s in texts]
    text_tensor = torch.zeros(maxlen, len(texts), dtype=torch.long)

    for i, text in enumerate(texts):
        text = text.upper()
        text_tensor[:, i] = torch.LongTensor([VOCAB_DIC.get(c, 0) for c in text])
    text_tensor = text_tensor.to(device)

    model.eval()
    with torch.no_grad():
        pred = model(text_tensor)

        for i, text in enumerate(texts, 0):
            print_text_class = pred[:, i][: len(text)].cpu().numpy()

            print(fns[i])
            # color_print(text, print_text_class)

            text_space = regex.sub(r"[\t\n]", " ", text).upper()
            extracted = extract(text_space, print_text_class)

            with open(fns[i].replace('txt', 'json'), 'w') as f:
                json.dump(extracted, f)

            # print(extracted)




def run_val(model):
    val_dataset = MyDataset(dict_path="data/val_data.pkl")
    validate(model, val_dataset, batch_size=1)


def run_predict_txt(model):
    root = 'test_data'
    for fn in sorted(os.listdir(root)):
        try:
            if not fn.endswith('txt'):
                continue
            fn = os.path.join(root, fn)
            text = get_text(fn)

            predict(args.device, model, [fn], [text])

        except Exception as e:
            print(str(e))
            print(fn)


if __name__ == "__main__":
    curr_model = MyModel0(len(VOCAB_DIC), 20, 512).to(args.device)
    curr_model.load_state_dict(torch.load('models/demo.pth', map_location=torch.device('cpu')))

    run_predict_txt(curr_model)
