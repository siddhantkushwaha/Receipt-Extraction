import argparse

import torch
from torch import nn, optim

from data import VOCAB_DIC, MyDataset
from models import MyModel0

from utils import color_print

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--device", default="cuda")
parser.add_argument("-b", "--batch_size", type=int, default=16)
parser.add_argument("-e", "--max_epoch", type=int, default=2000)
parser.add_argument("-v", "--val-at", type=int, default=100)
args = parser.parse_args()
args.device = torch.device(args.device)


def main():
    embed_size = 20
    hidden_size = 512
    model = MyModel0(vocab_size=len(VOCAB_DIC), embed_size=embed_size, hidden_size=hidden_size).to(args.device)

    train_dataset = MyDataset(dict_path="data/train_data.pkl")
    val_dataset = MyDataset(dict_path="data/val_data.pkl")

    criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.1, 1, 1.2, 0.8, 5], device=args.device))
    optimizer = optim.Adam(model.parameters())

    for i in range(args.max_epoch // args.val_at):
        train(
            model,
            train_dataset,
            criterion,
            optimizer,
            (i * args.val_at + 1, (i + 1) * args.val_at + 1),
            args.batch_size,
        )
        validate(model, val_dataset)

    torch.save(model.state_dict(), 'models/model_name.pth')


def train(model, dataset, criterion, optimizer, epoch_range, batch_size):
    model.train()

    for epoch in range(*epoch_range):
        optimizer.zero_grad()

        keys, text, truth = dataset.get_data(batch_size=batch_size, device=args.device)
        pred = model(text)

        loss = criterion(pred.view(-1, 5), truth.view(-1))
        loss.backward()

        optimizer.step()

        print("#{:04d} | Loss: {:.4f}".format(epoch, loss.item()))


def validate(model, dataset, batch_size=1):
    model.eval()
    with torch.no_grad():
        keys, text, truth = dataset.get_data(batch_size=batch_size, device=args.device)
        pred = model(text)
        for i, key in enumerate(keys):
            print_text, _ = dataset.data_dict[key]
            print_text_class = pred[:, i][: len(print_text)].cpu().numpy()

            color_print(print_text, print_text_class)


if __name__ == "__main__":
    main()
