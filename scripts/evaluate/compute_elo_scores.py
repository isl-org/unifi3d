"""
This scripts computes scores for each method using the Bradley-Terry model
assuming that we have data of pairwise preferences of the form
(Name of algorithm 0, Name of algorithm 1, Index of preferred (0 or 1))
"""

import argparse
from csv import reader
import os
from pathlib import Path, PosixPath
import sys

import torch
from torch.utils.data import Dataset, DataLoader
import aim


class CustomDataset(Dataset):
    """
    Custom Dataset class that is able to read a text file of the form
    (Name of algorithm 0, Name of algorithm 1, Index of preferred)
    """

    def __init__(self, data_dir, methods):
        self.methods = methods
        self.size = 0
        self.data = []

        methods_in_data = set()
        for data_file in data_dir:
            with open(data_file, "r") as dfile:
                csv_reader = reader(dfile)
                for line in csv_reader:
                    self.size += 1
                    self.data.append(line)
                    if self.methods is None:
                        methods_in_data.add(line[0])
                        methods_in_data.add(line[1])
        if self.methods is None:
            self.methods = list(methods_in_data)

    def __len__(self):

        return self.size

    def __getitem__(self, idx):

        datum = self.data[idx]

        method0_onehot = torch.zeros(len(self.methods))
        method0_index = self.methods.index(datum[0])
        method0_onehot[method0_index] = 1.0

        method1_onehot = torch.zeros(len(self.methods))
        method1_index = self.methods.index(datum[1])
        method1_onehot[method1_index] = 1.0

        preferred = torch.tensor(int(datum[2]), dtype=torch.long)

        return method0_onehot, method1_onehot, preferred


class ScoresModel(torch.nn.Module):

    def __init__(self, num_methods):
        super().__init__()

        # Parameterize the scores as the weights of a linear layer
        self.network = torch.nn.Linear(num_methods, 1, bias=False)

    def forward(self, x):
        return self.network(x)


def parse_args():
    parser = argparse.ArgumentParser(
        description="This script computes scores given a dataset of preferences.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--data_dir", nargs="+", type=Path, help="Paths to datasets")
    parser.add_argument(
        "--methods", nargs="+", type=str, help="Algorithms that we want to score"
    )
    parser.add_argument(
        "--exp_dir", type=Path, default="elo_scores", help="Folder to save/load scores"
    )
    parser.add_argument("--epochs", type=int, default=1)

    parser.add_argument(
        "--lr", type=float, default=0.001, help="Optimizer learning rate"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="L2 regularization on scores to stabilize computation",
    )
    parser.add_argument(
        "--batch_size", type=int, default=3, help="Batch size for training"
    )

    parser.add_argument("--use_cuda", action="store_true", help="Use CUDA")
    parser.add_argument("--save_model", action="store_true", help="Save trained scores")

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    if "--" in sys.argv:
        args, _ = parser.parse_known_args(sys.argv[sys.argv.index("--") + 1 :])
    else:
        args = parser.parse_args()
    print(args)
    return args


if __name__ == "__main__":

    args = parse_args()

    use_cuda = args.use_cuda and torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)

    dataset = CustomDataset(args.data_dir, args.methods)
    args.methods = dataset.methods
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    rater = ScoresModel(len(args.methods))
    optimizer = torch.optim.SGD(
        rater.parameters(), lr=args.lr, weight_decay=args.alpha, momentum=0.9
    )

    checkpoint_path = os.path.join(args.exp_dir, "network.tar")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        rater.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    training_loss = torch.nn.CrossEntropyLoss()

    run = aim.Run(experiment="compute-scores")
    args_dict = vars(args).copy()
    for key, value in args_dict.items():
        if isinstance(value, (Path, PosixPath)):
            args_dict[key] = str(value)
        if isinstance(value, list):
            new_val = ""
            for i in range(len(value)):
                new_val = new_val + "," + str(value[i])
            args_dict[key] = new_val
    run["hparams"] = args_dict

    for r in range(0, args.epochs):
        rater.train()

        for method0, method1, preferred in dataloader:
            optimizer.zero_grad()

            predictions0 = rater.forward(method0)
            predictions1 = rater.forward(method1)

            predictions = torch.cat((predictions0, predictions1), 1).cpu()
            out = training_loss(predictions, preferred)
            out.backward()
            optimizer.step()

            run.track(out.item(), name="training_loss")
            print(out.item())

        if args.save_model:
            torch.save(
                {
                    "model_state": rater.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                },
                checkpoint_path,
            )

    # Print results
    scores = rater.network.weight.detach().numpy()
    score_method = sorted(list(zip(scores[0], args.methods)), reverse=True)
    print("The scores for each method are as follows:")
    for i, (s, m) in enumerate(score_method):
        print(f"{i+1:2}  {m:30} {s}")
