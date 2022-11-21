import argparse
import os
import random
import numpy as np

parser = argparse.ArgumentParser()

### hyperparameters
parser.add_argument("--batch-size", type=int, default=128, help="batch size")
parser.add_argument("--feature-maps", type=int, default=64, help="number of feature maps")
parser.add_argument("--lr", type=float, default="0.1", help="initial learning rate (negative is for Adam, e.g. -0.001)")
parser.add_argument("--epochs", type=int, default=40, help="total number of epochs")
parser.add_argument("--milestones", type=str, default="20", help="milestones for lr scheduler, can be int (then milestones every X epochs) or list. 0 means no milestones")
parser.add_argument("--gamma", type=float, default=0.1, help="multiplier for lr at milestones")
parser.add_argument("--cosine", action="store_true", help="use cosine annealing scheduler with args.milestones as T_max")
parser.add_argument("--mixup", action="store_true", help="use of mixup since beginning")
parser.add_argument("--rotations", action="store_true", help="use of rotations self-supervision during training")
parser.add_argument("--preprocessing", type=str, default="", help="preprocessing sequence for few shot, can contain R:relu P:sqrt E:sphering and M:centering")
parser.add_argument("--manifold-mixup", type=int, default="0", help="deploy manifold mixup as fine-tuning as in S2M2R for the given number of epochs")

### pytorch options
parser.add_argument("--device", type=str, default="cuda:0", help="device(s) to use, for multiple GPUs try cuda:ijk, will not work with 10+ GPUs")
parser.add_argument("--dataset-path", type=str, default='/', help="dataset path")
parser.add_argument("--dataset", type=str, default="", help="dataset to use")
parser.add_argument("--save-features", type=str, default="", help="save features to file")
parser.add_argument("--save-model", type=str, default="", help="save model to file")
parser.add_argument("--test-features", type=str, default="", help="test features and exit")
parser.add_argument("--seed", type=int, default=1, help="set random seed manually, and also use deterministic approach")

### few-shot parameters
parser.add_argument("--n-shots", type=str, default="1", help="how many shots per few-shot run, can be int or list of ints. In case of episodic training, use first item of list as number of shots.")
parser.add_argument("--n-runs", type=int, default=10000, help="number of few-shot runs")
parser.add_argument("--n-ways", type=int, default=5, help="number of few-shot ways")
parser.add_argument("--n-queries", type=int, default=20, help="number of few-shot queries")
parser.add_argument("--ncm-loss", action="store_true", help="use ncm output instead of linear")


args = parser.parse_args()

if args.dataset_path[-1] != '/':
    args.dataset_path += "/"

if args.device[:5] == "cuda:" and len(args.device) > 5:
    args.devices = []
    for i in range(len(args.device) - 5):
        args.devices.append(int(args.device[i+5]))
    args.device = args.device[:6]
else:
    args.devices = [args.device]

if args.seed == -1:
    args.seed = random.randint(0, 1000000000)

try:
    n_shots = int(args.n_shots)
    args.n_shots = [n_shots]
except:
    args.n_shots = eval(args.n_shots)

try:
    milestone = int(args.milestones)
    args.milestones = list(np.arange(milestone, args.epochs + args.manifold_mixup, milestone))
except:
    args.milestones = eval(args.milestones)
if args.milestones == [] and args.cosine:
    args.milestones = [args.epochs + args.manifold_mixup]

if args.gamma == -1:
    if args.cosine:
        args.gamma = 1.
    else:
        args.gamma = 0.1
    
print("args, ", end='')
