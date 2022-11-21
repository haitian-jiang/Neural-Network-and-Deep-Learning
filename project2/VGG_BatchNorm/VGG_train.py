from matplotlib import pyplot as plt
from torch import nn
import numpy as np
import torch
import os
import random
from tqdm.notebook import tqdm as tqdm
from IPython import display

from models.vgg import VGG_A
from models.vgg_bn import VGG_A_BN
from data.loaders import get_cifar_loader


def get_accuracy(model, loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            y_pred = torch.max(out.data, 1)[1]
            total += y.size(0)
            correct += (y_pred == y).sum()
    return (100 * float(correct) / total)
  

def set_random_seeds(seed_value=0, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu': 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        


def train(model, optimizer, criterion, train_loader, val_loader, scheduler=None, epochs_n=100, best_model_path=None):
    model.to(device)
    val_accuracy = [0] * epochs_n

    batches_n = len(train_loader)
    losses = []
    for epoch in tqdm(range(epochs_n), unit='epoch'):
        if scheduler is not None:
            scheduler.step()
        model.train()

        for data in train_loader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        model.eval()
        val_accuracy[epoch] = get_accuracy(model, val_loader)
        print("val acc:", val_accuracy[epoch])

    return losses, val_accuracy
  
device=torch.device('cuda')
set_random_seeds(seed_value=6666, device=device)
model = VGG_A()
model_BN = VGG_A_BN()
lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr = lr)
criterion = nn.CrossEntropyLoss()


train_loader = get_cifar_loader(train=True)
val_loader = get_cifar_loader(train=False)
losses, val_acc = train(model, optimizer, criterion, train_loader, val_loader, epochs_n=20)
losses_BN, val_acc_BN = train(model_BN, optimizer, criterion, train_loader, val_loader, epochs_n=20)

# %matplotlib inline
plt.plot(range(len(losses)), losses, label='no BN')
plt.plot(range(len(losses_BN)), losses_BN, label='with BN')
plt.legend()
plt.xlabel('step')
plt.ylabel('loss')
plt.title('Loss of VGG training')
plt.show()

# %matplotlib inline
plt.plot(range(len(val_acc)), val_acc, label='no BN')
plt.plot(range(len(val_acc_BN)), val_acc_BN, label='with BN')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('validation accuracy')
plt.title('Accuracy of VGG on validation')
plt.show()