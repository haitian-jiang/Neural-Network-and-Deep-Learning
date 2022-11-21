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
            y_pred = torch.max(out.data, 1)[1]  # TODO
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
        
def train(model, optimizer, criterion, train_loader, val_loader, epochs_n=100, best_model_path=None):
    model.to(device)
    losses_list, grads, betas = [], [], []
    last_grad, last_param = 0, 0
    for epoch in tqdm(range(epochs_n), unit='epoch'):
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
            losses_list.append(loss.item())
            grad = model.classifier[4].weight.grad.clone()
            param = model.classifier[4].weight.clone()
            grad_diff = torch.norm(grad-last_grad).item()
            grads.append(grad_diff)
            betas.append(grad_diff/(torch.norm(param-last_param).item()+1e-10))
            last_grad, last_param = grad, param
    return losses_list, grads, betas

device=torch.device('cuda')
set_random_seeds(seed_value=6666, device=device)
lrs = [2e-3, 1e-3, 5e-4, 1e-4]
criterion = nn.CrossEntropyLoss()
train_loader = get_cifar_loader(train=True)
val_loader = get_cifar_loader(train=False)

all_losses, all_losses_bn = [], []
all_grads, all_grads_bn = [], []
all_betas, all_betas_bn = [], []
for lr in lrs:
    model = VGG_A()
    model_bn = VGG_A_BN()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer_bn = torch.optim.Adam(model_bn.parameters(), lr=lr)
    print(f"training vgg with lr {lr}")
    losses, grads, betas = train(model, optimizer, criterion, train_loader, val_loader, epochs_n=20)
    print(f"training bn with lr {lr}")
    losses_BN, grads_bn, betas_bn = train(model_bn, optimizer_bn, criterion, train_loader, val_loader, epochs_n=20)
    all_losses.append(losses)
    all_losses_bn.append(losses_BN)
    all_grads.append(grads)
    all_grads_bn.append(grads_bn)
    all_betas.append(betas)
    all_betas_bn.append(betas_bn)
    
a_losses = np.array(all_losses).T.tolist()
min_losses = [min(i) for i in a_losses]
max_losses = [max(i) for i in a_losses]
a_losses_bn = np.array(all_losses_bn).T.tolist()
min_losses_bn = [min(i) for i in a_losses_bn]
max_losses_bn = [max(i) for i in a_losses_bn]


# %matplotlib inline
plt.rcParams["figure.figsize"] = (20,10)
plt.plot(range(len(min_losses)-50), min_losses[50:], c='g',linewidth=0.3)
plt.plot(range(len(min_losses)-50), max_losses[50:], c='g',linewidth=0.3)
plt.plot(range(len(min_losses)-50), min_losses_bn[50:], c='r',linewidth=0.3)
plt.plot(range(len(min_losses)-50), max_losses_bn[50:], c='r',linewidth=0.3)
plt.fill_between(range(len(min_losses)-50), max_losses[50:], min_losses[50:], facecolor='g', alpha=0.3, label='no BN')
plt.fill_between(range(len(min_losses)-50), max_losses_bn[50:], min_losses_bn[50:], facecolor='r', alpha=0.3, label='with BN')
plt.xlabel('steps')
plt.ylabel('loss')
plt.title('Loss landscape')
plt.legend()
plt.show()


grad_diff_T = np.array(grad_diff).T.tolist()
grad_diff_bn_T = np.array(grad_diff_bn).T.tolist()
min_grad = list(map(min, grad_diff_T))
max_grad = list(map(max, grad_diff_T))
min_grad_bn = list(map(min, grad_diff_bn_T))
max_grad_bn = list(map(max, grad_diff_bn_T))

# %matplotlib inline
plt.rcParams["figure.figsize"] = (20,10)
plt.plot(range(len(min_grad)-50), min_grad[50:], c='g',linewidth=0.3)
plt.plot(range(len(min_grad)-50), max_grad[50:], c='g',linewidth=0.3)
plt.plot(range(len(min_grad)-50), min_grad_bn[50:], c='r',linewidth=0.3)
plt.plot(range(len(min_grad)-50), max_grad_bn[50:], c='r',linewidth=0.3)
plt.fill_between(range(len(min_grad)-50), max_grad[50:], min_grad[50:], facecolor='g', alpha=0.3, label='no BN')
plt.fill_between(range(len(min_grad)-50), max_grad_bn[50:], min_grad_bn[50:], facecolor='r', alpha=0.3, label='with BN')
plt.xlabel('steps')
plt.ylabel('grad pred')
plt.title('Gradient predictiveness')
plt.legend()
plt.show()


betas_T = np.array(all_betas).T.tolist()
betas_bn_T = np.array(all_betas_bn).T.tolist()
min_beta = list(map(min, betas_T))
max_beta = list(map(max, betas_T))
min_beta_bn = list(map(min, betas_bn_T))
max_beta_bn = list(map(max, betas_bn_T))

%matplotlib inline
plt.rcParams["figure.figsize"] = (20,10)
plt.plot(range(len(min_beta)-50), min_beta[50:], c='g',linewidth=0.3)
plt.plot(range(len(min_beta)-50), max_beta[50:], c='g',linewidth=0.3)
plt.plot(range(len(min_beta)-50), min_beta_bn[50:], c='r',linewidth=0.3)
plt.plot(range(len(min_beta)-50), max_beta_bn[50:], c='r',linewidth=0.3)
plt.fill_between(range(len(min_beta)-50), max_beta[50:], min_beta[50:], facecolor='g', alpha=0.3, label='no BN')
plt.fill_between(range(len(min_beta)-50), max_beta_bn[50:], min_beta_bn[50:], facecolor='r', alpha=0.3, label='with BN')
plt.xlabel('steps')
plt.ylabel('β-smooth')
plt.title('Loss landscape')
plt.legend()
plt.show()