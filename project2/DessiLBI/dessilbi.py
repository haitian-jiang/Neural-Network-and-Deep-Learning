import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from slbi_toolbox import SLBI_ToolBox
from collections import namedtuple
import matplotlib.pyplot as plt 
from slbi_adam import SLBI_ADAM_ToolBox

class LeNet(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.relu(self.conv3(x))
        x = x.view(-1, 120)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


train_dataset = torchvision.datasets.MNIST(root='datasets/MNIST', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='datasets/MNIST', train=False, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=512, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=512, shuffle=False)

model = LeNet().to(device)
layer_list = []
name_list = []
for name, p in model.named_parameters():
    name_list.append(name)
    if len(p.data.size()) == 4 or len(p.data.size()) == 2:
        layer_list.append(name)

lr = 0.01
kappa = 1
mu = 20
interval = 20

optimizer = SLBI_ADAM_ToolBox(model.parameters(), lr=lr, kappa=kappa, mu=mu, weight_decay=0)
optimizer.assign_name(name_list)
optimizer.initialize_slbi(layer_list)
criterion = nn.CrossEntropyLoss()

train_accs  = []
test_accs = []

def get_accuracy(test_loader):
    model.eval()
    correct = 0
    num = 0
    with torch.no_grad():
        for X,y in test_loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            pred = torch.max(out.data, 1)[1]
            correct += pred.eq(y).sum().item()
            num += X.shape[0]
        acc = correct / num 
    return acc 

for epoch in range(20):
    lr = lr * (0.1 ** (epoch //interval))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    losses = 0
    correct = 0
    num = 0
    model.train()
    model.train()
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        out = model(X)
        loss = criterion(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pred = torch.max(out.data, 1)[1]
        losses += loss.item()
        num += X.shape[0]
        correct += (y == pred).sum()
    losses /= num 

    train_acc = (correct / num).item()
    test_acc = get_accuracy(test_loader)
    train_accs.append(train_acc)
    test_accs.append(test_acc)
    correct = num = 0
    losses = 0
    optimizer.update_prune_order(epoch)

plt.figure()
plt.clf()
plt.plot(train_accs, label='train')
plt.plot(test_accs, label='test')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.savefig('lenet.png')

print('acc：',get_accuracy(test_loader) )
optimizer.prune_layer_by_order_by_name(80, 'conv3.weight', True)
print('acc after pruning conv3：',get_accuracy(test_loader))
optimizer.recover()
optimizer.prune_layer_by_order_by_list(80, ['conv3.weight','fc1.weight'], True)
print('acc after pruning conv3 and fc1：',get_accuracy(test_loader))
    