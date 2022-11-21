import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import torch.nn as nn
import torch
import numpy as np
import random
from torchvision.models.densenet import *
from torchvision import transforms
from torchvision import  datasets
import torch.utils.data
import torch.optim as optim
import time

bs = 128

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

setup_seed(0)

device = torch.device('cuda')
normalization = transforms.Normalize((0.8388, 0.8388, 0.8388), (0.31613973, 0.31613973, 0.31613973))

train_data = datasets.ImageFolder(r'./oracle_fs/img/oracle_200_1_shot_strokes/train',transform=transforms.Compose([
    # transforms.RandomRotation(60),
    # transforms.RandomCrop(48, pad_if_needed=True),
    transforms.ToTensor(),
    normalization
]))
print(train_data.classes) #get label
train_loader = torch.utils.data.DataLoader(train_data, batch_size=bs, shuffle=True)

test_data = datasets.ImageFolder(r'./oracle_fs/img/oracle_200_1_shot/test',transform=transforms.Compose([
    transforms.ToTensor(),
    normalization
]))
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1024)

densenet = densenet161().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(densenet.parameters(), lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=5e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [11, 25], gamma=0.1)

def getStat(train_data):
    print('Compute mean and variance for training data.')
    print(len(train_data))
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X, _ in train_loader:
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy()), list(std.numpy())


def get_train_accuracy(model):
    size = 0
    correct = 0
    with torch.no_grad():
        for data in train_loader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            _, y_pred = torch.max(outputs.data, 1)
            size += y.size(0)
            correct += (y_pred == y).sum().item()

    print('Train Accuracy : %.2f %%' % (100 * correct / size))
    return(correct / size)


from tqdm.notebook import tqdm

max_epoch = 20

time_start = time.time()
    
for epoch in tqdm(range(max_epoch)):

    densenet.train()
    running_loss = 0.0
    
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = densenet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    if scheduler is not None:
        scheduler.step()
    
    print('Epoch %d Loss: %.3f' %(epoch + 1, running_loss / len(train_loader)), end='\t')


    if epoch >= 0:
        correct = 0
        total = 0
        densenet.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = densenet(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the test set: %.2f %%'
                % (100 * correct / total))
    scheduler.step()
    

print('Finished Training! Totally Training Time Cost',
        time.time() - time_start)

save the model
PATH = './densenet161_1.pth'
torch.save(densenet.state_dict(), PATH)


"""Evaluate Performance"""

net = densenet161()
net.load_state_dict(torch.load(PATH))
correct = 0
total = 0
densenet.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = densenet(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test set: %.2f %%'
        % (100 * correct / total))