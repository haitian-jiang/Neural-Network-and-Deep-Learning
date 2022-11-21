import torch
import random
import numpy as np
import os
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib
from sklearn.linear_model import LinearRegression
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
matplotlib.use("Agg")

# 确保模型可复现
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class PartialDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, n_items=10):
        self.dataset = dataset
        self.n_items = n_items

    def __getitem__(self, num):
        return self.dataset.__getitem__(num)

    def __len__(self):
        return min(self.n_items, len(self.dataset))

    def partial(self):
        part_dataset = []
        for i in range(self.__len__()):
            part_dataset.append(self.__getitem__(i))
        return part_dataset


# def load_data_old(data_root="./data", batch_size=128, train=True, n_items=-1):
#     normalization = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
#     if train:
#         transform = transforms.Compose([
#             transforms.RandomCrop(32, padding=4),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             normalization
#         ])
#     else:
#         transform = transforms.Compose([
#             transforms.ToTensor(),
#             normalization
#         ])

#     dataset = torchvision.datasets.CIFAR10(root=data_root, train=train, transform=transform)
#     if n_items > 0:
#         dataset = PartialDataset(dataset, n_items).partial()
#     loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=16)
#     return loader




def load_data(data_root="./data", batch_size=128, train=True, n_items=-1):
    normalization = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    if train:
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalization
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalization
        ])

    dataset = torchvision.datasets.CIFAR10(root=data_root, train=train, transform=transform,download=False)
    if n_items > 0:
        dataset = PartialDataset(dataset, n_items).partial()
    if n_items > 0 or train:
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=16)
        return loader
    else: # val and test set
        val_set, test_set = torch.utils.data.random_split(dataset, [7680, 10000 - 7680])
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, num_workers=16)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=16)
        return val_loader, test_loader

def linearInd2Binary(ind,nLabels):
    n = len(ind)
    y = -torch.ones(n, nLabels)
    for i in range(n):
        y[i, ind[i]] = 1
    return y

def validation(model, loader):
    net, criterion, optimizer = model
    # evaluate the model
    # list_loader = [(x_.to(device), l_.to(device)) for (x_, l_) in loader] if not isinstance(loader, list) else loader
    net.eval()
    total_correct = total_loss = total_num = 0
    total_num = 0
    with torch.no_grad():
        for x, label in loader:
            x, label = x.to(device), label.to(device)
            pred = net(x)
            correct = torch.eq(pred.argmax(dim=1), label).float().sum().item()
            total_correct += correct
            total_num += x.size(0)
            loss = criterion(pred, label)
            total_loss += loss * x.size(0)
        acc = total_correct / total_num
        loss = total_loss / total_num
    return acc, loss


def train_model(epoch, train_loader, valid_loader, full_valid_ld, summery_writer, model, step, name):
    net, criterion, optimizer = model
    total_correct, total_loss, total_sample = 0, 0, 0
    for batch_idx, (x, label) in enumerate(train_loader):
        net.train()
        # forward pass
        x, label = x.to(device), label.to(device)
        pred = net(x)
        # label_binary = linearInd2Binary(label, 10).to(device)  # apply for MSELoss L1Loss
        # loss = criterion(pred, label_binary)
        loss = criterion(pred, label)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # get training accuracy per batch
        correct = torch.eq(pred.argmax(dim=1), label).float().sum().item()
        total_correct += correct
        total_sample += x.shape[0]
        total_loss += loss.item()*x.shape[0]
        if batch_idx % 50 == 0:
            step += 1
            valid_acc, valid_loss = validation(model, valid_loader)
            summery_writer.add_scalars(f'Loss/{name}', {'train': total_loss/total_sample, 'validation': valid_loss}, step)
            summery_writer.add_scalars(f'Acc/{name}', {'train': total_correct/total_sample, 'validation': valid_acc}, step)
            total_correct, total_loss, total_sample = 0, 0, 0


    # use the validation accuracy for the scheduler to decay learning rate
    valid_acc, valid_loss = validation(model, full_valid_ld)
    # summery_writer.add_scalar('Loss/valid', valid_loss, epoch)
    # summery_writer.add_scalar('Acc/valid', valid_acc, epoch)
    return valid_acc, step


def get_ILV_and_EFR(loss_list, train_error_list, val_error_list, stride=1, q=0.99, section_len=20):
    loss_LV = []
    train_acc_LV = []
    val_acc_LV = []

    loss_FR = []
    train_acc_FR = []
    val_acc_FR = []
    x_list = np.linspace(1, section_len, section_len).reshape((-1, 1))

    for index in range((len(loss_list) - section_len) // stride + 1):
        loss_section = loss_list[index:(index + section_len)]


        regression = LinearRegression().fit(x_list, loss_section)
        loss_LV.append(-regression.coef_[0] / np.mean(loss_section))
        loss_FR.append(np.sum(np.maximum(np.diff(loss_section, n=1), 0)) / section_len / np.mean(loss_section))

    q_list = np.logspace(1, len(loss_LV), num=len(loss_LV), base=q)
    q_list = q_list / np.sum(q_list)

    loss_ILV = np.sum(loss_LV * q_list)
    loss_EFR = np.sum(loss_FR * q_list)

    for index in range((len(train_error_list) - section_len) // stride + 1):
        train_section = train_error_list[index:(index + section_len)]
        val_section = val_error_list[index:(index + section_len)]

        regression = LinearRegression().fit(x_list, train_section)
        train_acc_LV.append(-regression.coef_[0] / np.mean(train_section))
        train_acc_FR.append(np.sum(np.maximum(np.diff(train_section, n=1), 0)) / section_len / np.mean(train_section))

        regression = LinearRegression().fit(x_list, val_section)
        val_acc_LV.append(-regression.coef_[0] / np.mean(val_section))
        val_acc_FR.append(np.sum(np.maximum(np.diff(val_section, n=1), 0)) / section_len / np.mean(val_section))

    q_list_acc = np.logspace(1, len(train_acc_LV), num=len(train_acc_LV), base=q)
    q_list_acc = q_list_acc / np.sum(q_list_acc)


    train_accuracy_ILV = np.sum(train_acc_LV * q_list_acc)
    val_accuracy_ILV = np.sum(val_acc_LV * q_list_acc)
    train_accuracy_EFR = np.sum(train_acc_FR * q_list_acc)
    val_accuracy_EFR = np.sum(val_acc_FR * q_list_acc)

    ILV = {"loss": loss_ILV, "train": train_accuracy_ILV, "val": val_accuracy_ILV}
    EFR = {"loss": loss_EFR, "train": train_accuracy_EFR, "val": val_accuracy_EFR}
    return ILV, EFR


def draw_loss(loss_list, experimentname="", filename='Loss.jpg', title='Loss', save=True):
    plt.figure(dpi=196)
    plt.plot(range(len(loss_list)), loss_list)
    plt.xlabel('Iteration')
    plt.ylabel('Loss of '+experimentname)
    plt.title(title)
    if save:
        plt.savefig(os.path.join('experiments', experimentname, filename))
    # plt.show()


def draw_acc(train_err, val_err, experimentname, filename='Accuracy.jpg', save=True):
    train_acc = [1-err for err in train_err]
    val_acc = [1-err for err in val_err]
    plt.figure(dpi=196)
    plt.plot(range(len(train_acc)), train_acc, c='g', label='Training')
    plt.plot(range(len(val_acc)), val_acc, c='b', label='Validation')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of '+experimentname)
    plt.legend()
    if save:
        plt.savefig(os.path.join('experiments', experimentname, filename))
    # plt.show()
