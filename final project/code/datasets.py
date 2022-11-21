from torchvision import transforms, datasets
from args import args
import torch

def oracle(shot=5):
    normalization = transforms.Normalize((0.8388, 0.8388, 0.8388), (0.31613973, 0.31613973, 0.31613973))
    train_data = datasets.ImageFolder(f'../oracle_fs/img/oracle_200_{shot}_shot_strokes/train.more',transform=transforms.Compose([
        transforms.ToTensor(),
        normalization
    ]))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_data = datasets.ImageFolder(f'../oracle_fs/img/oracle_200_{shot}_shot/test',transform=transforms.Compose([
        transforms.ToTensor(),
        normalization
    ]))
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1024)
    loaders = (train_loader, test_loader, test_loader)
    return loaders, [3, 50, 50], 200, False, False

def oracle_fs(shot=5):
    normalization = transforms.Normalize((0.8388, 0.8388, 0.8388), (0.31613973, 0.31613973, 0.31613973))
    if args.dataset_path == "/":
        args.dataset_path = f'../oracle_fs/img/oracle_200_{shot}_shot_strokes/train'
    train_data = datasets.ImageFolder(args.dataset_path, transform=transforms.Compose([
        transforms.ToTensor(),
        normalization
    ]))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    train_clean = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=False)
    test_data = datasets.ImageFolder(f'../oracle_fs/img/oracle_200_{shot}_shot/test',transform=transforms.Compose([
        transforms.ToTensor(),
        normalization
    ]))
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1024)
    loaders = (train_loader, train_clean, test_loader, test_loader)
    return loaders, [3, 50, 50], (200,200,200,20), True, False

def get_dataset(dataset_name):
    if dataset_name.lower() in ["oracle1", "oracle3", "oracle5"]:
        return oracle(int(dataset_name[-1]))
    elif dataset_name.lower() in ["oraclefs1", "oraclefs3", "oraclefs5"]:
        return oracle_fs(int(dataset_name[-1]))
    else:
        print("Unknown dataset!")

print("datasets, ", end='')
