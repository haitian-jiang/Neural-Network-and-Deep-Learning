from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir='.tsbd-optim/')

import datetime
import copy
from model import *
from utils import *
setup_seed(6666)


train_loader = load_data()
iter_val_loader = load_data(train=False, n_items=512)
valid_loader, test_loader = load_data(train=False)

net = ResNet18().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, min_lr=1e-4)

loss_list = []
train_err = []
val_err = []

start_time = datetime.datetime.now()
step = 0
best_acc = 0
best_model = None
tolerent = 0
for epoch in range(50):
    print(epoch, optimizer.param_groups[0]["lr"])
    acc, step = train_model(epoch, train_loader, iter_val_loader, valid_loader, writer, (net, criterion, optimizer), step, 'optim-SGD_momentum')
    if acc > best_acc:
        tolerent = 0
        best_acc = acc
        best_model = copy.deepcopy(net)
    else:
        tolerent += 1
        if tolerent == 5:
            break
    scheduler.step(acc)
end_time = datetime.datetime.now()
print('Training time:%d' % (end_time - start_time).seconds)

acc = validation((best_model, criterion, optimizer), test_loader)
print(f"Test accuracy: {acc}")