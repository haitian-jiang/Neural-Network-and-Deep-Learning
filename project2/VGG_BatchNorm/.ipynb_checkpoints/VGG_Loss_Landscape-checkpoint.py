import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
import os
import random
from tqdm import tqdm as tqdm
from IPython import display

from VGG_BatchNorm.models.vgg_bn import VGG_A
from data.loaders import get_cifar_loader

# ## Constants (parameters) initialization
# device_id = [0,1,2,3]
num_workers = 14
batch_size = 128

# add our package dir to path 
module_path = os.path.dirname(os.getcwd())
home_path = module_path
figures_path = os.path.join(home_path, 'reports', 'figures')
models_path = os.path.join(home_path, 'reports', 'models')


device = torch.device('cuda')


# Initialize your data loader 
train_loader = get_cifar_loader(train=True)
val_loader = get_cifar_loader(train=False)




# This function is used to calculate the accuracy of model classification
def get_accuracy(model, loader):
    ## --------------------
    # Add code as needed
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
    ## --------------------


# Set a random seed to ensure reproducible results
def set_random_seeds(seed_value=0, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu': 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# We use this function to complete the entire
# training process. In order to plot the loss landscape,
# you need to record the loss value of each step.
# Of course, as before, you can test your model
# after drawing a training round and save the curve
# to observe the training
def train(model, optimizer, criterion, train_loader, val_loader, scheduler=None, epochs_n=100, best_model_path=None):
    model.to(device)
    learning_curve = [0] * epochs_n
    train_accuracy_curve = [0] * epochs_n
    val_accuracy_curve = [0] * epochs_n
    max_val_accuracy = 0
    max_val_accuracy_epoch = 0

    batches_n = len(train_loader)
    losses_list = []
    grads = []
    for epoch in tqdm(range(epochs_n), unit='epoch'):
        if scheduler is not None:
            scheduler.step()
        model.train()

        loss_list = []  # use this to record the loss value of each step
        grad_list = []  # use this to record the loss gradient of each step
        learning_curve[epoch] = 0  # maintain this to plot the training curve

        for data in train_loader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)
            # You may need to record some variable values here
            # if you want to get loss gradient, use
            # grad = model.classifier[4].weight.grad.clone()
            ## --------------------
            # Add your code
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            #
            ## --------------------


            loss.backward()
            optimizer.step()

        losses_list += loss_list
        grads.append(grad_list)
        # display.clear_output(wait=True)
        # f, axes = plt.subplots(1, 2, figsize=(15, 3))

        # learning_curve[epoch] /= batches_n
        # axes[0].plot(learning_curve)

        # Test your model and save figure here (not required)
        # remember to use model.eval()
        ## --------------------
        # Add code as needed
        #
        model.eval()
        # train_accuracy_curve[epoch] = get_accuracy(model, train_loader)
        val_accuracy_curve[epoch] = get_accuracy(model, val_loader)
        ## --------------------

    # plt.plot(range(epochs_n), train_accuracy_curve, c='g',label='Training')
    # plt.plot(range(epochs_n), val_accuracy_curve, c='b',label='Testing')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.savefig('acc.jpg')
    # plt.show()
    # plt.cla()

    return losses_list, val_accuracy_curve


# Train your model
# feel free to modify
epo = 20
loss_save_path = ''
grad_save_path = ''

set_random_seeds(seed_value=6666, device=device)
model = VGG_A()
lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr = lr)
criterion = nn.CrossEntropyLoss()
loss, grads = train(model, optimizer, criterion, train_loader, val_loader, epochs_n=epo)
np.savetxt(os.path.join(loss_save_path, 'loss.txt'), loss, fmt='%s', delimiter=' ')
np.savetxt(os.path.join(grad_save_path, 'grads.txt'), grads, fmt='%s', delimiter=' ')

# Maintain two lists: max_curve and min_curve,
# select the maximum value of loss in all models
# on the same step, add it to max_curve, and
# the minimum value to min_curve
min_curve = []
max_curve = []
## --------------------
# Add your code
#
#
#
#
## --------------------

# Use this function to plot the final loss landscape,
# fill the area between the two curves can use plt.fill_between()
def plot_loss_landscape():
    ## --------------------
    # Add your code
    #
    #
    #
    #
    ## --------------------
    pass