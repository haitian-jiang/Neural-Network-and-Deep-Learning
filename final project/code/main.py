import torch
import numpy as np
import time
import random
import sys
from args import args
from utils import *
import datasets
import few_shot_eval
import resnet


criterion = torch.nn.CrossEntropyLoss()
last_update = 0

### main train function
def train(model, train_loader, optimizer, epoch, scheduler, mixup = False, mm = False):
    model.train()
    global last_update
    losses, total = 0., 0
    
    for batch_idx, (data, target) in enumerate(train_loader): 
        data, target = data.to(args.device), target.to(args.device)
        optimizer.zero_grad()

        if args.rotations: # generate self-supervised rotations for improved universality of feature vectors
            bs = data.shape[0] // 4
            target_rot = torch.LongTensor(data.shape[0]).to(args.device)
            target_rot[:bs] = 0
            data[bs:] = data[bs:].transpose(3,2).flip(2)
            target_rot[bs:2*bs] = 1
            data[2*bs:] = data[2*bs:].transpose(3,2).flip(2)
            target_rot[2*bs:3*bs] = 2
            data[3*bs:] = data[3*bs:].transpose(3,2).flip(2)
            target_rot[3*bs:] = 3

        if mixup:
            index_mixup = torch.randperm(data.shape[0])
            lam = random.random()            
            data_mixed = lam * data + (1 - lam) * data[index_mixup]
            output, _ = model(data_mixed)
            if args.rotations:
                output, output_rot = output
                loss = ((lam * criterion(output, target) + (1 - lam) * criterion(output, target[index_mixup])) + (lam * criterion(output_rot, target_rot) + (1 - lam) * criterion(output_rot, target_rot[index_mixup]))) / 2
            else:
                loss = lam * criterion(output, target) + (1 - lam) * criterion(output, target[index_mixup])
        else:
            output, _ = model(data)
            if args.rotations:
                output, output_rot = output
                loss = 0.5 * criterion(output, target) + 0.5 * criterion(output_rot, target_rot)                
            else:
                loss = criterion(output, target)

        # backprop loss
        loss.backward()
            
        losses += loss.item() * data.shape[0]
        total += data.shape[0]
        # update parameters
        optimizer.step()
        scheduler.step()

        length = len(train_loader)
        # print advances if at least 100ms have passed since last print
        if (batch_idx + 1 == length) or (time.time() - last_update > 0.1):
            if batch_idx + 1 < length:
                print("\r{:4d} {:4d} / {:4d} loss: {:.5f} time: {:s} lr: {:.5f} ".format(epoch, 1 + batch_idx, length, losses / total, format_time(time.time() - start_time), float(scheduler.get_last_lr()[0])), end = "")
            else:
                print("\r{:4d} loss: {:.5f} ".format(epoch, losses / total), end = '')
            last_update = time.time()

        if few_shot and total >= args.dataset_size and args.dataset_size > 0:
            break
            
    return { "train_loss" : losses / total}

# function to compute accuracy in the case of standard classification
def test(model, test_loader):
    model.eval()
    test_loss, accuracy, accuracy_top_5, total = 0, 0, 0, 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(args.device), target.to(args.device)
            output, _ = model(data)
            if args.rotations:
                output, _ = output
            test_loss += criterion(output, target).item() * data.shape[0]
            pred = output.argmax(dim=1, keepdim=True)
            accuracy += pred.eq(target.view_as(pred)).sum().item()

            # count total number of samples for averaging in the end
            total += target.shape[0]

    return { "test_loss" : test_loss / total, "test_acc" : accuracy / total, "test_acc_top_5" : accuracy_top_5 / total}

# function to train a model using args.epochs epochs
# at each args.milestones, learning rate is multiplied by args.gamma
def train_pipeline(model, loaders, mixup = False):
    global start_time
    start_time = time.time()

    if few_shot:
        train_loader, train_clean, val_loader, novel_loader = loaders
        for i in range(len(few_shot_meta_data["best_val_acc"])):
            few_shot_meta_data["best_val_acc"][i] = 0
    else:
        train_loader, val_loader, test_loader = loaders
    lr = args.lr

    for epoch in range(args.epochs + args.manifold_mixup):
        length = len(train_loader)

        if (args.cosine and epoch % args.milestones[0] == 0) or epoch == 0:
            optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = 0.9, weight_decay = 5e-4, nesterov = True)
            if args.cosine:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = args.milestones[0] * length)
                lr = lr * args.gamma
            else:
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = list(np.array(args.milestones) * length), gamma = args.gamma)

        train(model, train_loader, optimizer, (epoch + 1), scheduler, mixup = mixup, mm = epoch >= args.epochs)        
        
        if args.save_model != "" and not few_shot:
            if len(args.devices) == 1:
                torch.save(model.state_dict(), args.save_model)
            else:
                torch.save(model.module.state_dict(), args.save_model)
        
        if few_shot:
            res = few_shot_eval.update_few_shot_meta_data(model, train_clean, novel_loader, val_loader, few_shot_meta_data)
            for i in range(len(args.n_shots)):
                print("val-{:d}: {:.2f}%, nov-{:d}: {:.2f}% ({:.2f}%) ".format(args.n_shots[i], 100 * res[i][0], args.n_shots[i], 100 * res[i][2], 100 * few_shot_meta_data["best_novel_acc"][i]), end = '')
        else:
            test_stats = test(model, test_loader)
            print("test acc: {:.2f}%".format(100 * test_stats["test_acc"]))

    return few_shot_meta_data if few_shot else test_stats


### process main arguments
loaders, input_shape, num_classes, few_shot, top_5 = datasets.get_dataset(args.dataset)
### initialize few-shot meta data
if few_shot:
    num_classes, val_classes, novel_classes, elements_per_class = num_classes
    elements_val, elements_novel = [elements_per_class] * val_classes, [elements_per_class] * novel_classes
    elements_train = None

    val_runs = list(zip(*[few_shot_eval.define_runs(args.n_ways, s, args.n_queries, val_classes, elements_val) for s in args.n_shots]))
    val_run_classes, val_run_indices = val_runs[0], val_runs[1]
    novel_runs = list(zip(*[few_shot_eval.define_runs(args.n_ways, s, args.n_queries, novel_classes, elements_novel) for s in args.n_shots]))
    novel_run_classes, novel_run_indices = novel_runs[0], novel_runs[1]

    few_shot_meta_data = {
        "elements_train":elements_train,
        "val_run_classes" : val_run_classes,
        "val_run_indices" : val_run_indices,
        "novel_run_classes" : novel_run_classes,
        "novel_run_indices" : novel_run_indices,
        "best_val_acc" : [0] * len(args.n_shots),
        "best_val_acc_ever" : [0] * len(args.n_shots),
        "best_novel_acc" : [0] * len(args.n_shots)
    }

### prepare stats
run_stats = {}
if args.test_features != "":
    try:
        filenames = eval(args.test_features)
    except:
        filenames = args.test_features
    if isinstance(filenames, str):
        filenames = [filenames]
    features = [torch.load(fn) for fn in filenames]
    train_features = torch.cat([feat[0].to(args.device) for feat in features], dim=2)
    val_features = torch.cat([feat[1].to(args.device) for feat in features], dim=2)
    test_features = torch.cat([feat[2].to(args.device) for feat in features], dim=2)
    for i in range(len(args.n_shots)):
        val_acc, val_conf, test_acc, test_conf = few_shot_eval.evaluate_shot(i, train_features, val_features, test_features, few_shot_meta_data)
        print("{:d}-shot: {:.2f}% (± {:.2f}%)".format(args.n_shots[i], 100 * test_acc, 100 * test_conf))
    sys.exit()


model = resnet.ResNet18(args.feature_maps, input_shape, num_classes, few_shot, args.rotations).to(args.device)
if len(args.devices) > 1:
    model = torch.nn.DataParallel(model, device_ids = args.devices)

# training
test_stats = train_pipeline(model, loaders, mixup = args.mixup)

# assemble stats
for item in test_stats.keys():
    run_stats[item] = [test_stats[item].copy() if isinstance(test_stats[item], list) else test_stats[item]]

# print stats
if few_shot:
    for index in range(len(args.n_shots)):
        stats(np.array(run_stats["best_novel_acc"])[:,index], "{:d}-shot".format(args.n_shots[index]))
else:
    stats(run_stats["test_acc"], "test acc")
