import os
import tqdm

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import data
import models
import utils
import draw
import json

utils.logger.info("start loading data")

trainset = data.NewCifer10("train", data.transform_train)
valset = data.NewCifer10("val", data.transform_test)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=utils.read_config(
    "train.batch_size"), shuffle=True, num_workers=utils.read_config("data.num_workers"))
valloader = torch.utils.data.DataLoader(valset, batch_size=utils.read_config(
    "train.batch_size"), shuffle=False, num_workers=utils.read_config("data.num_workers"))

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

device = utils.read_config("device")

statistics = utils.read_config("statistics")

# Training


def train_epoch(epoch, net, criterion, trainloader, scheduler=None):
    utils.logger.info(f'{epoch}')
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if (batch_idx+1) % 50 == 0:
            utils.logger.info("iteration : %3d, loss : %0.4f, accuracy : %2.2f" % (
                batch_idx+1, train_loss/(batch_idx+1), 100.*correct/total))
    if scheduler is not None:
        scheduler.step()
    return train_loss/(batch_idx+1), correct/total


utils.logger.info("start loading model")

net = models.__dict__[utils.read_config("model.name")]().to(device)
# net = models.ResNet18().to(device)
loss = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(net.parameters(), lr=utils.read_config("train.optimizer.lr"), momentum=utils.read_config("train.optimizer.momentum"), weight_decay=(
    utils.read_config("train.optimizer.weight_decay") if utils.read_config("train.optimizer.use_weight_decay") else 0.0))

if utils.read_config("train.scheduler.use_scheduler"):
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=utils.read_config(
        "train.epoch"), eta_min=0)
else:
    scheduler = None

utils.logger.info("start training")

process_bar = tqdm.trange(0, utils.read_config("train.epoch"))

for epoch in process_bar:
    this_epoch_lr = optimizer.state_dict()['param_groups'][0]['lr']
    train_loss, train_acc = train_epoch(
        epoch, net, loss, trainloader, scheduler)

    process_bar.set_description(
        f"""Training epoch {epoch}/{utils.read_config("train.epoch")} train_acc:{train_acc}""")

    val_loss = 0.0
    val_acc = 0.0
    # net_has_been_saved = False
    if epoch % utils.read_config("train.per_statistic_epoch") == 0 or epoch == utils.read_config("train.epoch"):
        statistics["x"].append(epoch)
        statistics["lr"].append(this_epoch_lr)
        statistics["train_acc"].append(train_acc)
        statistics["train_loss"].append(train_loss)
        if train_acc >= statistics["best_train_acc"][-1][1]:
            statistics["best_train_acc"].append((epoch, train_acc))
        val_loss, val_acc = models.test(epoch, net, loss, valloader)
        statistics["val_acc"].append(val_acc)
        statistics["val_loss"].append(val_loss)
        if val_acc >= statistics["best_val_acc"][-1][1]:
            utils.save_checkpoint(net, val_acc, epoch, "best")
            if len(statistics["best_val_acc"]) >= 3:
                utils.delete_checkpoint(
                    statistics["best_val_acc"][-2][0], statistics["best_val_acc"][-2][1], "best")
            statistics["best_val_acc"].append((epoch, val_acc))
            # net_has_been_saved = True

    if epoch % utils.read_config("train.per_save_epoch") == 0 or epoch == utils.read_config("train.epoch")-1:
        utils.save_checkpoint(net, val_acc, epoch, "normal")

    utils.logger.info(("Epoch : %3d, training loss : %0.4f, training accuracy : %2.2f, test loss " +
                      ": %0.4f, test accuracy : %2.2f") % (epoch, train_loss, train_acc, val_loss, val_acc))

draw.draw_statistics(statistics)

with open(os.path.join(utils.read_config("save_path"), "statistics.json"), "w") as json_file:
    json.dump(statistics, json_file)
