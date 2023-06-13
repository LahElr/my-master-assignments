from transformers import AutoFeatureExtractor, VanForImageClassification
import utils
import torch
import data
import model
import tqdm
import os
import numpy
import random
import draw

torch.manual_seed(utils.read_config("seed"))
torch.backends.cudnn.deterministic = True
random.seed(utils.read_config("seed"))

if utils.read_config("mode") == "train":
    utils.logger.info("Training mode started.")
    _, train_dataloader = data.get_traindata()
    _, val_dataloader = data.get_testdata("val")
    utils.logger.info("Data loaded.")
    net = model.VanForImageMultiClassification(
        utils.read_config("model.scale"))
    net = net.to(utils.read_config("device"))
    utils.logger.info("Model loaded.")

    train_loss_records = []
    train_acc_records = []
    val_loss_records = []
    val_acc_records = []
    val_x = []
    best_val_loss = float("inf")

    optimizer = model.optimizer_index[utils.read_config("train.optim.type")](
        params=[_ for _ in net.parameters() if _.requires_grad],
        **utils.read_config("train.optim.args"))

    loss_fn = model.MultiCrossEntropyLoss()
    loss_fn = loss_fn.to(utils.read_config("device"))

    process_bar = tqdm.trange(0, utils.read_config("train.epoch"))

    for epoch in process_bar:
        utils.logger.info(f"Starting epoch {epoch}.")
        train_loss_epoch = 0.0
        train_acc_epoch = numpy.array([0.0 for _ in range(6)])
        net.train()

        for batch in train_dataloader:
            input = batch[0]
            input = input.to(utils.read_config("device"))
            target = batch[1]
            target = target.to(utils.read_config("device"))

            optimizer.zero_grad()
            y = net(input)
            loss = loss_fn(y, target)
            loss.backward()
            optimizer.step()

            train_loss_epoch += loss.item()
            train_acc_epoch += numpy.array(model.calculate_acc(y, target))
        train_acc = train_acc_epoch / len(train_dataloader)
        train_loss = train_loss_epoch / len(train_dataloader)

        train_loss_records.append(train_loss)
        train_acc_records.append(train_acc)

        process_bar.set_description(
            f"""Training epoch {epoch}/{utils.read_config("train.epoch")} train_acc:{train_acc.mean()}"""
        )
        utils.logger.info(
            f"Training epoch{epoch} finished, train acc {train_acc.mean():0.4}, train loss {train_loss:0.4}."
        )

        if (epoch + 1) % utils.read_config("train.per_statistic_epoch") == 0:
            val_x.append(epoch)
            val_loss_epoch = 0.0
            val_acc_epoch = 0.0
            net.eval()
            for batch in val_dataloader:
                input = batch[0]
                input = input.to(utils.read_config("device"))
                target = batch[1]
                target = target.to(utils.read_config("device"))

                y = net(input)
                loss = loss_fn(y, target)
                val_loss_epoch += loss.item()
                val_acc_epoch += numpy.array(model.calculate_acc(y, target))

            val_acc = val_acc_epoch / len(val_dataloader)
            val_loss = val_loss_epoch / len(val_dataloader)

            val_loss_records.append(val_loss)
            val_acc_records.append(val_acc)
            utils.logger.info(
                f"Validating epoch{epoch} finished, val acc {val_acc.mean():0.4}, val loss {val_loss:0.4}."
            )

            if val_loss <= best_val_loss:
                utils.logger.info(f"New best validation loss!")
                best_val_loss = val_loss
                utils.save_model(net, "best")
        if (epoch + 1) % utils.read_config(
                "train.per_save_epoch") == 0 or epoch == utils.read_config(
                    "train.epoch") - 1:
            utils.save_model(net, f"epoch_{epoch}")
            utils.logger.info(f"Saved at epoch {epoch}")

    train_acc_records = numpy.stack(train_acc_records)  #epoch,6
    val_acc_records = numpy.stack(val_acc_records)
    statistics = {
        "train_loss": train_loss_records,
        "train_x": list(range(utils.read_config("train.epoch"))),
        "val_loss": val_loss_records,
        "val_x": val_x,
        "train_acc": train_acc_records.mean(axis=1).tolist(),
        "val_acc": val_acc_records.mean(axis=1).tolist(),
        "train_acc_full": train_acc_records,
        "val_acc_full": val_acc_records
    }
    draw.draw_statistics(statistics)

else:
    utils.logger.info("Testing mode started.")
    _, test_dataloader = data.get_testdata("test")
    utils.logger.info("Data loaded.")
    net = model.VanForImageMultiClassification(
        utils.read_config("model.scale"))
    state_dict = torch.load(utils.read_config("checkpoint"))
    net.load_state_dict(state_dict)
    net = net.to(utils.read_config("device"))
    utils.logger.info("Model loaded.")

    preds = []
    _ = os.path.split(utils.read_config("checkpoint"))
    output_path = os.path.join(_[0],
                               f"{'_'.join(_[1].split('.')[:-1])}_output.txt")
    process_bar = tqdm.tqdm(test_dataloader)
    net.eval()

    utils.logger.info("Running started.")
    with open(output_path, "w") as output_file:
        for batch in process_bar:
            input, _ = batch
            input = input.to(utils.read_config("device"))
            preds = net(input)  # 6 tensors of [bs,?]
            pred_labels = [_.argmax(1) for _ in preds]  # 6 tensors of [bs]
            pred_labels = torch.stack(pred_labels)  #6, bs
            for i in range(pred_labels.shape[1]):
                output_file.write(
                    " ".join(map(str, map(int, pred_labels[:, i].tolist()))) +
                    "\n")
    utils.logger.info(f"Running finished, results saved to {output_path}.")
