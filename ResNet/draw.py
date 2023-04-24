from sys import _xoptions
from typing import Iterable
from matplotlib import pyplot
import matplotlib
import os
import utils


def plot_dual_axis_pic(ax1_data: Iterable, ax2_data: Iterable, save_path: os.PathLike, x_label: str, y1_label: str, y2_label: str) -> None:
    '''
    This function make a dual axes plot and save it to specified path

    parameters:
    `ax1_data`: a list of all data to be ploted to the first axis, each line should be passed in this form: `[label,X,Y]`.
    `ax2_data`: a list of all data to be ploted to the second axis, each line should be passed in this form: `[label,X,Y]`.
    `save_path`: where to save the generated picture
    `x_label`: the name of x axis
    `y1_label`: the name of the first y axis
    `y2_label`: the name of the second y axis
    '''
    fig, ax1 = pyplot.subplots(figsize=(15., 6.))
    ax1.set_title(".".join(os.path.split(save_path)[-1].split(".")[:-1]))
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y1_label)
    for label, X, Y in ax1_data:
        ax1.plot(X, Y, label=label)

    ax2 = ax1.twinx()
    ax2.set_ylabel(y2_label)
    for label, X, Y in ax2_data:
        ax2.plot(X, Y, label=label)

    fig.legend()

    pyplot.savefig(save_path, dpi=300, bbox_inches="tight")
    pyplot.clf()


def plot_single_axis_pic(data: Iterable, save_path: os.PathLike, x_label: str, y_label: str) -> None:
    '''
    This function make a dual axes plot and save it to specified path

    parameters:
    `data`: a list of all data to be ploted, each line should be passed in this form: `[label,X,Y]`.
    `save_path`: where to save the generated picture
    `x_label`: the name of x axis
    `y_label`: the name of the y axis
    '''
    fig, ax1 = pyplot.subplots(figsize=(15., 6.))
    ax1.set_title(".".join(os.path.split(save_path)[-1].split(".")[:-1]))
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    for label, X, Y in data:
        ax1.plot(X, Y, label=label)

    fig.legend()

    pyplot.savefig(save_path, dpi=300, bbox_inches="tight")
    pyplot.clf()


def draw_statistics(statistics):
    pics_path = os.path.join(utils.read_config("save_path"), "pics")
    if not os.path.exists(pics_path):
        os.makedirs(pics_path)
    plot_dual_axis_pic(
        ax1_data=[["train loss", statistics['x'], statistics['train_loss']],
                  ["val loss", statistics['x'], statistics['val_loss']]],
        ax2_data=[["train accurancy", statistics['x'], statistics['train_acc']],
                  ["val accurancy", statistics['x'], statistics['val_acc']],
                  ["best train accurancy"] + list(zip(*statistics['best_train_acc'][1:])),
                  ["best val accurancy"]+list(zip(*statistics['best_val_acc'][1:]))],
        x_label="epoch", y1_label="loss", y2_label="accurancy",
        save_path=os.path.join(pics_path, "loss and acc.png"))
    plot_single_axis_pic(data=[["learning rate", statistics['x'], statistics['lr']]],
                         x_label="epoch", y_label="learning rate", save_path=os.path.join(pics_path, "lr.png"))
