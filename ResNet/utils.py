import time
import json
import os
from typing import Any
import utils
import numpy
import random
import torch
import logging
import argparse

'''When imported in jupyter, the args would confict. Must tell if it is running in jupyter or not first before using argparse
'''
def is_in_interactive() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True #Jupyter
        elif shell == 'TerminalInteractiveShell':
            return False #IPython
        else:
            return False
    except NameError:
        return False

if is_in_interactive():
    args = argparse.Namespace(config=None)
else:
    parser = argparse.ArgumentParser(description="specify configs.json")
    parser.add_argument("-c","--config",default=None,required=False)
    args = parser.parse_args()

# -------config-------


def get_cur_time() -> str:
    '''
    This function returns a string of current time in format `%Y%m%d_%H%M%S`
    '''
    return time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))

if args.config is None:
    config = json.load(open(r"./configs.json", "r"))
else:
    config = json.load(open(args.config,"r"))
config['save_path'] = os.path.join(
    config['save_path'], config["expriment_name"], get_cur_time())

os.makedirs(config["save_path"])

if config["save_config"]:
    with open(os.path.join(config["save_path"], "config.json"), "w") as json_file:
        json.dump(config, json_file)


def read_config(url: str = None) -> Any:
    '''
    This function returns the requested config item, or the whole config file if `url` is None

    an example url: `"train.optimizer.lr"`
    '''
    if url is None:
        return config
    else:
        url = url.split(".")
        target = config
        for item in url:
            try:
                target = target[item]
            except KeyError:
                raise KeyError(
                    f"utils.read_config: your requested config item {item} of {target} does not exist!")
        return target


def save_config() -> None:
    """
    This function would save the current config into the `config.save_path`
    """
    path_to_save = os.path.join(config["save_path"], "config.json")
    if not os.path.exists(config["save_path"]):
        os.makedirs(config['save_path'])
        save_config()
    json.dump(config, open(path_to_save, "w"))

# -------seed-------


def set_seed_for_all(seed: int = None) -> None:
    '''
    This function sets seed for `random`, `numpy`, and `torch`.
    '''
    if seed is None:
        seed = utils.read_config("seed")
    random.seed(seed)  # seed for module random
    numpy.random.seed(seed)  # seed for numpy
    torch.manual_seed(seed)  # seed for PyTorch CPU
    torch.cuda.manual_seed(seed)  # seed for current PyTorch GPU
    torch.cuda.manual_seed_all(seed)  # seed for all PyTorch GPUs


set_seed_for_all()

# -------saving and loading model-------

if not os.path.exists(os.path.join(read_config("save_path"), "checkpoints")):
    os.makedirs(os.path.join(read_config("save_path"), "checkpoints"))


def save_checkpoint(net: torch.nn.Module, acc: float = None, epoch: int = None, name=None) -> None:
    '''
    This function saves a chekcpoint

    parameters:
    `net`: the model to be saved
    `acc`: optional, the acc of this checkpoint
    `epoch`: optional, at which epoch this checkpoint is made
    `name`: optional, the name of the model or checkpoint
    '''
    utils.logger.info(f"saving checkpoint at epoch {epoch}, of acc {acc}.")
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
    }
    if name is None:
        name = "net"
    name = f"{name}_{epoch}_{acc:.2f}.pth"
    name = os.path.join(read_config("save_path"), "checkpoints", name)
    torch.save(state, name)

def delete_checkpoint(epoch:int,acc:float,name:str=None)->None:
    '''This function can delete the specified checkpoint before this, to save storge
    '''
    if name is None:
        name = "net"
    name = f"{name}_{epoch}_{acc:.2f}.pth"
    name = os.path.join(read_config("save_path"), "checkpoints", name)
    os.remove(name)

def load_checkpoint(net:torch.nn.Module,path:str)->None:
    '''This function load the `pth` file into `net`
    '''
    with open(path,"rb") as pth_file:
        state = torch.load(pth_file)
    net.load_state_dict(state['net'])

# -------logging-------


logging.basicConfig(filename=os.path.join(read_config("save_path"), "log.log"), filemode="w",
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - [%(levelname)s]: %(message)s')

logger = logging.getLogger("ResNet")  # This is the logger for this project
logger.setLevel(logging.DEBUG)

# logger.info("\n"+"-"*7+"New log started"+"-"*7+"\n")
