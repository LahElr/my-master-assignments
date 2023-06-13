import json
import os
import time
import torch
import logging


def get_cur_time() -> str:
    r'''
    This function returns a string of current time in format `%Y%m%d_%H%M%S`
    '''
    return time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))


config = json.load(open(r"./config.json", "r"))
config['save_path'] = os.path.join(config['save_path'], config["exp_name"],
                                   get_cur_time())
if not os.path.exists(config["save_path"]):
    os.makedirs(config['save_path'])

# This **** complicated thing is necessary just so that matplotlib would not mess up my logs
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(config["exp_name"])
logger_ch = logging.FileHandler(filename=f"{config['save_path']}/log.log",
                                mode="w")
logger_ch.setLevel(logging.NOTSET)
logger_ch.setFormatter(
    logging.Formatter(
        fmt=
        "%(asctime)s - %(name)s - %(levelname)-9s - %(filename)-8s : %(lineno)s line - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"))
logger.addHandler(logger_ch)
logger.setLevel(logging.NOTSET)
logger.propagate = False
logging.getLogger("matplotlib").propagate = False
logging.getLogger("PIL").propagate = False
logging.getLogger("urllib3").propagate = False


def read_config(url: str = None):
    r'''
    This function returns the requested config item, or the whole config file if `url` is None
    example of `url`: "train.optim.lr"
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
                    f"utils.read_config: your requested config item {url} does not exist!"
                )
        return target


def save_config():
    r"""
    This function would save the current config into the `config.save_path`
    """
    path_to_save = os.path.join(config["save_path"], "config.json")
    if not os.path.exists(config["save_path"]):
        os.makedirs(config['save_path'])
        save_config()
    json.dump(config, open(path_to_save, "w"))

if config["save_config"]:
    save_config()


def save_model(model: torch.nn.Module, name: str = None):
    r'''
    Save a model
    '''
    if not os.path.exists(config["save_path"]):
        os.makedirs(config['save_path'])
    if name is None:
        name = get_cur_time()
    torch.save(model.state_dict(),
               os.path.join(config['save_path'], f"{name}.ckpt"))
    # model.load_state_dict(torch.load(PATH))
