Deep Learning and Application course assignment, simple ResNet tryout

All configs are stored in `configs.json`, to run the code, please run `main.py`.

`main.py` accecpts a parameter `--config`, where you can specify your custom config file, but no configs can be made other than the config file.

You can modify the config file in `tasks` folder and use a batch file to automatically run multiple experiments, the configs of experiments described in the report are already in the `tasks` folder, and you can refer to `run_tasks.sh` to run the experiments.

The environment is as documented in `environment.txt`.

Tested and ran in my system with i7-12700H and Nvidia RTX 3060, under Ubuntu 20.04 LTS operating system with Linux core `5.10.102.1-microsoft-standard-WSL2`, CUDA version is 11.6.

If you have encountered and problem about file, please try modifying the relative paths in configs and `utils.py` to absolute paths.

The results will be saved to the `save` folder, make sure you have modeified the `expriment_name` entry in the config file. The results and logs of each running will be saved to the `f"save/{experiment_name}/{time}` folder, the `checkpoints` folder contains the checkpoint of best validating accuracy and normal save points in the training procedure, the `pics` folder contains the statistic figures generated after running, the `config.json` is a copy of the config the running uses, and the `statistics.json` is a copy of the statistic values of the running.

The `dataset_observe.ipynb` is an interactive notebook to observe the dataset, and `test.ipynb` is to test the performance of specified checkpoint on the testing set.

The `data` folder should contain the data used in the experiments, if not, they will be automatically downloaded when running.
