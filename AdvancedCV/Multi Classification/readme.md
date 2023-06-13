* Thrid party libraries used in experiment:
  * Matplotlib: https://matplotlib.org/
  * NumPy: https://numpy.org/
  * Pillow: https://python-pillow.org/
  * Pytorch: https://pytorch.org/
  * scikit-learn: https://scikit-learn.org/stable/index.html
  * timm: https://timm.fast.ai/
  * Torchvision: https://pytorch.org/vision/stable/index.html
  * Transformers: https://huggingface.co/docs/transformers/index
  
* How to reimplement testing:
1. Dive into the "workspace" folder.
2. Establish the environment using `pip install requirments.txt`.
3. Run `python main.py`.
		* When first running the code, you may need to wait for Transformers to download necessary files.
4. "model_output.txt" should appear next to "model.ckpt" after running.
5. "save" folder should also appear, which contains logs of testing.

* How to reimplement training:
1. Dive into the "workspace" folder.
2. Establish the environment using `pip install requirments.txt`
3. Open "config.json"
  1. Change the value of "mode" to "train".
  2. Change the value of "data"->"path" to the path of your dataset folder that contains "img" and "split" folders.
  3. Make other modifications to "device", "train"->"batch_size" etc. for your need.
4. Run `python main.py`.
    * When first running the code, you may need to wait for Transformers to download necessary files.
5. The logs and results should show in: "save/{config["exp_name"]}/{time}" folder, which contains:
    * "log.log": the logs
    * "best.ckpt": the checkpoint of the model with the lowest val. loss.
    * "config.json": a copy of the config file used in training.
    * "epoch_{number}.ckpt": the checkpoint of the model after {number}-th epoch of training.
    * "pics" folder: contains figures describing performance changes during training.
    You can follow the steps in "How to reimplement testing" to test the checkpoints generated here, the result file would always show next to the checkpoint that generates it.
    
Do note that the change of hardware, batch size, seed, etc. may cause slight change of result.
Please note that the VAN model code was open sourced under Apache 2.0 License at: https://github.com/Visual-Attention-Network/VAN-Classification
Please contact me if anything abnormal occurs.
Codes in "draw.py" and "utils.py" are reused code from my previous works, thus have more complete comments.
