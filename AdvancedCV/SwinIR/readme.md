Contains code only.

How to run:
1. dive into the workspace folder
2. fetch the pretrained checkpoint at: https://download.openmmlab.com/mmediting/swinir/swinir_x4s64w8d4e60_8xb4-lr2e-4-500k_div2k-d6622d03.pth
3. put data in the data folder
4. dive into the jupyter file and change all file paths for your convience
5. dive into the swinir_config_base.py, change all file paths for your convience
6. run the jupyter file cell by cell
	1. If you are not in google colab, donot run the first cell
7. to generate the result of real-world test set
	1. dive into the swinir_config_base.py
	2. change the ...data/test/LQ... part to .../data/test_real...
	3. rerun the 8th cell in the jupyter file

My codes were run on google colab with NVIDIA Tesla V100 environment. Errors may occur in different environment, please adjust the parameters for your convience.
Do kindly note that results may differ with different random seeds or environments. Please contact me if anything went wrong. Thank you.
