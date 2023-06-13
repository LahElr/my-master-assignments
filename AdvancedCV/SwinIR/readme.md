Contains code only.

How to run:
	dive into the workspace folder
	fetch the pretrained checkpoint at: https://download.openmmlab.com/mmediting/swinir/swinir_x4s64w8d4e60_8xb4-lr2e-4-500k_div2k-d6622d03.pth
	put data in the data folder
	dive into the jupyter file and change all file paths for your convience
	dive into the swinir_config_base.py, change all file paths for your convience
	run the jupyter file cell by cell
		If you are not in google colab, donot run the first cell
	to generate the result of real-world test set
		dive into the swinir_config_base.py
		change the ...data/test/LQ... part to .../data/test_real...
		rerun the 8th cell in the jupyter file

My codes were run on google colab with NVIDIA Tesla V100 environment. Errors may occur in different environment, please adjust the parameters for your convience.
Do kindly note that results may differ with different random seeds or environments. Please contact me if anything went wrong. Thank you.
