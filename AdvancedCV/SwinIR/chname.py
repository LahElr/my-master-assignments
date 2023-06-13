import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", default = r"swinir/20230420_123618")
parser.add_argument("--path_prefix", default = r"/content/drive/MyDrive/acv_ass_2/work_dirs")
args = parser.parse_args()
path = args.path

if not path.startswith("/"):
    path = os.path.join(args.path_prefix, path)

path = os.path.join(path,r"vis_data/vis_image")

file_names = os.listdir(path)
for file_name in file_names:
    new_name = file_name.split("_")[0]
    new_name = new_name+".png"
    os.rename(os.path.join(path,file_name), os.path.join(path,new_name))

print("chname.py: task finished.")