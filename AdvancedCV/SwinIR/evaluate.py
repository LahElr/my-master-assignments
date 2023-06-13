import os
import sys

import numpy as np
from PIL import Image
#import utils
#import torch

def read_image(path):
    img = Image.open(path)
    img = np.array(img) / 255.
    
    return img * 255


def psnr(img1, img2):
    #assert img1.shape == img2.shape
    # bs, h, w, 3
    #mean_dim = tuple(range(len(img1.shape)-3),len(img1.shape))
    #mse_value = torch.mean(torch.square(img1-img2),dim=mean_dim, keep_dim=False)
    #ret = 20. * torch.log10(255./torch.sqrt(mse_value))
    #if len(ret.shape)>0:
    #    ret = torch.split(ret,1,dim=0)
    #return ret
    mse_value = np.mean((img1 - img2)**2)
    return 20. * np.log10(255. / np.sqrt(mse_value))

r'''
def evaluate(model, dataloader):
    utils.logger.info("Start evaluation...")
    psnr_list = []
    model.eval()
    with torch.no_grad():
        for lr_fig_batch, hr_fig_batch in dataloader:
            output_fig_batch = model(lr_fig_batch)
            score = psnr(output_fig_batch,hr_fig_batch)
            psnr_list.append(score)
    overall_score = np.mean(psnr_list)
    utils.logger.info("Evaluation finished!")
    utils.logger.info(f"PSNR socre: {overall_score}.")
    utils.logger.info("")
'''
input_dir = sys.argv[1]
output_dir = sys.argv[2]

submit_dir = os.path.join(input_dir, "res")
truth_dir = os.path.join(input_dir, "ref")

if not os.path.isdir(submit_dir):
    print("%s doesn't exist" % submit_dir)

if os.path.isdir(submit_dir) and os.path.isdir(truth_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    submit_dir_list = os.listdir(submit_dir)
    if len(submit_dir_list) == 1:
        submit_dir = os.path.join(submit_dir, "%s" % submit_dir_list[0])
        assert os.path.isdir(submit_dir)

    psnr_list = []
    for idx in range(400):
        pred_img = read_image(os.path.join(submit_dir, "%05d.png" % idx))
        gt_img = read_image(os.path.join(truth_dir, "%05d.png" % idx))
        psnr_list.append(psnr(pred_img, gt_img))

    mean_psnr = np.mean(psnr_list)

    # Create the evaluation score path
    output_filename = os.path.join(output_dir, 'scores.txt')

    with open(output_filename, 'w') as f3:
        f3.write('PSNR: {}'.format(mean_psnr))
