import utils
import he
import numpy
import cv2
import os

def process_one_image(image_path,output_root_path):
    if not os.path.exists(output_root_path):
        os.makedirs(output_root_path)

    image_name = '.'.join(os.path.split(image_path)[-1].split('.')[:-1])
    this_image_result_path = os.path.join(output_root_path,image_name)
    if not os.path.exists(this_image_result_path):
        os.makedirs(this_image_result_path)

    image = utils.read_img(image_path)
    image_gray = utils.channel_map(image, "gray")
    utils.save_img(image_gray,os.path.join(this_image_result_path,"gray.png"))
    image_he_cv = he.open_cv_hsitogram_mapper(image_gray)
    utils.save_img(image_he_cv,os.path.join(this_image_result_path,"cv.png"))
    utils.draw_histogram(image_gray,"gray",32,title="gray",fname=os.path.join(this_image_result_path,"histo_gray.png"),show=False)
    utils.draw_histogram(image_he_cv,"gray",32,title="cv",fname=os.path.join(this_image_result_path,"histo_cv.png"),show=False)
    heer_configs = [
        {"title":"no_clip","config":{"image":image,"channel":"gray","bin_count":32,"limit":255,"pixel_in_bin_function":"zero","clip":None,"clip_way":"distribution"}},
        {"title":"clip_0.03_dist","config":{"image":image,"channel":"gray","bin_count":32,"limit":255,"pixel_in_bin_function":"zero","clip":0.03,"clip_way":"distribution"}},
        {"title":"clip_0.03_zoom","config":{"image":image,"channel":"gray","bin_count":32,"limit":255,"pixel_in_bin_function":"zero","clip":0.03,"clip_way":"zoom"}},
        # {"title":"clip_0.2_dist","config":{"image":image,"channel":"gray","bin_count":32,"limit":255,"pixel_in_bin_function":"zero","clip":0.2,"clip_way":"distribution"}},
        # {"title":"clip_0.2_zoom","config":{"image":image,"channel":"gray","bin_count":32,"limit":255,"pixel_in_bin_function":"zero","clip":0.2,"clip_way":"zoom"}},
    ]
    for heer_config in heer_configs:
        process_one_heer(heer_config['config'],os.path.join(this_image_result_path,heer_config['title']))
    
def process_one_heer(config,heer_root_path):
    if not os.path.exists(heer_root_path):
        os.makedirs(heer_root_path)

    heer = he.simple_histogram_equlization_mapper(**config)
    pib_funcs = ["reverse","original","zero","full_equalization"]
    for pib_func in pib_funcs:
        image_processed = heer(pixel_in_bin_function = pib_func)
        utils.save_img(image_processed,os.path.join(heer_root_path,f"{pib_func}.png"))
        utils.draw_histogram(image_processed,"gray",32,title=f"{pib_func}",fname=os.path.join(heer_root_path,f"histo_{pib_func}.png"),show=False)

image_source_path = r"../samples"
sample_images = os.listdir(image_source_path)
for sample_image in sample_images:
    process_one_image(os.path.join(image_source_path,sample_image),r"../results")




