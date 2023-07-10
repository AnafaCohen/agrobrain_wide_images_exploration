import os
import numpy as np
import pandas as pd
import json
os.environ["NAMESPACE"]="research"
os.environ["PROFILE"]="local"
from agrobrain_util.runtime.evironment import RuntimeEnv
from agrobrain_util.infra.app_config import application_config as cfg

import shutil
import glob
from skimage import io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Polygon

from multiprocessing import Pool
import json
import random
import glob
import datetime
from tqdm import tqdm
from datetime import datetime

from skimage.transform import resize
from skimage.filters import gaussian
import skimage.measure

from PIL import Image

import dtlpy as dl
if dl.token_expired():
    dl.login()

env = RuntimeEnv()

DATA_DIR = "data"

DATASET_NAME = "anafa_2023_06_23_resolution_lim_dataset_1"
VERSION = 0

F = 24 # mm
CCD_X = 35 # mm
CCD_Y = 26.3 # mm
IMAGE_X_PX = 5184 # px (zoom)
IMAGE_Y_PX = 3888 # px (zoom)


def get_wide_image_resolution(height, focal_length=F):
    fp_x_m = (height / focal_length) * CCD_X # m
    fp_y_m = (height / focal_length) * CCD_Y # m
    # total_ground_area = fp_x_m * fp_y_m # m^2
    resolution = 0.5*(fp_x_m/image_x_px_wide + fp_y_m/image_y_px_wide)*1000 # mm/px
    return resolution

def get_image_resolution(height, focal_length):
    fp_x_m = (height / focal_length) * CCD_X # m
    fp_y_m = (height / focal_length) * CCD_Y # m
    # total_ground_area = fp_x_m * fp_y_m # m^2
    resolution = 0.5*(fp_x_m/IMAGE_X_PX + fp_y_m/IMAGE_Y_PX)*1000 # mm/px
    return resolution, fp_x_m, fp_y_m


def get_new_x_y_px(resolution, new_resolution):
    x_px = int(resolution/new_resolution * IMAGE_X_PX)
    # print(f"resolution deviation: {resolution/new_resolution}")
    y_px = int(resolution/new_resolution * IMAGE_Y_PX)
    return x_px, y_px

def create_resolution_dict():
    #  CREATE RESOLUTION DICTIONARY FROM WIDE IMAGES HEIGHTS
    resolution_dict = {}
    step_size = 5
    heights_options = np.arange(20, 45, step_size)
    for h in heights_options:
        resolution_dict[h] = get_wide_image_resolution(h, F)
    return resolution_dict


def process_image(im_json_path):
    image_id = int(os.path.basename(im_json_path).replace(".json", ""))
    image_data = images_data[images_data['imageID'] == image_id]
    im_path = env.download_image(image_id)
    original_image = io.imread(im_path)


    focal_length = image_data['focalLength'].values[0]
    flight_height = image_data['heightAboveGround'].values[0]
    zoom_image_resolution, fp_x_m, fp_y_m = get_image_resolution(flight_height, focal_length)
    wide_im_id = int(images_data['wideImageID'][0])

    if wide_im_id not in wide_images_data['imageID'].values:
        return

    saved_original_image_path = os.path.join(resized_images_folder, f"{image_id}_original.jpg")
    io.imsave(saved_original_image_path, original_image)

    resolution_dict = create_resolution_dict()

    for i, h in enumerate(resolution_dict.keys()):
        resized_image_shape = get_new_x_y_px(zoom_image_resolution, resolution_dict[h])
        # print(resolution_dict[h], resized_image_shape)
        processed_image_name = f"{image_id}_{i}_h_{h}_resolution1_{resolution_dict[h]:.2f}.jpg"
        processed_image_path = os.path.join(resized_images_folder, processed_image_name)

        processed_image = resize(original_image, (resized_image_shape[1], resized_image_shape[0], 3), anti_aliasing=True)
        # processed_image = resize(processed_image, (original_image.shape[0], original_image.shape[1]), mode='constant', anti_aliasing=False)
        processed_image = (processed_image * 255).astype(np.uint8)

        io.imsave(processed_image_path, processed_image)
        processed_image = None


if __name__ == "__main__":

    dataloop_local_data_dir = os.path.join(DATA_DIR, f"dataloop")
    annotation_local_path = os.path.join(dataloop_local_data_dir, f"annotations_{DATASET_NAME}_v{VERSION}")

    jsons_folder = os.path.join(annotation_local_path, "json")
    jsons_paths_list = glob.glob(os.path.join(jsons_folder, "*.json"))



    images_data = pd.read_csv(os.path.join(DATA_DIR, 'resolution_test', 'resolution_test_images_dataframe_1000_images_full_data_1.csv'))
    images_data = images_data[~images_data['wideImageID'].isna()].reset_index(drop=True)

    wide_images_data = pd.read_csv((os.path.join(DATA_DIR, "resolution_test", "wide_images_data.csv")))

    jsons_paths_list = [json_path for json_path in jsons_paths_list if int(os.path.basename(json_path).replace(".json", "")) in images_data['imageID'].values]


    random_image_id = int(os.path.basename(np.random.choice(jsons_paths_list)).replace(".json", ""))
    example_image_id = random_image_id


    resized_images_folder = os.path.join("images", "resized_images")

    example_wide_image_id = env.eti_api.get_matching_wide_images([example_image_id])[0]
    wide_im_path = env.download_image(int(example_wide_image_id))
    wide_image = io.imread(wide_im_path)
    wide_image_shape = wide_image.shape
    wide_image_area_pixels = wide_image_shape[0] * wide_image_shape[1]

    wide_images_data['wide_resolution'] = np.sqrt(wide_images_data['footprintArea'] / wide_image_area_pixels) * 1000

    image_x_px_wide = wide_image.shape[1]
    image_y_px_wide = wide_image.shape[0]





    # RESIZE ALL IAMGES FROM JSON_PATH_LIST

    resized_images_folder = os.path.join("images", "resized_images")
    os.makedirs(resized_images_folder, exist_ok=True)


    with Pool() as pool:
        # Apply the process_image function to each item in jsons_paths_list
        results = list(tqdm(pool.imap(process_image, jsons_paths_list), total=len(jsons_paths_list)))


    # for im_json_path in tqdm(jsons_paths_list):
    #     image_id = int(os.path.basename(im_json_path).replace(".json", ""))
    #     image_data = images_data[images_data['imageID'] == image_id]
    #     im_path = env.download_image(image_id)
    #     original_image = io.imread(im_path)


    #     focal_length = image_data['focalLength'].values[0]
    #     flight_height = image_data['heightAboveGround'].values[0]
    #     zoom_image_resolution, fp_x_m, fp_y_m = get_image_resolution(flight_height, focal_length)
    #     wide_im_id = int(images_data['wideImageID'][0])

    #     if wide_im_id not in wide_images_data['imageID'].values:
    #         # print("1")
    #         continue

    #     wide_image_resolution = wide_images_data[wide_images_data['imageID']==wide_im_id]['wide_resolution'].values[0]

    #     saved_original_image_path = os.path.join(resized_images_folder, f"{image_id}_original.jpg")
    #     io.imsave(saved_original_image_path, original_image)

    #     for i, h in enumerate(resolution_dict.keys()):
    #         resized_image_shape = get_new_x_y_px(zoom_image_resolution, resolution_dict[h])
    #         # print(resolution_dict[h], resized_image_shape)
    #         processed_image_name = f"{image_id}_{i}_h_{h}_resolution1_{resolution_dict[h]:.2f}.jpg"
    #         processed_image_path = os.path.join(resized_images_folder, processed_image_name)

    #         processed_image = resize(original_image, (resized_image_shape[1], resized_image_shape[0], 3), anti_aliasing=True)
    #         # processed_image = resize(processed_image, (original_image.shape[0], original_image.shape[1]), mode='constant', anti_aliasing=False)
    #         # processed_image = (image_data * 255).astype(np.uint8)
    #         io.imsave(processed_image_path, processed_image)
    #         processed_image = None

