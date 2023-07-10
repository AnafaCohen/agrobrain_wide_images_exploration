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
import json
import random
import glob
import datetime
from tqdm import tqdm
from datetime import datetime

from PIL import Image

import dtlpy as dl
if dl.token_expired():
    dl.login()

env = RuntimeEnv()
categories_dict = cfg['tags']['categories']

DATA_DIR = "data"

def add_box(item, top=100, left=100, bottom=400, right=400):
    builder = item.annotations.builder()
    builder.add(
        annotation_definition=dl.Box(
            top=top, left=left, bottom=bottom, right=right, label="annotation box"
        )
    )
    item.annotations.upload(annotations=builder)



def delete_annotations(item):
    annotations = item.annotations.list()
    for annotation in annotations:
        is_deleted = item.annotations.delete(annotation_id=annotation.id)




def add_boxes(item, items_data):
    for i, box in items_data['boxes_coords'].iterrows():
        # add_box(item, box['top'], box['left'], box['bottom'], box['right'])
        add_box(item, box['left'], box['top'], box['right'], box['bottom'])




def get_grid_coords(image_shape):
    grid_shape = (image_shape[1]//7, image_shape[0]//8)
    top_array = np.arange(0, image_shape[1] - grid_shape[0] - 1, grid_shape[0])
    left_array = np.arange(0, image_shape[0] - 1, grid_shape[1])

    # left_array = np.arange(1, image_shape[0] - grid_shape[1], grid_shape[1])

    mesh_top, mesh_left = np.meshgrid(top_array, left_array)
    top_left_list = np.stack((mesh_top.ravel(), mesh_left.ravel()), axis=1)
    grid_coords = pd.DataFrame(top_left_list, columns = ["top", "left"])
    grid_coords['bottom'] = grid_coords['top'] + grid_shape[0]
    grid_coords['right'] = grid_coords['left'] + grid_shape[1]

    ng_y = grid_coords[grid_coords['top'] > image_shape[1]]
    ng_x = grid_coords[grid_coords['right'] > image_shape[0]]
    # len(ng_y), len(ng_x)

    example_image_id = items_df['name'][0].replace(".jpg", "")
    example_im_path = env.download_image(int(example_image_id))
    example_image = io.imread(example_im_path)

    # fig, ax = plt.subplots()
    # ax.imshow(example_image)
    # ax.scatter(grid_coords['top'], grid_coords['left'], c='red', marker='o')
    # plt.savefig(os.path.join(DATA_DIR, f'grid_image_1.png'))



    return grid_coords



def add_boxes_coords(items_df, image_shape, num_boxes=10, grid_shape=(507, 434)):
    grid_coords = get_grid_coords(image_shape)
    items_df['boxes_coords'] = [grid_coords.sample(n=10).reset_index(drop=True) for i in range(len(items_df))]
    return items_df


def add_boxes_annotation_dataloop(items_df):
    for index, item_data in items_df.iterrows():
        item_id = item_data['id']
        print(f"adding box to image {item_data['name']}, item id: {item_id}")
        item = dataset.items.get(item_id = item_id)
        add_boxes(item, item_data)



def get_image_shape(example_image_id):
    example_im_path = env.download_image(int(example_image_id))
    example_image = io.imread(example_im_path)
    return example_image.shape

if __name__ == "__main__":

    PROJECT_NAME = 'Taranis AI Annotation Projects'
    # TASK_NAME = 'anafa_2023_06_07_wide_full_filtered'
    # DATASET_NAME = "anafa_2023_07_06_wide_full_ttt_filtered"
    # DATASET_NAME = "anafa_2023_06_27_wide_images_first_tagging_task"
    DATASET_NAME = "Anafa_second"

    # TASK_NAME = 'anafa_2023_06_27_wide_images_first_tagging_task'


    project = dl.projects.get(project_name=PROJECT_NAME)
    # task = project.tasks.get(task_name=TASK_NAME)
    # dataset = project.datasets.get(dataset_id=task.dataset.id)
    dataset = project.datasets.get(dataset_name=DATASET_NAME)

    items = dataset.items.list()
    items_df = items.to_df()

    image_shape = get_image_shape(example_image_id = items_df['name'][0].replace(".jpg", ""))[:-1]

    items_df = add_boxes_coords(items_df, image_shape).reset_index(drop=True)

    add_boxes_annotation_dataloop(items_df)




    print("done")