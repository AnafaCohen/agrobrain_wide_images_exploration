import os
import numpy as np
import pandas as pd
import json
os.environ["NAMESPACE"]="research"
os.environ["PROFILE"]="local"
from agrobrain_util.runtime.evironment import RuntimeEnv
from agrobrain_util.infra.app_config import application_config as cfg
from matplotlib.patches import Polygon as PolygonPatch

from agrobrain_image_processing.canopy.canopy import canopy_by_hsv
from agrobrain_canopy.canopy_cover.canopy_cover import CanopyCover

import glob
from skimage import io
from shapely.geometry import Point, Polygon
from skimage.util import crop

import json
import glob
from tqdm import tqdm

import dtlpy as dl
if dl.token_expired():
    dl.login()

env = RuntimeEnv()

def download_datasets_annotations_from_dataloop(dataset_name, task_name="None", version=0):
    dataloop_local_data_dir = os.path.join(DATA_DIR, f"dataloop")
    annotation_local_path = os.path.join(dataloop_local_data_dir, f"annotations_{dataset_name}_task_{task_name}_v{version}")
    dataset = project.datasets.get(dataset_name=dataset_name)
    dataset.download_annotations(local_path=annotation_local_path)
    print(f"Done downloading annotations for the dataset: {dataset_name}.\n Local path: {annotation_local_path}")

def get_jsons_paths_list(dataset_name, task_name="None", version=0, sub_folder=""):
    dataloop_local_data_dir = os.path.join(DATA_DIR, f"dataloop")
    annotation_local_path = os.path.join(dataloop_local_data_dir, f"annotations_{dataset_name}_task_{task_name}_v{version}")
    if sub_folder=="":
        jsons_folder = os.path.join(annotation_local_path, "json")
    else:
        jsons_folder = os.path.join(annotation_local_path, "json", sub_folder)
    jsons_paths_list = glob.glob(os.path.join(jsons_folder, "*.json"))
    return jsons_paths_list



def get_infestation_avg(box_labels):
     infestation_dict = {
         f"Infestation.Up_to_10%": 5,
         f"Infestation.10%-25%": 17.5,
         f"Infestation.25%-50%": 37.5,
         f"Infestation.50%-75%": 65.5,
         f"Infestation.above_75%": 87.5
     }
     labels_numbers = box_labels.replace(infestation_dict)
     return labels_numbers.mean()



def add_poly_box(boxes):
    for i, row in boxes.iterrows():
        x_values = [coord['x'] for coord in boxes['coordinates'][i]]
        y_values = [coord['y'] for coord in boxes['coordinates'][i]]
        minx, maxx = x_values
        miny, maxy = y_values
        bounding_box_coords = [(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny)]
        boxes.at[i, 'box_poly'] = Polygon(bounding_box_coords)
    return boxes


def fit_points_to_boxes(boxes, labels):
    all_boxes_points_lists = []
    all_boxes_points_index_lists = []
    for i, box_row in boxes.iterrows():
        pt_list = []
        pt_index_list = []
        for j, label_row in labels.iterrows():
            x_value = int(label_row['coordinates']['x'])
            y_value = int(label_row['coordinates']['y'])
            shapely_point = Point(x_value, y_value)
            is_inside = box_row['box_poly'].contains(shapely_point)
            if is_inside:
                pt_list.append(shapely_point)
                pt_index_list.append(j)
        all_boxes_points_lists.append(pt_list)
        all_boxes_points_index_lists.append(pt_index_list)
    boxes["fitting_points"] = all_boxes_points_lists
    boxes["fitting_points_indexes"] = all_boxes_points_index_lists
    return boxes




def get_box_label_and_infestation_avg(box_df, labels):
    points_indexes = box_df['fitting_points_indexes']
    box_labels = labels.iloc[points_indexes]

    infestation_rows = box_labels[box_labels['label'].str.contains("Infestation")].index
    no_weeds_rows = box_labels[box_labels['label'].str.contains("No")].index

    if len(infestation_rows) > len(no_weeds_rows):
        infestation_average = get_infestation_avg(labels.iloc[infestation_rows]['label'])
        box_final_label = labels.iloc[infestation_rows]['label'].value_counts().idxmax()
        votes = labels.iloc[infestation_rows]['label'].value_counts().max()
    else:
        box_final_label = labels.iloc[no_weeds_rows]['label'].value_counts().idxmax()
        votes = labels.iloc[no_weeds_rows]['label'].value_counts().max()
        if len(infestation_rows) > 0:
            infestation_average = get_infestation_avg(labels.iloc[infestation_rows]['label'])
        else:
            infestation_average = 0
    return box_final_label, infestation_average, votes

def get_box_canopy_info(box_coordinates, index_canopy_image, hsv_canopy_image):
    x_values = [coord['x'] for coord in box_coordinates]
    y_values = [coord['y'] for coord in box_coordinates]
    minx, maxx = min(x_values), max(x_values)
    miny, maxy = min(y_values), max(y_values)

    cropped_image_hsv = crop(hsv_canopy_image, ((miny, hsv_canopy_image.shape[0] - maxy), (minx, hsv_canopy_image.shape[1] - maxx)))
    cropped_image_index_canopy = crop(index_canopy_image, ((miny, index_canopy_image.shape[0] - maxy), (minx, index_canopy_image.shape[1] - maxx)))

    box_size = cropped_image_hsv.shape[0] * cropped_image_hsv.shape[1]

    box_hsv_canopy_sum = np.count_nonzero(cropped_image_hsv)
    box_hsv_canopy_percent = round(box_hsv_canopy_sum/box_size, 2)
    box_canopy_index_sum = np.count_nonzero(cropped_image_index_canopy)
    box_canopy_index_percent = round(box_canopy_index_sum/box_size, 2)
    box_canopy_avg_hsv_index_sum = round(sum([box_hsv_canopy_sum, box_canopy_index_sum])/2, 2)
    return box_hsv_canopy_sum, box_hsv_canopy_percent, box_canopy_index_sum, box_canopy_index_percent, box_canopy_avg_hsv_index_sum


if __name__ == "__main__":
    DATA_DIR = "data"

    project = dl.projects.get(project_name='Taranis AI Annotation Projects')

    # DOWNLOAD ANNOTATIONS

    # ANNOTATIONS_DATASET_NAME = "anafa_2023_07_17_infestation_21_images"
    ANNOTATIONS_DATASET_NAME = "anafa_tagging_methodology_1000_images_2023_07_24"
    # download_datasets_annotations_from_dataloop(ANNOTATIONS_DATASET_NAME)


    annotations_jsons_paths_list = get_jsons_paths_list(ANNOTATIONS_DATASET_NAME)


    # PRINT RANDOM IMAGE WITH ANNOTATIONS

    dl_annotations_df = pd.DataFrame()
    for json_path in tqdm(annotations_jsons_paths_list, leave=False):
        # json_path = np.random.choice(annotations_jsons_paths_list)
        with open(json_path) as file:
            json_data = json.load(file)

        im_id = int(os.path.basename(json_path).replace(".json",""))
        im_path = env.download_image(int(im_id))
        image = io.imread(im_path)
        print(f"calculating image {im_id} index canopy map...")
        index_canopy_image = CanopyCover.canopy_cover(im_path)[0].astype(np.uint8) * 255
        print(f"calculating image {im_id} hsv canopy map...")
        hsv_canopy_image = canopy_by_hsv(image).astype(np.uint8) * 255


        label_types = [json_data['annotations'][i]['label'] for i in range(len(json_data['annotations']))]
        boxes = pd.DataFrame([json_data['annotations'][i] for i in range(len(json_data['annotations'])) if json_data['annotations'][i]['label'] == 'annotation box']).reset_index(drop=True)
        labels = pd.DataFrame([json_data['annotations'][i] for i in range(len(json_data['annotations'])) if json_data['annotations'][i]['label'] != 'annotation box']).reset_index(drop=True)
        boxes = add_poly_box(boxes)
        boxes = fit_points_to_boxes(boxes, labels)

        boxes["box_final_label"] = None
        boxes["infestation_average"] = None
        for i, box_row in tqdm(boxes.iterrows()):
            box_final_label, infestation_average, votes = get_box_label_and_infestation_avg(box_row, labels)
            boxes.at[i, "box_final_label"] = box_final_label
            boxes.at[i, "infestation_average"] = infestation_average
            boxes.at[i, "votes"] = votes
            box_hsv_canopy_sum, box_hsv_canopy_percent, box_canopy_index_sum, box_canopy_index_percent, box_canopy_avg_hsv_index_sum = get_box_canopy_info(box_row['coordinates'], index_canopy_image, hsv_canopy_image)
            boxes.at[i, "box_hsv_canopy_sum"] = box_hsv_canopy_sum
            boxes.at[i, "box_hsv_canopy_percent"] = box_hsv_canopy_percent
            boxes.at[i, "box_canopy_index_sum"] = box_canopy_index_sum
            boxes.at[i, "box_canopy_index_percent"] = box_canopy_index_percent
            boxes.at[i, "box_canopy_avg_hsv_index_sum"] = box_canopy_avg_hsv_index_sum
        boxes['image_id'] = im_id
        dl_annotations_df = pd.concat([dl_annotations_df, boxes], ignore_index=True)
    dl_annotations_df.to_csv(f"data/infestation_coverage/dataloop_annotations_boxes_dataset_{ANNOTATIONS_DATASET_NAME}.csv")
    print("Done.")
