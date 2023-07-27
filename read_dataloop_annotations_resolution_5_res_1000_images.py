import os
import numpy as np
import pandas as pd
import json
os.environ["NAMESPACE"]="research"
os.environ["PROFILE"]="local"
from agrobrain_util.runtime.evironment import RuntimeEnv
from agrobrain_util.infra.app_config import application_config as cfg
from matplotlib.patches import Polygon as PolygonPatch

import shutil
import glob
from skimage import io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from functools import reduce
from shapely.geometry import Point, Polygon
# import shapely.geometry
# from shapely.ops import  intersection

import json
import random
import glob
import datetime
from tqdm import tqdm
from datetime import datetime
import pprint

from skimage.transform import resize
from skimage.filters import gaussian
import skimage.measure

from PIL import Image

import dtlpy as dl
if dl.token_expired():
    dl.login()



def download_datasets_annotations_from_dataloop(dataset_name, task_name, version):
    dataloop_local_data_dir = os.path.join(DATA_DIR, f"dataloop")
    annotation_local_path = os.path.join(dataloop_local_data_dir, f"annotations_{dataset_name}_task_{task_name}_v{version}")
    dataset = project.datasets.get(dataset_name=dataset_name)
    dataset.download_annotations(local_path=annotation_local_path)
    print(f"Done downloading annotations for the dataset: {dataset_name}.\n Local path: {annotation_local_path}")


def get_jsons_paths_list(dataset_name, task_name, version, sub_folder=""):
    dataloop_local_data_dir = os.path.join(DATA_DIR, f"dataloop")
    annotation_local_path = os.path.join(dataloop_local_data_dir, f"annotations_{dataset_name}_task_{task_name}_v{version}")
    if sub_folder=="":
        jsons_folder = os.path.join(annotation_local_path, "json")
    else:
        jsons_folder = os.path.join(annotation_local_path, "json", sub_folder)
    jsons_paths_list = glob.glob(os.path.join(jsons_folder, "*.json"))
    return jsons_paths_list


def calculate_iou(polygon1, polygon2):
    intersection_area = polygon1.intersection(polygon2).area
    union_area = polygon1.union(polygon2).area
    iou = intersection_area / union_area
    return float("{:.2f}".format(iou))


def calculate_all_ious(polygon_list):
    num_polygons = len(polygon_list)
    iou_matrix = [[0.0 for _ in range(num_polygons)] for _ in range(num_polygons)]

    for i in range(num_polygons):
        for j in range(i+1, num_polygons):
            iou = calculate_iou(polygon_list[i], polygon_list[j])
            iou_matrix[i][j] = iou
            iou_matrix[j][i] = iou

    return iou_matrix


def get_polygons_list_from_json_data(polygons_json_data):
    shapely_polygons = []
    annotators = []
    for i in range(len(polygons_json_data['annotations'])):
        x_values = [coord['x'] for coord in polygons_json_data['annotations'][i]['coordinates'][0]]
        y_values = [coord['y'] for coord in polygons_json_data['annotations'][i]['coordinates'][0]]
        shapely_polygon = Polygon(list(zip(x_values, y_values)))
        if shapely_polygon.is_valid:
            shapely_polygons.append(shapely_polygon)
            annotators.append(polygons_json_data['annotations'][i]['updatedBy'])

    return shapely_polygons, annotators


def almost_fully_contained(poly1, poly2):
    smaller_poly_area = min(poly1.area, poly2.area)
    if poly1.intersection(poly2).area >= 0.8 * smaller_poly_area:
        return True
    else:
        return False


def polygons_contains_each_other(polygon1, polygon2):
    if polygon1.contains(polygon2) or polygon2.contains(polygon1) or almost_fully_contained(polygon1, polygon2):
        return True
    else:
        return False




def find_overlapping_polygons_indexes(polygons_list):
    polygons_df = pd.DataFrame()
    num_polygons = len(polygons_list)
    iou_matrix = calculate_all_ious(polygons_list)
    overlapping_polygons_dict = {}
    indexes_to_remove = []
    for i in range(num_polygons):
        overlapping_polygons_dict[i] = [i]
        current_polygon = polygons_list[i]
        for j in range(i+1, num_polygons):
            if iou_matrix[i][j] > 0.5 or polygons_contains_each_other(current_polygon, polygons_list[j]):
                overlapping_polygons_dict[i].append(j)
                indexes_to_remove.append(j)

    overlapping_polygons = [overlapping_polygons_dict[key] for key in overlapping_polygons_dict.keys() if key not in indexes_to_remove]
    return overlapping_polygons


def reduce_contained_polygons(overlapping_polygons_list):
    polygons_to_remove = set()
    for i, poly1 in enumerate(overlapping_polygons_list):
        # Check if the polygon has already been marked for removal
        if i in polygons_to_remove:
            continue
        for j, poly2 in enumerate(overlapping_polygons_list):
            # Skip comparing a polygon to itself
            if i == j:
                continue
            # Check if the area of poly1 is smaller than poly2 and if poly1 is contained within poly2
            if poly1.within(poly2):
                polygons_to_remove.add(i)
    # Create a new list without the polygons to be removed
    reduced_poly_list = [overlapping_polygons_list[i] for i in range(len(overlapping_polygons_list)) if i not in polygons_to_remove]
    return reduced_poly_list


def intersection_with_condition(x, y):
    intersection_result = x.intersection(y)
    if not intersection_result.is_empty:
        return intersection_result
    else:
        return x
        # return x, y



def ensure_tuple(item):
    if isinstance(item, tuple):
        return item
    else:
        return (item,)


def get_polygons_intersection(overlapping_polygons_list):
    # overlapping_polygons_list = reduce_contained_polygons(overlapping_polygons_list)
    # intersection_polygon = reduce(lambda x, y: x.intersection(y), overlapping_polygons_list)
    intersection_polygon = reduce(intersection_with_condition, overlapping_polygons_list)
    intersection_polygon = ensure_tuple(intersection_polygon)
    intersection_polygon = tuple([list(p.exterior.coords) for p in intersection_polygon])
    return intersection_polygon


def get_polygons_union(overlapping_polygons_list):
    union_polygon = reduce(lambda x, y: x.union(y), overlapping_polygons_list)
    return union_polygon


def get_intersection_area(overlapping_polygons_list):
    intersection_result = get_polygons_intersection(overlapping_polygons_list)
    intersection_result = [Polygon(coords) for coords in intersection_result]
    if len(intersection_result)>1:
        area = sum([p.area for p in intersection_result])
    else:
        area = intersection_result[0].area
    return area


def get_polygons_iou(overlapping_polygons_list):
    intersection_area = get_intersection_area(overlapping_polygons_list)
    iou = intersection_area/get_polygons_union(overlapping_polygons_list).area
    iou = float(f"{iou:.2f}")
    return iou


def get_polygons_box(polygon, dilation = 0):
    minx, miny, maxx, maxy = polygon.bounds
    bounding_box_coords = [(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny)]
    bounding_box = Polygon(bounding_box_coords)
    return bounding_box


def calc_polygons_avarage_area(overlapping_polygons):
    areas = [polygon.area for polygon in overlapping_polygons]
    average_area = sum(areas) / len(areas)
    return average_area


def calc_mm_from_pixel_area(area_in_pixels, image_data):
    image_resolution = image_data['zoom_resolution'].values[0]
    l_mm = int(np.sqrt(area_in_pixels) * image_resolution)
    return l_mm


def calc_area_mm2_from_pixel_area(area_in_pixels, image_data):
    image_resolution = image_data['zoom_resolution'].values[0]
    a_mm2 = int(area_in_pixels * image_resolution * image_resolution)
    return a_mm2


def create_polygons_df(polygons_list, image_id, annotators, image_data):
    polygons_df = pd.DataFrame()
    polygons_df['original_polygons'] = None
    polygons_df['poly_intersection'] = None
    polygons_df['overlapping_polygons_indexes'] = find_overlapping_polygons_indexes(polygons_list)
    for index, row in polygons_df.iterrows():
        overlapping_polygons_indexes = polygons_df['overlapping_polygons_indexes'][index]
        overlapping_polygons = list(map(lambda i: polygons_list[i], overlapping_polygons_indexes))
        polygons_df.at[index, 'original_polygons'] = (list(map(lambda i: polygons_list[i], overlapping_polygons_indexes)),)
        polygons_df.at[index, 'original_polygons_avg_area_pixel'] = int(calc_polygons_avarage_area(overlapping_polygons))
        polygons_df.at[index, 'original_polygons_avg_length_mm'] = calc_mm_from_pixel_area(polygons_df.at[index, 'original_polygons_avg_area_pixel'], image_data)
        polygons_df.at[index, 'original_polygons_avg_area_mm2'] = calc_area_mm2_from_pixel_area(polygons_df.at[index, 'original_polygons_avg_area_pixel'], image_data)
        polygons_df.at[index, 'num_of_original_polygons'] = len(polygons_df.at[index, 'original_polygons'][0])
        polygons_df.at[index, 'poly_intersection'] = json.dumps(get_polygons_intersection(overlapping_polygons))
        polygons_df.at[index, 'poly_union'] = get_polygons_union(overlapping_polygons)
        polygons_df.at[index, 'poly_union_area_pixel'] = int(polygons_df.at[index, 'poly_union'].area)
        polygons_df.at[index, 'poly_union_length_mm'] = calc_mm_from_pixel_area(polygons_df.at[index, 'poly_union_area_pixel'], image_data)
        polygons_df.at[index, 'poly_union_area_mm2'] = calc_area_mm2_from_pixel_area(polygons_df.at[index, 'poly_union_area_pixel'], image_data)
        polygons_df.at[index, 'poly_iou'] = get_polygons_iou(overlapping_polygons)
        polygons_df.at[index, 'poly_box'] = get_polygons_box(polygons_df.at[index, 'poly_union'])
        polygons_df.at[index, 'poly_box_area_pixel'] = int(polygons_df.at[index, 'poly_box'].area)
        polygons_df.at[index, 'poly_box_length_mm'] = calc_mm_from_pixel_area(polygons_df.at[index, 'poly_box_area_pixel'], image_data)
        polygons_df.at[index, 'poly_box_area_mm2'] = calc_area_mm2_from_pixel_area(polygons_df.at[index, 'poly_box_area_pixel'], image_data)
        polygons_df.at[index, 'annotators'] = str(list(map(lambda i: annotators[i], polygons_df['overlapping_polygons_indexes'][index])))
    polygons_df['image_id'] = image_id
    return polygons_df



def save_image_with_tags(polygons_list, image_id, annotators, points_list, image_data):
    im_path = env.download_image(int(image_id))
    image = io.imread(im_path)
    polygons_df = create_polygons_df(polygons_list, image_id, annotators, image_data)

    line_styles = ['solid', 'dashed', 'dotted', 'dashdot']

    for index, row in polygons_df.iterrows():
        plt.figure()
        plt.imshow(image)

        poly_intersection = [Polygon(coords) for coords in json.loads(row['poly_intersection'])]
        for intersection_poly in poly_intersection:
            intersection_patch = PolygonPatch(np.array(intersection_poly.exterior.coords), facecolor='none', edgecolor='cornflowerblue', linewidth=2)
            plt.gca().add_patch(intersection_patch)
        union_patch = PolygonPatch(np.array(row['poly_union'].exterior.coords), facecolor='none', edgecolor='orange', linewidth=2)
        box_patch = PolygonPatch(np.array(row['poly_box'].exterior.coords), facecolor='none', edgecolor='springgreen', linewidth=2)

        minx, miny, maxx, maxy = row['poly_union'].bounds

        plt.gca().add_patch(union_patch)
        plt.gca().add_patch(box_patch)

        for i, poly in enumerate(row['original_polygons'][0]):
            linestyle = line_styles[i % len(line_styles)]
            poly_patch = PolygonPatch(np.array(poly.exterior.coords), facecolor='none', edgecolor='black', linewidth=1, linestyle=linestyle)
            plt.gca().add_patch(poly_patch)

        for point in points_list:
            plt.plot(x_value, y_value, "mo")


        num_of_original_polygons = len(row['original_polygons'][0])
        text_x, text_y = minx, miny-50
        plt.text(text_x, text_y, "intersection", fontsize=12, color='cornflowerblue', ha='center', va='center')
        plt.text(text_x, text_y+20, "union", fontsize=12, color='orange', ha='center', va='center')
        plt.title(f"Image {image_id}, IOU: {row['poly_iou']}\nnum_orig_polygons_{num_of_original_polygons}")

        plt.xlim(minx-100, maxx+100)
        plt.ylim(maxy+100, miny-100)

        image_with_polygons_path = f'images/output_image_with_polygons_{image_id}_{index}_with_points.jpg'
        plt.savefig(image_with_polygons_path)
        plt.close()
        print(f"Saved image {image_with_polygons_path}.")


def get_original_image_shape():
    example_image_id = 9638024
    example_im_path = env.download_image(int(example_image_id))
    example_image = io.imread(example_im_path)
    original_image_width = example_image.shape[1]
    original_image_height = example_image.shape[0]
    return original_image_width,original_image_height


def get_resolutions_list_from_points_jsons(points_jsons_paths_list):
    resolution_list = np.unique([float(os.path.basename(json).replace(".json", "").split("_")[-1])for json in points_jsons_paths_list])
    return resolution_list



if __name__ == "__main__":

    env = RuntimeEnv()

    DATA_DIR = "data"

    project = dl.projects.get(project_name='Taranis AI Annotation Projects')

    # DOWNLOAD ANNOTATIONS

    # POINT TAGS

    # POINTS_DATASET_NAME = "anafa_2023_07_12_resolution_lim_5_res_20_images"
    # POINTS_TASK_NAME = 'anafa_2023_07_12_resolution_lim_5_res_20_images'
    # POINTS_VERSION = 0

    POINTS_DATASET_NAME = "anafa_2023_07_11_resolution_limitation_5_res"
    POINTS_TASK_NAME = 'anafa_2023_07_11_resolution_limitation_5_res'
    POINTS_VERSION = 0

    # download_datasets_annotations_from_dataloop(POINTS_DATASET_NAME, POINTS_TASK_NAME, POINTS_VERSION)
    points_jsons_paths_list = get_jsons_paths_list(POINTS_DATASET_NAME, POINTS_TASK_NAME, POINTS_VERSION, sub_folder="resized_images")
    points_task = project.tasks.get(task_name=POINTS_TASK_NAME)


    # POLYGONS TAGS

    POLYGONS_DATASET_NAME = "anafa_2023_06_23_resolution_lim_dataset_1"
    POLYGONS_TASK_NAME = 'anafa_2023_06_23_resolution_lim_first_task_01'
    POLYGONS_VERSION = 0

    # download_datasets_annotations_from_dataloop(POLYGONS_DATASET_NAME, POLYGONS_TASK_NAME, POLYGONS_VERSION)
    polygons_jsons_paths_list = get_jsons_paths_list(POLYGONS_DATASET_NAME, POLYGONS_TASK_NAME, POLYGONS_VERSION)
    polygons_task = project.tasks.get(task_name=POLYGONS_TASK_NAME)


    points_task_image_ids_list = np.unique([int(os.path.basename(p).split("_")[0]) for p in points_jsons_paths_list])

    problematic_images_list = [7851797, 8908805, 9104069, 9104165, 6238883, 6239082]
    points_task_image_ids_list = points_task_image_ids_list[~np.isin(points_task_image_ids_list, problematic_images_list)]


    filtered_polygons_jsons_paths_list = [elem for elem in polygons_jsons_paths_list if any(str(id_) in os.path.basename(elem) for id_ in points_task_image_ids_list)]

    print(f"len points_jsons_paths_list:{len(points_jsons_paths_list)}, len polygons_jsons_paths_list: {len(polygons_jsons_paths_list)}, len filtered_polygons_jsons_paths_list: {len(filtered_polygons_jsons_paths_list)}")

    # points_task_image_ids_list = [6580458, 9445268]
    # points_task_image_ids_list = [9638024]

    original_image_width, original_image_height = get_original_image_shape()
    resolutions_list = get_resolutions_list_from_points_jsons(points_jsons_paths_list)

    images_data = pd.read_csv(os.path.join(DATA_DIR, 'resolution_test', 'resolution_test_images_dataframe_1000_images_full_data_1.csv'))

    polygons_df = pd.DataFrame()
    points_df =pd.DataFrame()
    tags_df =pd.DataFrame()


    for image_id in tqdm(points_task_image_ids_list):
        print(image_id)

        # POLYGONS
        polygons_tags_json = [path for path in polygons_jsons_paths_list if str(image_id) in path][0]
        with open(polygons_tags_json) as file:
            polygons_json_data = json.load(file)

        image_polygons_list, annotators = get_polygons_list_from_json_data(polygons_json_data)
        image_data = images_data[images_data['imageID']==image_id]
        image_polygons_df = create_polygons_df(image_polygons_list, image_id, annotators, image_data)

        if len(image_polygons_df) == 0:
            continue


        for res in resolutions_list:
            image_polygons_df[f'res_{res}'] = 0


        # POINTS
        points_df = pd.DataFrame()
        points_tags_jsons_list = [path for path in points_jsons_paths_list if str(image_id) in path]
        for points_tags_json in points_tags_jsons_list:
            with open(points_tags_json) as file:
                points_json_data = json.load(file)

            resolution = float(os.path.basename(points_tags_json).replace(".json", "").split("_")[-1])
            flight_height = int(os.path.basename(points_tags_json).replace(".json", "").split("_")[3])

            # image_polygons_df[f'res_{resolution}'] = 0

            width = points_json_data['metadata']['system']['width']
            height = points_json_data['metadata']['system']['height']
            width_factor = original_image_width / width
            height_factor = original_image_height / height

            points_list = []
            for i in range(len(points_json_data['annotations'])):
                x_value = int(points_json_data['annotations'][i]['coordinates']['x']) * width_factor
                y_value = int(points_json_data['annotations'][i]['coordinates']['y']) * height_factor

                shapely_point = Point(x_value, y_value)
                points_list.append(shapely_point)

                is_included = np.any([poly.contains(shapely_point) for poly in image_polygons_df['poly_union']])
                matching_polygon = [poly for poly in image_polygons_df['poly_union'] if poly.contains(shapely_point)]
                for index, row in image_polygons_df.iterrows():
                    if row['poly_union'].contains(shapely_point):
                        image_polygons_df.at[index, f'res_{resolution}'] +=1
                        # row[f'res_{resolution}'] +=1
                # print("here")

                polygons_df = pd.concat([polygons_df, image_polygons_df], ignore_index=True)



            # save_image_with_tags(polygons_list, image_id, annotators, points_list, image_data)


    polygons_df.to_csv(os.path.join(DATA_DIR, "resolution_limitation_tags_dataframe_3.csv"))
    print("Done.")












# WORKS FOR SURE!

    # for image_id in points_task_image_ids_list:
    #     print(image_id)
    #     points_tags_json = [path for path in points_jsons_paths_list if str(image_id) in path][0]
    #     polygons_tags_json = [path for path in polygons_jsons_paths_list if str(image_id) in path][0]

    #     with open(points_tags_json) as file:
    #         points_json_data = json.load(file)

    #     with open(polygons_tags_json) as file:
    #         polygons_json_data = json.load(file)

    #     # POLYGONS
    #     image_polygons_list, annotators = get_polygons_list_from_json_data(polygons_json_data)
    #     image_polygons_df = create_polygons_df(image_polygons_list, image_id, annotators)
    #     polygons_df = pd.concat([polygons_df, image_polygons_df], ignore_index=True)

    #     # POINTS
    #     width = points_json_data['metadata']['system']['width']
    #     height = points_json_data['metadata']['system']['height']
    #     width_factor = original_image_width / width
    #     height_factor = original_image_height / height

    #     points_list = []
    #     for i in range(len(points_json_data['annotations'])):
    #         x_value = int(points_json_data['annotations'][i]['coordinates']['x']) * width_factor
    #         y_value = int(points_json_data['annotations'][i]['coordinates']['y']) * height_factor

    #         shapely_point = Point(x_value, y_value)
    #         points_list.append(shapely_point)

    #         is_inside = [poly.contains(shapely_point) for poly in polygons_df['poly_union']]
    #         is_included = np.any(is_inside)

    #     # save_image_with_tags(polygons_list, image_id, annotators, points_list, image_data)







