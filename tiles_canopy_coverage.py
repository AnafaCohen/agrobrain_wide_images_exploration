import os
import numpy as np
import pandas as pd
import argparse
import skimage.io as skio
from datetime import datetime
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from shapely.geometry import box
import imagesize
from datetime import datetime

from agrobrain_image_processing.canopy.canopy import canopy_by_hsv
from agrobrain_canopy.canopy_cover.canopy_cover import CanopyCover
from agrobrain_util.runtime.evironment import RuntimeEnv
from agrobrain_apis.data_store.data_store import DataStoreAPI


N_BOXES_X = 7
N_BOXES_Y = 8

class Canopy_Coverage_Calculator():
    def __init__(self,
                 env,
                 wide_image_ids_list,
                 wide_image_ids_json_path,
                #  output_csv_path,
                 experiment_name="infestation_by_canopy_coverage_v0"):
        self.env = env
        self.input_image_ids_list = wide_image_ids_list
        self.input_image_ids_json_path = wide_image_ids_json_path
        # self.output_csv_path = output_csv_path
        self.boxes_coords = None
        self.example_image_shape = None
        self.boxes_df = None
        self.image_ids_list_not_filtered = self.read_wide_images_ids_list()
        self.images_data = self.get_images_data_from_eti()
        self.image_ids_list = self.filter_image_ids_list()
        self.set_example_image_attributes()
        self.grid_coords = self.get_grid_coords()
        self.experiment_name = experiment_name
        self.datetime_now = self.get_datetime_now()

    def get_datetime_now(self):
        return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


    def read_wide_images_ids_list(self):
        if len(self.input_image_ids_json_path) > 0:
            with open(self.input_image_ids_json_path, "r") as json_file:
                wide_images_ids = json.load(json_file)
        else:
            wide_images_ids = self.input_image_ids_list
        return wide_images_ids


    def set_example_image_attributes(self):
        example_image_id = self.image_ids_list[0]
        example_im_path = self.env.download_image(int(example_image_id))
        self.example_image_shape = {"width": imagesize.get(example_im_path)[0],
                                    "height": imagesize.get(example_im_path)[1]}


    def check_image_data(self, image_data):
        if (image_data['cameraAngle'] > -95) & (image_data['cameraAngle'] < -85) & (image_data['deleted'] == False):
            return True
        return False

    def get_images_data_from_eti(self):
        # Gets data from eti to  the UNFILTERED list of images
        images_data = self.env.eti_api.get_images_data(self.image_ids_list_not_filtered, type_ids=[2])['images']
        return pd.DataFrame(images_data)


    def filter_image_ids_list(self):
        filtered_image_ids_list = [image_data['imageID'] for i, image_data in self.images_data.iterrows() if self.check_image_data(image_data)]

        return filtered_image_ids_list


    def get_grid_coords(self):
        self.grid_shape = (self.example_image_shape['width']//N_BOXES_X, self.example_image_shape['height']//N_BOXES_Y)

        # self.grid_shape = (self.example_image_shape[1]//7, self.example_image_shape[0]//8)
        top_array = np.arange(0, self.example_image_shape['width'] - self.grid_shape[0] - 1, self.grid_shape[0])
        left_array = np.arange(0, self.example_image_shape['height'] - 1, self.grid_shape[1])

        mesh_top, mesh_left = np.meshgrid(top_array, left_array)
        top_left_list = np.stack((mesh_top.ravel(), mesh_left.ravel()), axis=1)
        grid_coords = pd.DataFrame(top_left_list, columns = ["top", "left"])
        grid_coords['bottom'] = grid_coords['top'] + self.grid_shape[0]
        grid_coords['right'] = grid_coords['left'] + self.grid_shape[1]
        boxes_coords_list = []
        for _, row in grid_coords.iterrows():
            # shapely_box = box(row['left'], row['top'], row['right'], row['bottom'])
            box = [int(row['left']), int(row['top']), int(row['right']), int(row['bottom'])]

            boxes_coords_list.append(box)
        self.boxes_coords = boxes_coords_list


    def get_canopy_cover_maps(self, image, im_path):
        im_id = int(os.path.basename(im_path).split(".")[0])
        print(f"calculating image {im_id} index canopy map...", end="\r", flush=True)
        index_canopy_map = CanopyCover.canopy_cover(im_path)[0].astype(np.uint8) * 255
        print(f"calculating image {im_id} hsv canopy map...", end="\r", flush=True)
        hsv_canopy_map = canopy_by_hsv(image).astype(np.uint8) * 255
        return index_canopy_map, hsv_canopy_map



    def calculate_canopy_percentage(self, index_canopy_map, hsv_canopy_map):
        im_size = index_canopy_map.shape[0] * index_canopy_map.shape[1]
        hsv_canopy_sum = np.count_nonzero(hsv_canopy_map)
        hsv_canopy_percent = int(hsv_canopy_sum/im_size * 100)
        canopy_index_sum = np.count_nonzero(index_canopy_map)
        canopy_index_percent = int(canopy_index_sum/im_size * 100)
        return hsv_canopy_percent, canopy_index_percent


    def get_box_dict(self, idx, box, image_id, order_id, index_canopy_map, hsv_canopy_map, image_hsv_canopy_percent, image_canopy_index_percent):
        left, top, right, bottom = box
        # left, top, right, bottom = box.bounds

        cropped_canopy_index_map = index_canopy_map[int(left):int(right), int(top):int(bottom)]
        cropped_canopy_hsv_map = hsv_canopy_map[int(left):int(right), int(top):int(bottom)]
        box_hsv_canopy_percent, box_canopy_index_percent = self.calculate_canopy_percentage(cropped_canopy_index_map, cropped_canopy_hsv_map)
        avg = int(sum([box_hsv_canopy_percent, box_canopy_index_percent])/2)
        box_dict = {'image_id': image_id,
                    'box_index': idx,
                    'orderID': order_id,
                    'image_hsv_canopy_percent': image_hsv_canopy_percent,
                    'image_canopy_index_percent': image_canopy_index_percent,
                    'box_coords': box,
                    'hsv_canopy_percent': box_hsv_canopy_percent,
                    'index_canopy_percent': box_canopy_index_percent,
                    'canopy_cover_avg': avg}
        return box_dict


    def create_tiles_canopy_cover_csv(self, save_backup_csv=False):
        box_dicts_list = []
        for image_id in tqdm(self.image_ids_list):
            im_path = self.env.download_image(int(image_id))
            image = skio.imread(im_path)
            index_canopy_map, hsv_canopy_map = self.get_canopy_cover_maps(image, im_path)
            image_hsv_canopy_percent, image_canopy_index_percent = self.calculate_canopy_percentage(index_canopy_map, hsv_canopy_map)
            order_id = self.images_data[self.images_data['imageID']==image_id]['orderID'].values[0]
            for idx, box in enumerate(self.boxes_coords, start=1):
                box_dict = self.get_box_dict(idx, box, image_id, order_id, index_canopy_map, hsv_canopy_map, image_hsv_canopy_percent, image_canopy_index_percent)
                box_dicts_list.append(box_dict)
            if save_backup_csv:
                boxes_df = pd.DataFrame(box_dicts_list)
                backup_boxes_path = os.path.dirname(self.output_csv_path) + os.path.basename(self.output_csv_path.split(".")[0]) + f"_{image_id}." + os.path.basename(self.output_csv_path.split(".")[1])
                boxes_df.to_csv(backup_boxes_path, index=False)
            self.boxes_df = pd.DataFrame(box_dicts_list)
            self.store_boxes_predictions(image_id)
        # self.boxes_df = pd.DataFrame(box_dicts_list)
        # self.boxes_df.to_csv(self.output_csv_path, index=False)

    def store_boxes_predictions(self, image_id):
        data_api = DataStoreAPI()
        for i, box in self.boxes_df.iterrows():
            data_api.store("infestation_wide_images", payload=box.to_dict(), image_id=image_id, experiment=self.experiment_name, metadata={"timestamp": self.datetime_now})
            # data_api.store("late_corn_weed_species", file_payload_path=im_file_name, image_id=image_id, experiment=experiment_name, metadata={"type": "image", "timestamp": int(time.time())})

def get_run_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wide_image_ids_list", type=json.loads)
    parser.add_argument("--wide_image_ids_json_path", type=str)
    # parser.add_argument("--output_csv_path", type=str)
    args = parser.parse_args()
    return args



if __name__ == "__main__":

    env = RuntimeEnv()

    args = get_run_arguments()

    canopy_coverage_calculator = Canopy_Coverage_Calculator(env,
                                                            wide_image_ids_list = args.wide_image_ids_list,
                                                            wide_image_ids_json_path = args.wide_image_ids_json_path)
                                                            # output_csv_path = args.output_csv_path)
    canopy_coverage_calculator.create_tiles_canopy_cover_csv()

