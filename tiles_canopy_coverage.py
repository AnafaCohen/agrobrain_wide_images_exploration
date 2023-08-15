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

from agrobrain_image_processing.canopy.canopy import canopy_by_hsv
from agrobrain_canopy.canopy_cover.canopy_cover import CanopyCover
from agrobrain_util.runtime.evironment import RuntimeEnv





class Canopy_Coverage_Calculator():
    def __init__(self):
        self.get_run_arguments()
        self.boxes_coords = None
        self.example_image_id = None
        self.example_im_path = None
        self.example_image = None
        self.example_image_shape = None
        self.boxes_df = None
        self.wide_images_ids = self.read_wide_images_ids()
        self.image_ids_list = self.filter_image_ids_list()
        self.set_example_image_attributes()
        self.grid_coords = self.get_grid_coords()




    def get_run_arguments(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--wide_image_ids_list", type=json.loads)
        parser.add_argument("--wide_image_ids_json_path", type=str)
        parser.add_argument("--output_csv_path", type=str)
        self.args = parser.parse_args()

    def read_wide_images_ids(self):
        if len(self.args.wide_image_ids_json_path) > 0:
            with open(self.args.wide_image_ids_json_path, "r") as json_file:
                wide_images_ids = json.load(json_file)
        else:
            wide_images_ids = self.args.wide_image_ids_list
        return wide_images_ids

    def set_example_image_attributes(self):
        self.example_image_id = self.wide_images_ids[0]
        self.example_im_path = env.download_image(int(self.example_image_id))
        self.example_image = skio.imread(self.example_im_path)
        self.example_image_shape = self.example_image.shape



    def check_image_data(self, image_data):
        if (image_data['cameraAngle'] > -95) & (image_data['cameraAngle'] < -85) & (image_data['deleted'] == False):
            return True
        return False


    def filter_image_ids_list(self):
        images_data = env.eti_api.get_images_data(self.wide_images_ids, type_ids=[2])['images']
        filtered_image_ids_list = [image_data['imageID'] for image_data in images_data if self.check_image_data(image_data)]
        return filtered_image_ids_list


    def get_grid_coords(self):
        self.grid_shape = (self.example_image_shape[1]//7, self.example_image_shape[0]//8)
        top_array = np.arange(0, self.example_image_shape[1] - self.grid_shape[0] - 1, self.grid_shape[0])
        left_array = np.arange(0, self.example_image_shape[0] - 1, self.grid_shape[1])

        mesh_top, mesh_left = np.meshgrid(top_array, left_array)
        top_left_list = np.stack((mesh_top.ravel(), mesh_left.ravel()), axis=1)
        grid_coords = pd.DataFrame(top_left_list, columns = ["top", "left"])
        grid_coords['bottom'] = grid_coords['top'] + self.grid_shape[0]
        grid_coords['right'] = grid_coords['left'] + self.grid_shape[1]
        boxes_coords_list = []
        for _, row in grid_coords.iterrows():
            shapely_box = box(row['left'], row['top'], row['right'], row['bottom'])
            boxes_coords_list.append(shapely_box)
        self.boxes_coords = boxes_coords_list


    def get_canopy_cover_maps(self, image, im_path):
        im_id = int(os.path.basename(im_path).replace(".jpg", ""))
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


    def create_tiles_canopy_cover_csv(self):
        box_dicts_list = []
        for image_id in tqdm(self.image_ids_list):
            im_path = env.download_image(int(image_id))
            image = skio.imread(im_path)
            index_canopy_map, hsv_canopy_map = self.get_canopy_cover_maps(image, im_path)
            image_hsv_canopy_percent, image_canopy_index_percent = self.calculate_canopy_percentage(index_canopy_map, hsv_canopy_map)
            order_id = env.eti_api.get_images_order_ids([image_id])[0]
            for idx, box in enumerate(self.boxes_coords, start=1):
                left, top, right, bottom = box.bounds
                cropped_image = image[int(left):int(right), int(top):int(bottom)]
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
                box_dicts_list.append(box_dict)
        self.boxes_df = pd.DataFrame(box_dicts_list)
        self.boxes_df.to_csv(self.args.output_csv_path, index=False)






if __name__ == "__main__":

    env = RuntimeEnv()
    canopy_coverage_calculator = Canopy_Coverage_Calculator()
    canopy_coverage_calculator.create_tiles_canopy_cover_csv()







    # args = get_run_arguments()
    # image_ids_list = filter_image_ids(args.wide_image_ids)
    # boxes_coords_list = get_grid_coords(image_ids_list[0])

    # box_dicts_list = []
    # for image_id in tqdm(image_ids_list):
    #     im_path = env.download_image(int(image_id))
    #     image = skio.imread(im_path)
    #     index_canopy_map, hsv_canopy_map = get_canopy_cover_maps(image, im_path)
    #     image_hsv_canopy_percent, image_canopy_index_percent =calculate_canopy_percentage(index_canopy_map, hsv_canopy_map)
    #     order_id = env.eti_api.get_images_order_ids([image_id])[0]
    #     for idx, box in enumerate(boxes_coords_list, start=1):
    #         left, top, right, bottom = box.bounds
    #         cropped_image = image[int(left):int(right), int(top):int(bottom)]
    #         cropped_canopy_index_map = index_canopy_map[int(left):int(right), int(top):int(bottom)]
    #         cropped_canopy_hsv_map = hsv_canopy_map[int(left):int(right), int(top):int(bottom)]
    #         box_hsv_canopy_percent, box_canopy_index_percent =calculate_canopy_percentage(cropped_canopy_index_map, cropped_canopy_hsv_map)
    #         avg = int(sum([box_hsv_canopy_percent, box_canopy_index_percent])/2)
    #         box_dict = {'image_id': image_id,
    #                     'box_index': idx,
    #                     'orderID': order_id,
    #                     'image_hsv_canopy_percent': image_hsv_canopy_percent,
    #                     'image_canopy_index_percent': image_canopy_index_percent,
    #                     'box_coords': box,
    #                     'hsv_canopy_percent': box_hsv_canopy_percent,
    #                     'index_canopy_percent': box_canopy_index_percent,
    #                     'canopy_cover_avg': avg}
    #         box_dicts_list.append(box_dict)
    # boxes_df = pd.DataFrame(box_dicts_list)
    # boxes_df.to_csv(args.output_csv_path, index=False)
    # print("Done.")

