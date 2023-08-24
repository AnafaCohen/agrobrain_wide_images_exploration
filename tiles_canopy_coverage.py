import os
import numpy as np
import pandas as pd
import argparse
import skimage.io as skio
from datetime import datetime
from tqdm import tqdm
import json
import imagesize
from datetime import datetime

from agrobrain_image_processing.canopy.canopy import canopy_by_hsv
from agrobrain_canopy.canopy_cover.canopy_cover import CanopyCover
from agrobrain_util.runtime.evironment import RuntimeEnv
from agrobrain_apis.data_store.data_store import DataStoreAPI


N_TILES_X = 7
N_TILES_Y = 8

class Canopy_Coverage_Calculator():
    def __init__(self,
                 env,
                 wide_image_ids_list,
                 wide_image_ids_json_path,
                 canopy_algo_name,
                 experiment_name):
        self.env = env
        self.input_image_ids_list = wide_image_ids_list
        self.input_image_ids_json_path = wide_image_ids_json_path
        self.canopy_algo_name = canopy_algo_name,
        self.experiment_name = experiment_name,
        # self.tiles_df = None
        self.image_ids_list_not_filtered = self.read_wide_images_ids_list()
        self.images_data = self.get_images_data_from_eti()
        self.image_ids_list = self.filter_image_ids_list()
        self.example_image_shape = self.set_example_image_attributes()
        self.tiles_coords = self.get_grid_coords()
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
        example_image_shape = {"width": imagesize.get(example_im_path)[0],
                               "height": imagesize.get(example_im_path)[1]}
        return example_image_shape


    def check_image_data(self, image_data):
        if (image_data['cameraAngle'] > -95) & (image_data['cameraAngle'] < -85) & (image_data['deleted'] == False):
            return True
        return False

    def get_images_data_from_eti(self):
        images_data = self.env.eti_api.get_images_data(self.image_ids_list_not_filtered, type_ids=[2])['images']
        return pd.DataFrame(images_data)


    def filter_image_ids_list(self):
        filtered_image_ids_list = [image_data['imageID'] for i, image_data in self.images_data.iterrows() if self.check_image_data(image_data)]
        return filtered_image_ids_list


    def get_grid_coords(self):
        self.grid_shape = (self.example_image_shape['width']//N_TILES_X, self.example_image_shape['height']//N_TILES_Y)
        top_array = np.arange(0, self.example_image_shape['width'] - self.grid_shape[0] - 1, self.grid_shape[0])
        left_array = np.arange(0, self.example_image_shape['height'] - 1, self.grid_shape[1])
        mesh_top, mesh_left = np.meshgrid(top_array, left_array)
        top_left_list = np.stack((mesh_top.ravel(), mesh_left.ravel()), axis=1)
        grid_coords = pd.DataFrame(top_left_list, columns = ["top", "left"])
        grid_coords['bottom'] = grid_coords['top'] + self.grid_shape[0]
        grid_coords['right'] = grid_coords['left'] + self.grid_shape[1]
        tiles_coords_list = []
        for _, row in grid_coords.iterrows():
            tile = [int(row['left']), int(row['top']), int(row['right']), int(row['bottom'])]
            tiles_coords_list.append(tile)
        return tiles_coords_list


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


    def get_tile_dict(self, idx, tile, image_id, order_id, index_canopy_map, hsv_canopy_map, image_hsv_canopy_percent, image_canopy_index_percent):
        left, top, right, bottom = tile
        cropped_canopy_index_map = index_canopy_map[int(left):int(right), int(top):int(bottom)]
        cropped_canopy_hsv_map = hsv_canopy_map[int(left):int(right), int(top):int(bottom)]
        tile_hsv_canopy_percent, tile_canopy_index_percent = self.calculate_canopy_percentage(cropped_canopy_index_map, cropped_canopy_hsv_map)
        avg = int(sum([tile_hsv_canopy_percent, tile_canopy_index_percent])/2)
        tile_dict = {'image_id': image_id,
                    'tile_index': idx,
                    'orderID': int(order_id),
                    'image_hsv_canopy_percent': image_hsv_canopy_percent,
                    'image_canopy_index_percent': image_canopy_index_percent,
                    'tiles_coords': tile,
                    'hsv_canopy_percent': tile_hsv_canopy_percent,
                    'index_canopy_percent': tile_canopy_index_percent,
                    'canopy_cover_avg': avg}
        return tile_dict


    def calc_tiles_canopy_cover(self):
        # tiles_dicts_list = []
        for image_id in tqdm(self.image_ids_list):
            tiles_dicts_list = []
            im_path = self.env.download_image(int(image_id))
            image = skio.imread(im_path)
            index_canopy_map, hsv_canopy_map = self.get_canopy_cover_maps(image, im_path)
            image_hsv_canopy_percent, image_canopy_index_percent = self.calculate_canopy_percentage(index_canopy_map, hsv_canopy_map)
            order_id = self.images_data[self.images_data['imageID']==image_id]['orderID'].values[0]
            for idx, tile in enumerate(self.tiles_coords, start=1):
                tile_dict = self.get_tile_dict(idx, tile, image_id, order_id, index_canopy_map, hsv_canopy_map, image_hsv_canopy_percent, image_canopy_index_percent)
                tiles_dicts_list.append(tile_dict)
            self.store_tiles_predictions(image_id, tiles_dicts_list)
            # self.tiles_df = pd.DataFrame(tiles_dicts_list)
            # self.store_tiles_predictions(image_id)

    def store_tiles_predictions(self, image_id, tiles_dicts_list):
        data_api = DataStoreAPI()
        # data_api.store(self.canopy_algo_name, image_id=image_id, order_id=tiles_dicts_list[0]['orderID'], experiment=self.experiment_name, metadata={"timestamp": self.datetime_now, "tiles_data_list":tiles_dicts_list})
        data_api.store(self.canopy_algo_name, payload=tiles_dicts_list, image_id=image_id, order_id=tiles_dicts_list[0]['orderID'], experiment=self.experiment_name, metadata={"timestamp": self.datetime_now})

        # print("here")

        # for i, tile in self.tiles_df.iterrows():
            # data_api.store(self.canopy_algo_name, payload=tile.to_dict(), image_id=image_id, order_id=tile.orderID, experiment=self.experiment_name, metadata={"timestamp": self.datetime_now, "tiles_coords": tile.tiles_coords})
            # print("here")


def get_run_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wide_image_ids_list", type=json.loads)
    parser.add_argument("--wide_image_ids_json_path", type=str)
    parser.add_argument("--canopy_algo_name", type=str)
    parser.add_argument("--experiment_name", type=str)
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    env = RuntimeEnv()
    args = get_run_arguments()
    canopy_coverage_calculator = Canopy_Coverage_Calculator(env,
                                                            wide_image_ids_list = args.wide_image_ids_list,
                                                            wide_image_ids_json_path = args.wide_image_ids_json_path,
                                                            canopy_algo_name = args.canopy_algo_name,
                                                            experiment_name = args.experiment_name)
    canopy_coverage_calculator.calc_tiles_canopy_cover()

