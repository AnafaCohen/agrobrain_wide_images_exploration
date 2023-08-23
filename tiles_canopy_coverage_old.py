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


def get_run_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wide_image_ids", type=json.loads)
    parser.add_argument("--output_csv_path", type=str)
    args = parser.parse_args()
    return args


def check_image_data(image_data):
    if (image_data['cameraAngle'] > -95) & (image_data['cameraAngle'] < -85) & (image_data['deleted'] == False):
        return True
    return False


def filter_image_ids(wide_image_ids):
    images_data = env.eti_api.get_images_data(wide_image_ids, type_ids=[2])['images']
    filtered_image_ids_list = [image_data['imageID'] for image_data in images_data if check_image_data(image_data)]
    return filtered_image_ids_list


def get_canopy_cover_maps(image, im_path):
    im_id = int(os.path.basename(im_path).replace(".jpg", ""))
    print(f"calculating image {im_id} index canopy map...", end="\r", flush=True)
    index_canopy_map = CanopyCover.canopy_cover(im_path)[0].astype(np.uint8) * 255
    print(f"calculating image {im_id} hsv canopy map...", end="\r", flush=True)
    hsv_canopy_map = canopy_by_hsv(image).astype(np.uint8) * 255
    return index_canopy_map, hsv_canopy_map


def calculate_canopy_percentage(index_canopy_map, hsv_canopy_map):
    im_size = index_canopy_map.shape[0] * index_canopy_map.shape[1]
    im_size_hsv = hsv_canopy_map.shape[0] * hsv_canopy_map.shape[1]
    hsv_canopy_sum = np.count_nonzero(hsv_canopy_map)
    hsv_canopy_percent = int(hsv_canopy_sum/im_size * 100)
    canopy_index_sum = np.count_nonzero(index_canopy_map)
    canopy_index_percent = int(canopy_index_sum/im_size * 100)
    return hsv_canopy_percent, canopy_index_percent



def save_histograms(hsv_canopy_percent_list, canopy_index_percent_list, dir="data"):
    data_lists = {"hsv_percent_list": hsv_canopy_percent_list,
                  "index_percent_list": canopy_index_percent_list}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for i in range(len(data_lists)):
        data = data_lists[list(data_lists.keys())[i]]
        axes[i].hist(data, bins=20, color='lightseagreen', alpha=0.7)
        axes[i].set_xlabel('Canopy coverage percent')
        axes[i].set_ylabel('Frequency')
        axes[i].set_title(list(data_lists.keys())[i].replace("_list", ""))
    plt.tight_layout()
    current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    plt.savefig(os.path.join(dir, f'side_by_side_histograms_{current_datetime}.jpg'))
    print('h')



def get_grid_coords(example_image_id):
    example_im_path = env.download_image(int(example_image_id))
    example_image = skio.imread(example_im_path)
    example_image_shape = example_image.shape
    grid_shape = (example_image_shape[1]//7, example_image_shape[0]//8)

    top_array = np.arange(0, example_image_shape[1] - grid_shape[0] - 1, grid_shape[0])
    left_array = np.arange(0, example_image_shape[0] - 1, grid_shape[1])

    mesh_top, mesh_left = np.meshgrid(top_array, left_array)
    top_left_list = np.stack((mesh_top.ravel(), mesh_left.ravel()), axis=1)
    grid_coords = pd.DataFrame(top_left_list, columns = ["top", "left"])
    grid_coords['bottom'] = grid_coords['top'] + grid_shape[0]
    grid_coords['right'] = grid_coords['left'] + grid_shape[1]
    box_list = []
    box_size_list=[]
    for _, row in grid_coords.iterrows():
        shapely_box = box(row['left'], row['top'], row['right'], row['bottom'])
        box_list.append(shapely_box)
        box_size_list.append(shapely_box.area)
    return box_list



if __name__ == "__main__":
    env = RuntimeEnv()
    args = get_run_arguments()
    image_ids_list = filter_image_ids(args.wide_image_ids)
    boxes_coords_list = get_grid_coords(image_ids_list[0])

    box_dicts_list = []
    for image_id in tqdm(image_ids_list):
        im_path = env.download_image(int(image_id))
        image = skio.imread(im_path)
        index_canopy_map, hsv_canopy_map = get_canopy_cover_maps(image, im_path)
        image_hsv_canopy_percent, image_canopy_index_percent =calculate_canopy_percentage(index_canopy_map, hsv_canopy_map)
        order_id = env.eti_api.get_images_order_ids([image_id])[0]
        for idx, box in enumerate(boxes_coords_list, start=1):
            left, top, right, bottom = box.bounds
            cropped_image = image[int(left):int(right), int(top):int(bottom)]
            cropped_canopy_index_map = index_canopy_map[int(left):int(right), int(top):int(bottom)]
            cropped_canopy_hsv_map = hsv_canopy_map[int(left):int(right), int(top):int(bottom)]
            box_hsv_canopy_percent, box_canopy_index_percent =calculate_canopy_percentage(cropped_canopy_index_map, cropped_canopy_hsv_map)
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
    boxes_df = pd.DataFrame(box_dicts_list)
    boxes_df.to_csv(args.output_csv_path, index=False)
    print("Done.")





    #     chosen_canopy_map = choose_canopy_map(index_canopy_map, hsv_canopy_map)
    # hsv_canopy_percent_list = []
    # canopy_index_percent_list = []

    #     # ~~~~~ tmp~~~~~~
    #     hsv_canopy_percent, canopy_index_percent = calculate_canopy_percentage(index_canopy_map, hsv_canopy_map)
    #     hsv_canopy_percent_list.append(hsv_canopy_percent)
    #     canopy_index_percent_list.append(canopy_index_percent)
    #     # ~~~~~~~~~~~~~~~
    #     print(f"something {image_id} ", end="\r", flush=True)

    #     # chosen_canopy_map, chosen_canopy_map_type, reason = choose_canopy_map(index_canopy_map, hsv_canopy_map)
    # data_dir = os.path.join("data", "canopy_coverage")
    # os.makedirs(data_dir, exist_ok=True)
    # save_histograms(hsv_canopy_percent_list, canopy_index_percent_list, dir=data_dir)
    # print("Done")





# -- read image
# -- calculate canopy maps:  hsv, canopy index
# choose map for the image
# crop image
# calculate canopy value for each tile
# save csv with tile details, including coordinates, image_id and canopy value




    # def create_tiles_canopy_cover_csv(self):
    #     box_dicts_list = []
    #     for image_id in tqdm(self.image_ids_list):
    #         im_path = self.env.download_image(int(image_id))
    #         image = skio.imread(im_path)
    #         index_canopy_map, hsv_canopy_map = self.get_canopy_cover_maps(image, im_path)
    #         image_hsv_canopy_percent, image_canopy_index_percent = self.calculate_canopy_percentage(index_canopy_map, hsv_canopy_map)
    #         order_id = self.env.eti_api.get_images_order_ids([image_id])[0]
    #         for idx, box in enumerate(self.boxes_coords, start=1):
    #             left, top, right, bottom = box.bounds
    #             cropped_canopy_index_map = index_canopy_map[int(left):int(right), int(top):int(bottom)]
    #             cropped_canopy_hsv_map = hsv_canopy_map[int(left):int(right), int(top):int(bottom)]
    #             box_hsv_canopy_percent, box_canopy_index_percent = self.calculate_canopy_percentage(cropped_canopy_index_map, cropped_canopy_hsv_map)
    #             avg = int(sum([box_hsv_canopy_percent, box_canopy_index_percent])/2)
    #             box_dict = {'image_id': image_id,
    #                         'box_index': idx,
    #                         'orderID': order_id,
    #                         'image_hsv_canopy_percent': image_hsv_canopy_percent,
    #                         'image_canopy_index_percent': image_canopy_index_percent,
    #                         'box_coords': box,
    #                         'hsv_canopy_percent': box_hsv_canopy_percent,
    #                         'index_canopy_percent': box_canopy_index_percent,
    #                         'canopy_cover_avg': avg}
    #             box_dicts_list.append(box_dict)
    #     self.boxes_df = pd.DataFrame(box_dicts_list)
    #     self.boxes_df.to_csv(self.output_csv_path, index=False)


