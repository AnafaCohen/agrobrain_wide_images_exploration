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
from skimage.draw import polygon
from shapely import wkt

from agrobrain_image_processing.canopy.canopy import canopy_by_hsv
from agrobrain_canopy.canopy_cover.canopy_cover import CanopyCover
from agrobrain_util.runtime.evironment import RuntimeEnv



class Infestation_Heatmap_Creator():
    def __init__(self,
                 env,
                 input_tiles_csv_path,
                 output_tiles_csv_path,
                 data_dir,
                 model_name,
                 field_heatmap_path=None,
                 save_output_csv = True,
                 save_histograms_bool = False,
                 save_heatmaps_bool = True):
        self.env=env
        self.model_name = model_name
        self.example_image_id = None
        self.example_im_path = None
        self.example_image = None
        self.example_image_shape = None
        self.input_tiles_csv_path = input_tiles_csv_path
        self.input_tiles_df = pd.read_csv(self.input_tiles_csv_path)
        self.output_tiles_df = self.input_tiles_df.copy()
        self.output_tiles_csv_path = output_tiles_csv_path
        self.save_output_csv =save_output_csv
        self.field_heatmap_path = field_heatmap_path # if None: not saving the heatmap
        self.save_histograms_bool = save_histograms_bool
        self.save_heatmaps_bool = save_heatmaps_bool
        self.data_dir = data_dir
        self.heatmaps_folder = os.path.join(self.data_dir, "heatmaps")
        self.set_example_image_attributes()


    def set_example_image_attributes(self):
        self.example_image_id = self.output_tiles_df['image_id'][0]
        self.example_im_path = self.env.download_image(int(self.example_image_id))
        self.example_image = skio.imread(self.example_im_path)
        self.example_image_shape = self.example_image.shape



    def save_histograms(self, order_tiles_df):
        order_id = order_tiles_df['orderID'][0]
        data_lists = {"hsv_canopy_percent": order_tiles_df['hsv_canopy_percent'],
                      "index_canopy_percent": order_tiles_df['index_canopy_percent'],
                      "canopy_cover_avg": order_tiles_df['canopy_cover_avg']}
        fig, axes = plt.subplots(1, 3, figsize=(10, 4))

        for i in range(len(data_lists)):
            data = data_lists[list(data_lists.keys())[i]]
            axes[i].hist(data, bins=20, color='lightseagreen', alpha=0.7)
            axes[i].set_xlabel('Canopy coverage percent')
            axes[i].set_ylabel('Frequency')
            axes[i].set_title(list(data_lists.keys())[i])
        plt.tight_layout()
        current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        plt.savefig(os.path.join(self.data_dir, f'canopy_histograms_order_{order_id}_{current_datetime}.jpg'))


    def get_order_canopy_mean(self, images_list):
        order_tiles_df = self.input_tiles_df[self.input_tiles_df['image_id'].isin(images_list)].reset_index(drop=True)
        canopy_mean_value = order_tiles_df['canopy_cover_avg'].mean()
        if self.save_histograms_bool:
            self.save_histograms(order_tiles_df)
        return canopy_mean_value


    def get_order_canopy_types_mean_std(self, images_list):
        order_tiles_df = self.input_tiles_df[self.input_tiles_df['image_id'].isin(images_list)].reset_index(drop=True)
        hsv_mean = order_tiles_df['hsv_canopy_percent'].mean()
        hsv_std = order_tiles_df['hsv_canopy_percent'].std()
        index_mean = order_tiles_df['index_canopy_percent'].mean()
        index_std = order_tiles_df['index_canopy_percent'].std()
        return hsv_mean, hsv_std, index_mean, index_std


    def predict_tile_infestation_level_mean_subtraction(self, row):
        orders_canopy_mean_value = self.orders_df[self.orders_df['orderID'] == row['orderID']]['order_canopy_mean'].values[0]
        label = row['canopy_cover_avg'] - orders_canopy_mean_value
        return int(label)


    def predict_tile_infestation_level_histograms_heuristics(self, row, zero_value=5, mean_distance=10):

        orders_canopy_hsv_mean = self.orders_df[self.orders_df['orderID'] == row['orderID']]['order_hsv_mean'].values[0]
        orders_canopy_hsv_std = self.orders_df[self.orders_df['orderID'] == row['orderID']]['order_hsv_std'].values[0]
        orders_canopy_index_mean = self.orders_df[self.orders_df['orderID'] == row['orderID']]['order_index_mean'].values[0]
        orders_canopy_index_std = self.orders_df[self.orders_df['orderID'] == row['orderID']]['order_index_std'].values[0]

        if orders_canopy_hsv_mean <= zero_value and orders_canopy_index_mean <= zero_value:
            return 0
        elif orders_canopy_hsv_mean <= zero_value and orders_canopy_index_mean > zero_value:
            return row['index_canopy_percent']
        elif orders_canopy_hsv_mean > zero_value and orders_canopy_index_mean <= zero_value:
            return row['hsv_canopy_percent']
        elif abs(orders_canopy_hsv_mean - orders_canopy_index_mean) <= mean_distance:
            return (row['index_canopy_percent'] + row['hsv_canopy_percent'])/2
        else:
            return max(row['index_canopy_percent'], row['hsv_canopy_percent'])


    def create_output_tiles_df(self):
        self.output_tiles_df['infestation_level_by_mean_subtraction'] = self.output_tiles_df.apply(self.predict_tile_infestation_level_mean_subtraction, axis=1)
        self.output_tiles_df['infestation_level_by_histograms_heuristics'] = self.output_tiles_df.apply(self.predict_tile_infestation_level_histograms_heuristics, axis=1)
        if self.save_output_csv:
            self.output_tiles_df.to_csv(self.args.output_tiles_csv_path)


    def create_image_heatmap(self, image_id):
        image_tiles_df = self.output_tiles_df[self.output_tiles_df['image_id'] == image_id]
        image_height, image_width, _ = self.example_image_shape
        heatmap = np.zeros((image_height, image_width))
        for coords, val in zip(image_tiles_df["box_coords"], image_tiles_df["infestation_level"]):
            coords = wkt.loads(coords)
            min_x, min_y, max_x, max_y = map(int, coords.bounds)
            heatmap[min_x:max_x, min_y:max_y] += val
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        return heatmap




    def save_heatmaps(self):
        if self.save_heatmaps_bool:
            print(f"Saving heatmaps to {self.heatmaps_folder}")
            os.makedirs(self.heatmaps_folder, exist_ok=True)
            for i, row in tqdm(self.orders_df.iterrows(), total=len(self.orders_df)):
                for image_id in list(row['image_ids_list']):
                    image_heatmap_path = os.path.join(self.heatmaps_folder, f"{image_id}_heatmap_{self.model_name}.jpg")
                    heatmap = self.create_image_heatmap(image_id)
                    fig, ax = plt.subplots()
                    heatmap_plot = ax.imshow(heatmap, cmap='hot', interpolation='nearest', origin='lower')
                    cbar = plt.colorbar(heatmap_plot, ax=ax)
                    cbar.set_label("Infestation level")
                    plt.savefig(image_heatmap_path, bbox_inches='tight')
                    plt.close()



    def create_infestation_level_csv(self):
        print(f"creating infestation level csv file.")
        self.orders_df = pd.DataFrame(self.input_tiles_df.groupby('orderID')['image_id'].apply(lambda x: list(set(x)))).reset_index()
        self.orders_df.rename(columns={'image_id': 'image_ids_list'}, inplace=True)
        tqdm.pandas()
        self.orders_df['order_canopy_mean'] = self.orders_df['image_ids_list'].progress_apply(self.get_order_canopy_mean)

        hsv_and_index_stats = self.orders_df['image_ids_list'].progress_apply(self.get_order_canopy_types_mean_std)
        self.orders_df[['order_hsv_mean', 'order_hsv_std', 'order_index_mean', 'order_index_std']] = pd.DataFrame(hsv_and_index_stats.tolist(), index=self.orders_df.index)

        # self.save_canopy_index_and_hsv_stats_histograms()

        self.create_output_tiles_df()
        self.save_heatmaps()


def get_run_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_tiles_csv_path", type=str)
    parser.add_argument("--output_tiles_csv_path", type=str)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--model_name", type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    env = RuntimeEnv()

    args = get_run_arguments()

    infestation_heatmap_creator = Infestation_Heatmap_Creator(env,
                                                              input_tiles_csv_path = args.input_tiles_csv_path,
                                                              output_tiles_csv_path = args.output_tiles_csv_path,
                                                              data_dir = args.data_dir,
                                                              model_name = args.model_name,
                                                              save_output_csv = True, save_histograms_bool = False, save_heatmaps_bool = False)
    infestation_heatmap_creator.create_infestation_level_csv()

    print("Done.")






    # def save_canopy_index_and_hsv_stats_histograms(self, order_tiles_df):
    #     order_id = order_tiles_df['orderID'][0]
    #     data_lists = {"order_hsv_mean": order_tiles_df['order_hsv_mean'],
    #                   "order_hsv_std": order_tiles_df['order_hsv_std'],
    #                   "order_index_mean": order_tiles_df['order_index_mean'],
    #                   "order_index_std": order_tiles_df['order_index_std']}
    #     fig, axes = plt.subplots(2, 2, figsize=(10, 4))

    #     for i in range(len(data_lists)):
    #         data = data_lists[list(data_lists.keys())[i]]
    #         axes[i].hist(data, bins=20, color='lightseagreen', alpha=0.7)
    #         axes[i].set_xlabel(data_lists.keys()[i])
    #         axes[i].set_ylabel('Frequency')
    #         axes[i].set_title(list(data_lists.keys())[i])
    #     plt.tight_layout()
    #     current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    #     plt.savefig(os.path.join(self.args.data_dir, f'canopy_stats_order_{order_id}_{current_datetime}.jpg'))