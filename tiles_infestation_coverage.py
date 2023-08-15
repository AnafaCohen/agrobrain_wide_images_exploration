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
                 field_heatmap_path=None,
                 save_output_csv = True,
                 save_histograms_bool = True,
                 save_heatmaps_bool = True):
        self.get_run_arguments()
        self.example_image_id = None
        self.example_im_path = None
        self.example_image = None
        self.example_image_shape = None
        self.input_tiles_csv_path = self.args.input_tiles_csv_path
        self.input_tiles_df = pd.read_csv(self.input_tiles_csv_path)
        self.output_tiles_df = self.input_tiles_df.copy()
        self.output_tiles_csv_path = self.args.output_tiles_csv_path
        self.save_output_csv =save_output_csv
        self.field_heatmap_path = field_heatmap_path # if None: not saving the heatmap
        self.save_histograms_bool = save_histograms_bool
        self.save_heatmaps_bool = save_heatmaps_bool
        self.heatmaps_folder = os.path.join(self.args.data_dir, "heatmaps")
        self.set_example_image_attributes()

    def get_run_arguments(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--input_tiles_csv_path", type=str)
        parser.add_argument("--output_tiles_csv_path", type=str)
        parser.add_argument("--data_dir", type=str)
        parser.add_argument("--model_name", type=str)
        self.args = parser.parse_args()


    def set_example_image_attributes(self):
        self.example_image_id = self.output_tiles_df['image_id'][0]
        self.example_im_path = env.download_image(int(self.example_image_id))
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
        plt.savefig(os.path.join(self.args.data_dir, f'canopy_histograms_order_{order_id}_{current_datetime}.jpg'))


    def get_order_canopy_threshold(self, images_list):
        order_tiles_df = self.input_tiles_df[self.input_tiles_df['image_id'].isin(images_list)].reset_index(drop=True)
        if self.save_histograms_bool:
            self.save_histograms(order_tiles_df)
        # TODO: WRITE HURISTICS
        threshold = order_tiles_df['canopy_cover_avg'].mean()

        hsv_mean = order_tiles_df['hsv_canopy_percent'].mean()
        hsv_std = order_tiles_df['hsv_canopy_percent'].std()

        index_mean = order_tiles_df['index_canopy_percent'].mean()
        index_std = order_tiles_df['index_canopy_percent'].std()
        return threshold


    def predict_tile_infestation_level(self, row):
        # TODO: WRITE HURISTICS
        orders_threshold = self.orders_df[self.orders_df['orderID'] == row['orderID']]['order_canopy_threshold'].values[0]
        label = row['canopy_cover_avg'] - orders_threshold
        return int(label)


    def create_output_tiles_df(self):
        self.output_tiles_df['infestation_level'] = self.output_tiles_df.apply(self.predict_tile_infestation_level, axis=1)
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
                    image_heatmap_path = os.path.join(self.heatmaps_folder, f"{image_id}_heatmap_{self.args.model_name}.jpg")
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
        self.orders_df['order_canopy_threshold'] = self.orders_df['image_ids_list'].progress_apply(self.get_order_canopy_threshold)
        self.create_output_tiles_df()
        self.save_heatmaps()


if __name__ == "__main__":
    env = RuntimeEnv()

    infestation_heatmap_creator = Infestation_Heatmap_Creator()
    infestation_heatmap_creator.create_infestation_level_csv()

    print("Done.")