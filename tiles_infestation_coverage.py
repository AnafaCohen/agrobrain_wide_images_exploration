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
from agrobrain_apis.data_store.data_store import DataStoreAPI


class Infestation_Heatmap_Creator():
    def __init__(self,
                 input_tiles_csv_path,
                 output_tiles_csv_path,
                #  canopy_experiment_name,
                 save_output_csv = True):
        self.input_tiles_csv_path = input_tiles_csv_path
        self.input_tiles_df = pd.read_csv(self.input_tiles_csv_path)
        self.orders_df = self.create_orders_df()
        self.output_tiles_csv_path = output_tiles_csv_path
        self.output_tiles_df = None
        self.save_output_csv = save_output_csv
        self.output_tiles_df = None
        self.canopy_experiment_name="infestation_by_canopy_coverage_v0"


    def get_boxes_canopy_coverage_data(self, image_id):
        data_api = DataStoreAPI()
        for i, box in self.boxes_df.iterrows():
            data_api.store("infestation_wide_images", payload=box.to_dict(), image_id=image_id, experiment=self.experiment_name, metadata={"timestamp": self.datetime_now})


    def create_output_tiles_df(self):
        self.output_tiles_df = self.input_tiles_df.copy()
        self.output_tiles_df['infestation_level_by_mean_subtraction'] = self.output_tiles_df.apply(self.predict_tile_infestation_level_mean_subtraction, axis=1)
        self.output_tiles_df['infestation_level_by_histograms_heuristics'] = self.output_tiles_df.apply(self.predict_tile_infestation_level_histograms_heuristics, axis=1)
        if self.save_output_csv:
            self.output_tiles_df.to_csv(self.output_tiles_csv_path)


    def group_df_by_orders(self, input_tiles_df):
        orders_df = input_tiles_df.groupby('orderID')
        return orders_df


    def get_order_canopy_stats(self, order_df):
        hsv_mean = order_df['hsv_canopy_percent'].mean()
        hsv_std = order_df['hsv_canopy_percent'].std()
        index_mean = order_df['index_canopy_percent'].mean()
        index_std = order_df['index_canopy_percent'].std()
        return hsv_mean, hsv_std, index_mean, index_std

    def create_orders_df(self):
        df_grouped_by_orders = self.group_df_by_orders(self.input_tiles_df)
        all_orders_list = []
        for order_id, group_indices in df_grouped_by_orders.groups.items():
            order_df = self.input_tiles_df.loc[group_indices]
            hsv_mean, hsv_std, index_mean, index_std = self.get_order_canopy_stats(order_df)
            canopy_mean_value = order_df['canopy_cover_avg'].mean()
            order_out = {
                "orderID": order_id,
                "order_hsv_mean": hsv_mean,
                "order_hsv_std": hsv_std,
                "order_index_mean": index_mean,
                "order_index_std": index_std,
                "order_canopy_mean": canopy_mean_value}
            all_orders_list.append(order_out)
        orders_df = pd.DataFrame(all_orders_list)
        return orders_df

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


def get_run_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_tiles_csv_path", type=str)
    parser.add_argument("--output_tiles_csv_path", type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_run_arguments()

    infestation_heatmap_creator = Infestation_Heatmap_Creator(input_tiles_csv_path = args.input_tiles_csv_path,
                                                              output_tiles_csv_path = args.output_tiles_csv_path,
                                                              save_output_csv = True)
    infestation_heatmap_creator.create_output_tiles_df()

    print("Done.")
