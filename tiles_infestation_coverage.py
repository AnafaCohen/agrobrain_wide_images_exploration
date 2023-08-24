import pandas as pd
import numpy as np
import argparse
from datetime import datetime


from agrobrain_apis.data_store.data_store import DataStoreAPI

MEAN_DIFF = 10
CANOPY_MODEL_NAMES_MAP = {0: 'hsv', 1: 'index'}
class Infestation_Heatmap_Creator():
    def __init__(self,
                 infestation_algo_name,
                 infestation_experiment_name,
                 canopy_algo_name,
                 canopy_experiment_name,
                 store_output=True):
        self.infestation_algo_name = infestation_algo_name
        self.infestation_experiment_name = infestation_experiment_name,
        self.canopy_algo_name = canopy_algo_name
        self.canopy_experiment_name=canopy_experiment_name
        self.store_output = store_output
        self.canopy_models = CANOPY_MODEL_NAMES_MAP
        self.tiles_canopy_coverage_df = self.get_tiles_canopy_coverage_data()
        self.orders_df = self.create_orders_df()
        self.images_dataframes_dict = self.create_images_dataframes_dict()
        self.datetime_now = self.get_datetime_now()

    def get_datetime_now(self):
        return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


    def create_images_dataframes_dict(self):
        unique_image_ids = np.unique(self.tiles_canopy_coverage_df['image_id'])
        images_dataframes_dict = {}
        for image_id in unique_image_ids:
            image_df = self.tiles_canopy_coverage_df[self.tiles_canopy_coverage_df['image_id']==image_id].reset_index(drop=True)
            order_id = image_df['orderID'][0]
            image_index_canopy_percent = image_df.reset_index(drop=True)['image_canopy_index_percent'][0]
            image_hsv_canopy_percent = image_df.reset_index(drop=True)['image_hsv_canopy_percent'][0]
            image_chosen_canopy_model = self.choose_image_canopy_model(image_index_canopy_percent, image_hsv_canopy_percent, order_id)
            image_df['canopy_model'] = image_chosen_canopy_model
            images_dataframes_dict[image_id] = image_df
        return images_dataframes_dict


    def choose_image_canopy_model(self, image_index_canopy_percent, image_hsv_canopy_percent, order_id):
        order_hsv_mean = self.orders_df[self.orders_df['orderID']==order_id]['order_hsv_mean'].reset_index(drop=True)[0]
        order_index_mean = self.orders_df[self.orders_df['orderID']==order_id]['order_index_mean'].reset_index(drop=True)[0]
        if abs(image_index_canopy_percent - image_hsv_canopy_percent) < MEAN_DIFF:
            return self.canopy_models[0]
        elif abs(image_index_canopy_percent - order_index_mean) < abs(image_hsv_canopy_percent - order_hsv_mean):
            return self.canopy_models[1]
        else:
            return self.canopy_models[0]


    def get_tiles_canopy_coverage_data(self):
        data_api = DataStoreAPI()
        all_images_canopy_data = []
        canopy_stored_data_list = data_api.list(self.canopy_algo_name, experiment=self.canopy_experiment_name)
        image_ids_list =[int(canopy_stored_data_list[i]['object']['image_id']) for i in range(len(canopy_stored_data_list))]
        for image_id in image_ids_list:
            image_id_data = data_api.get(self.canopy_algo_name, experiment=self.canopy_experiment_name, image_id=image_id)
            all_images_canopy_data.extend(image_id_data)
        canopy_data_df = pd.DataFrame(all_images_canopy_data)
        return canopy_data_df


    def predict_tiles_infestation_level(self):
        for image_id, image_df in self.images_dataframes_dict.items():
            image_df['pred_infestation_level_by_mean_subtraction'] = image_df.apply(self.predict_tile_infestation_level_mean_subtraction, axis=1)
            image_df['pred_tile_infestation_level_per_image'] = image_df.apply(self.pred_tile_infestation_level_per_image, axis=1)
            self.store_image_output_in_db(image_id,image_df)
        # self.store_output_in_db()


    def store_image_output_in_db(self, image_id, image_tiles_df):
        data_api = DataStoreAPI()
        data_api.store(self.infestation_algo_name, payload=image_tiles_df.to_dict(), image_id=image_id, experiment=self.infestation_experiment_name, metadata={"timestamp": self.datetime_now})
        # CHECKUPS:
        # data_api.list(self.infestation_algo_name, experiment=self.infestation_experiment_name, metadata={"timestamp": self.datetime_now})
        # len(data_api.list(self.infestation_algo_name, experiment=self.infestation_experiment_name, metadata={"timestamp": self.datetime_now}))
        # data_api.get(self.infestation_algo_name, experiment=self.infestation_experiment_name, metadata={"timestamp": self.datetime_now})



    def store_output_in_db(self):
        data_api = DataStoreAPI()
        for image_id in self.images_dataframes_dict:
            image_tiles_df = self.images_dataframes_dict[image_id]
            data_api.store(self.infestation_algo_name, payload=image_tiles_df.to_dict(), image_id=image_id, experiment=self.infestation_experiment_name, metadata={"timestamp": self.datetime_now})


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
        df_grouped_by_orders = self.group_df_by_orders(self.tiles_canopy_coverage_df)
        all_orders_list = []
        for order_id, group_indices in df_grouped_by_orders.groups.items():
            order_df = self.tiles_canopy_coverage_df.loc[group_indices]
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
        return {'canopy_model': 'mean_subtruction', 'canopy_percent': int(label)}


    def pred_tile_infestation_level_per_image(self, row):
        canopy_model = self.images_dataframes_dict[row['image_id']]['canopy_model'].values[0]
        canopy_percent = row[f"{canopy_model}_canopy_percent"]
        return {'canopy_model': canopy_model, 'canopy_percent': canopy_percent}


def get_run_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infestation_algo_name", type=str)
    parser.add_argument("--canopy_algo_name", type=str)
    parser.add_argument("--canopy_experiment_name", type=str)
    parser.add_argument("--infestation_experiment_name", type=str)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_run_arguments()

    infestation_heatmap_creator = Infestation_Heatmap_Creator(infestation_algo_name = args.infestation_algo_name,
                                                              infestation_experiment_name = args.infestation_experiment_name,
                                                              canopy_algo_name = args.canopy_algo_name,
                                                              canopy_experiment_name = args.canopy_experiment_name)
    infestation_heatmap_creator.predict_tiles_infestation_level()

    print("Done.")
