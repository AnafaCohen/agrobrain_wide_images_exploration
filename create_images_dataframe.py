import os
import pandas as pd
import warnings
from tqdm import tqdm
os.environ["NAMESPACE"]="research"
os.environ["PROFILE"]="local"
from agrobrain_util.runtime.evironment import RuntimeEnv
from agrobrain_util.infra.app_config import application_config as cfg

env = RuntimeEnv()
categories_dict = cfg['tags']['categories']

# READ ORDERS CSV FROM JIRA DATA
orders_csv_2022_path = '/mnt/disks/datasets/wide_images/us_2022_emergence_analysis_jira.csv'
orders_df = pd.read_csv(orders_csv_2022_path)
orders_df = orders_df.dropna(subset=['Order ID'])
orders_df['Order ID'] = orders_df['Order ID'].astype(int)
orders_list = list(orders_df['Order ID'])

# ADD IMAGES DATA BY ORDER ID FROM ETI

example_images_df = env.eti_api.get_images_data_by_orderid(orders_list[0])['images']
images_df = pd.DataFrame(columns=example_images_df[0].keys())

folder_dir = '/mnt/disks/datasets/wide_images/images_df_folder_1'
os.makedirs(folder_dir, exist_ok=True)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    for i, order in enumerate(tqdm(orders_list)):
        order_df = pd.DataFrame(env.eti_api.get_images_data_by_orderid(order)['images'])
        images_df = pd.concat([images_df, order_df], axis='rows', ignore_index=True)
        if len(images_df) > 500:
            images_df.to_csv(f"/mnt/disks/datasets/wide_images/images_df_folder_1/images_df_{i}.csv")
            images_df = pd.DataFrame(columns=example_images_df[0].keys())
    images_df.to_csv(f"/mnt/disks/datasets/wide_images/images_df_folder_1/images_df_{i}.csv")