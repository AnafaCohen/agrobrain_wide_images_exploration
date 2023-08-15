import pandas as pd
import argparse


class Infestation_Checker():
    def __init__(self,
                 download_dl_annotations=False):
        self.get_run_arguments()
        self.infestatin_df = pd.read_csv(self.args.infestation_level_csv_path)
        self.download_dl_annotations = download_dl_annotations

    def get_run_arguments(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--infestation_level_csv_path", type=str)
        parser.add_argument("--output_path", type=str)
        parser.add_argument("--dl_annotations_dataset_name", type=str)
        parser.add_argument("--dl_local_data_dir", type=str)
        self.args = parser.parse_args()


    def check_infestation_level_csv(self):
        print(self.infestatin_df.columns)
        print("here")



    def download_dl_annotations_to_local_dir(self):

        print("here")


    def read_dl_annotations(self):
        if self.download_dl_annotations:
            self.download_dl_annotations_to_local_dir()
        print("here")



if __name__ == "__main__":
    DL_PROJECT_NAME = "Taranis AI Annotation Projects"

    infestation_checker = Infestation_Checker(download_dl_annotations=True)
    infestation_checker.check_infestation_level_csv()
    # infestation_checker.read_dl_annotations()


    print("Done.")

