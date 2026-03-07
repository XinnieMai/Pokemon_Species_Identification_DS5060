# This is a placeholder, loader will be developed via a feature branch and then merged into main
# this is to create file structure

# https://www.kaggle.com/datasets/vishalsubbiah/pokemon-images-and-types

import kagglehub
import pandas as pd


#Download data via kagglehub if it doesn't already eist
class Loader:

    data_path = None
    def download_data(self):
        # Check if dataset is already downloaded
        self.data_path = kagglehub.dataset_download("vishalsubbiah/pokemon-images-and-types")
        print("Dataset downloaded to:", self.data_path)


    def get_dictionary(self):
       return pd.read_csv(self.data_path + "/pokemon.csv")
    
    def get_images(self):