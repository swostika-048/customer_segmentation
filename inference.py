import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from helper.utils import load_data,preprocess_data
from src.clusterning import KMeansScratch
import pandas as pd
# from sklearn.cluster import KMeans


def load_model(file_name):
    """
    Load a trained model from a file using pickle.
    """
    with open(file_name, 'rb') as file:
        model = pickle.load(file)
    print(f"Model loaded from {file_name}.")
    return model



if __name__ == "__main__":
    model_file = 'model/kmeans_scratch_model.pkl'
    df=load_data('/home/swostika/Documents/datascience/ds_projects/market_basket/data/archive/Mall_Customers.csv')
    kmeans_scratch = KMeansScratch.load_model(model_file)

    

    processed_data = preprocess_data(df)
    print(f"processed:{preprocess_data}")
    

    cluster = kmeans_scratch.predict(processed_data)

    
    print(f"cluster:{cluster}")
    print(f"The customer belongs to cluster {cluster[0]}.")






