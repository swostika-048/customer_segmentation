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

    kmeans_scratch = KMeansScratch.load_model(model_file)

    
    input_data = pd.DataFrame({
        'Age': [45],
        'Annual Income (k$)': [34],
        'Spending Score (1-100)': [45],
        'Gender': ['Male']
    })
    print(f"data:{input_data}")
    processed_data = preprocess_data(input_data)
    print(f"processed:{preprocess_data}")
    

    cluster = kmeans_scratch.predict(processed_data)

    
    print(f"cluster:{cluster}")
    print(f"The customer belongs to cluster {cluster[0]}.")






