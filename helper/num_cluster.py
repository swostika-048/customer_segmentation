import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import os,sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# print(f"path:{os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))}")
from utils import load_data,preprocess_data,statistical_info,check_null
from src.clusterning import KMeansScratch
# print("Current Python Path:")
# print(sys.path)
# for p in sys.path:
#     print(p)


def determine_optimal_k(X_scaled):
    wcss = []
    k_values = range(1, 11)
    
    for k in k_values:
        kmeans = KMeansScratch(n_clusters=k, max_iter=300)  
        kmeans.fit(X_scaled)
        wcss.append(kmeans.calculate_wcss(X_scaled)) 
    # Plot the elbow graph
    plt.figure(figsize=(10, 5))
    plt.plot(k_values, wcss, marker='o')
    plt.title('Elbow Method For Optimal K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.show()
    

    diffs = np.diff(wcss)
    second_diffs = np.diff(diffs)
    optimal_k = np.argmin(second_diffs) + 2 
    
    return optimal_k


if __name__ == "__main__":
    df = load_data('/home/swostika/Documents/datascience/ds_projects/market_basket/data/archive/Mall_Customers.csv')
    X_scaled = preprocess_data(df)
    
    optimal_k = determine_optimal_k(X_scaled)
    print(f"The optimal number of clusters (k) is: {optimal_k}")
