import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import sys,os
import pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def load_data(file_path):
    """Load the dataset from a CSV file."""
    return pd.read_csv(file_path)

def load_model(model_path):
    # Load the traed model
    with open(model_path, 'rb') as model_file:
        kmeans = pickle.load(model_file)
    return kmeans

def preprocess_data(data, features):
    """Standardize the selected features in the dataset."""
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data[features])
    return data_scaled


def determine_optimal_clusters(data_scaled):
    """Determine the optimal number of clusters using the Elbow Method."""
    sse = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data_scaled)
        sse.append(kmeans.inertia_)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), sse, marker='o')
    plt.title('Elbow Method to Determine Optimal Number of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('SSE')
    plt.show()


def add_cluster_labels(data, clusters):
    """Add the cluster labels to the original dataset."""
    data['Cluster'] = clusters
    return data

def feature_def():
    features = ['Age', 'Annual_Income', 'Spending_Score', 'Transaction_Amount', 
                'Transaction_Frequency', 'Web_Activity', 'Product_Views', 'Time_on_Website']
    return features