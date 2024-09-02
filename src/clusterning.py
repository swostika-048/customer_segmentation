# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# import pickle 
# import os,sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# print(f"path:{os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))}")
# from helper.utils import load_data,preprocess_data,statistical_info,check_null,plot_clusters
# print("Current Python Path:")
# for p in sys.path:
#     print(p)



# class KMeansScratch:
#     def __init__(self, n_clusters=3, max_iter=100):
#         self.n_clusters = n_clusters
#         self.max_iter = max_iter

#     def fit(self, X):
#         np.random.seed(42)
#         random_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
#         self.centroids = X[random_indices]
        
#         for i in range(self.max_iter):
#             self.labels = self._assign_clusters(X)
#             new_centroids = self._calculate_centroids(X)
#             if np.all(self.centroids == new_centroids):
#                 break
#             self.centroids = new_centroids
    
#     def _assign_clusters(self, X):
#         distances = np.zeros((X.shape[0], self.n_clusters))
#         for k in range(self.n_clusters):
#             distances[:, k] = np.linalg.norm(X - self.centroids[k], axis=1)
#         return np.argmin(distances, axis=1)
    
#     def _calculate_centroids(self, X):
#         centroids = np.zeros((self.n_clusters, X.shape[1]))
#         for k in range(self.n_clusters):
#             centroids[k] = X[self.labels == k].mean(axis=0)
#         return centroids
    
#     def predict(self, X):
#         return self._assign_clusters(X)


#     def calculate_wcss(self, X):
#         wcss = 0
#         for i in range(self.n_clusters):
#             cluster_points = X[self.labels == i]
#             wcss += np.sum((cluster_points - self.centroids[i]) ** 2)
#         return wcss

#     @staticmethod
#     def load_model(file_name):
#         """
#         Load a trained model from a file using pickle.
#         """
#         with open(file_name, 'rb') as file:
#             model = pickle.load(file)
#         print(f"Model loaded from {file_name}.")
#         return model

# def save_model(file_name, model):
#     """
#     Save the trained model to a file using pickle.

#     Parameters:
#     file_name (str): The name of the file where the model will be saved.
#     model (object): The trained model to be saved.
#     """
#     with open(file_name, 'wb') as file:
#         pickle.dump(model, file)
#     print(f"Model saved to {file_name}.")

# if __name__ == "__main__":
#     df = load_data('data/archive/Mall_Customers.csv')
#     print(f"df:{df}")
#     X_scaled = preprocess_data(df)


#     kmeans_scratch = KMeansScratch(n_clusters=5, max_iter=100)
#     kmeans_scratch.fit(X_scaled)
#     print(f"X_scaled:{X_scaled}")
    

#     df['Cluster'] = kmeans_scratch.predict(X_scaled)
#     # print(kmeans_scratch.centroids)
    
 
#     df.to_csv('output/clustered_data.csv', index=False)
#     print("Clustering complete. Results saved to 'clustered_data.csv'.")
    
    
#     save_model('model/kmeans_scratch_model.pkl',kmeans_scratch)

#     # Example of loading the model
#     loaded_model = KMeansScratch.load_model('model/kmeans_scratch_model.pkl')
#     labels = loaded_model.predict(X_scaled)
#     print(f"labels:{labels}")
#     # plot_clusters(X_scaled,kmeans_scratch)


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import os, sys

# Adding the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print(f"path: {os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))}")

# Importing helper functions
from helper.utils import load_data, preprocess_data, statistical_info, check_null, plot_clusters

print("Current Python Path:")
for p in sys.path:
    print(p)


class KMeansScratch:
    def __init__(self, n_clusters=3, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X):
        np.random.seed(42)
        random_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.centroids = X[random_indices]

        for i in range(self.max_iter):
            self.labels = self._assign_clusters(X)
            new_centroids = self._calculate_centroids(X)
            if np.all(self.centroids == new_centroids):
                break
            self.centroids = new_centroids

    def _assign_clusters(self, X):
        distances = np.zeros((X.shape[0], self.n_clusters))
        for k in range(self.n_clusters):
            distances[:, k] = np.linalg.norm(X - self.centroids[k], axis=1)
        return np.argmin(distances, axis=1)

    def _calculate_centroids(self, X):
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            centroids[k] = X[self.labels == k].mean(axis=0)
        return centroids

    def predict(self, X):
        return self._assign_clusters(X)

    def calculate_wcss(self, X):
        wcss = 0
        for i in range(self.n_clusters):
            cluster_points = X[self.labels == i]
            wcss += np.sum((cluster_points - self.centroids[i]) ** 2)
        return wcss

    @staticmethod
    def load_model(file_name):
        """
        Load a trained model from a file using pickle.
        """
        with open(file_name, 'rb') as file:
            model = pickle.load(file)
        print(f"Model loaded from {file_name}.")
        return model

    @staticmethod
    def save_model(file_name, model):
        """
        Save the trained model to a file using pickle.

        Parameters:
        file_name (str): The name of the file where the model will be saved.
        model (object): The trained model to be saved.
        """
        with open(file_name, 'wb') as file:
            pickle.dump(model, file)
        print(f"Model saved to {file_name}.")


if __name__ == "__main__":
    df = load_data('data/archive/Mall_Customers.csv')
    print(f"df: {df}")
    X_scaled= preprocess_data(df)


    kmeans_scratch = KMeansScratch(n_clusters=5, max_iter=100)
    kmeans_scratch.fit(X_scaled)
    df['Cluster'] = kmeans_scratch.predict(X_scaled)

    df.to_csv('output/clustered_data.csv', index=False)
    print("Clustering complete. Results saved to 'clustered_data.csv'.")

    # Save the trained model
    KMeansScratch.save_model('model/kmeans_scratch_model.pkl', kmeans_scratch)


    # Example of loading the model
    loaded_model = KMeansScratch.load_model('model/kmeans_scratch_model.pkl')
    labels = loaded_model.predict(X_scaled)
    print(f"labels: {labels}")

    # Optional: Uncomment to visualize clusters
    # plot_clusters(X_scaled, kmeans_scratch)
