import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util_s.data_visualization import visualize_clusters,visualize_clusters_3d
from util_s.utils import feature_def,load_data,preprocess_data
import pickle

class KMeansScratch:
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        self.centroids = {}
        self.labels_ = np.zeros(X.shape[0])

        random_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        for i in range(self.n_clusters):
            self.centroids[i] = X[random_indices[i]]

      
        for i in range(self.max_iter):
           
            self.labels_ = self._assign_clusters(X)

            
            prev_centroids = self.centroids.copy()

            
            self._update_centroids(X)

            
            is_converged = self._is_converged(prev_centroids, self.centroids)
            if is_converged:
                break

    def _assign_clusters(self, X):
        labels = np.zeros(X.shape[0])
        for i, data_point in enumerate(X):
            distances = [np.linalg.norm(data_point - self.centroids[centroid]) for centroid in self.centroids]
            labels[i] = np.argmin(distances)
        return labels

    def _update_centroids(self, X):
        for i in range(self.n_clusters):
            points = [X[j] for j in range(len(X)) if self.labels_[j] == i]
            if points:  # Avoid empty clusters
                self.centroids[i] = np.mean(points, axis=0)

    def _is_converged(self, prev_centroids, current_centroids):
        for i in range(self.n_clusters):
            if np.linalg.norm(current_centroids[i] - prev_centroids[i]) > self.tol:
                return False
        return True

    def predict(self, X):
        return self._assign_clusters(X)


def main():
    data = pd.read_csv('data/customer_data.csv')
    label_encoder = LabelEncoder()
    data['Gender'] = label_encoder.fit_transform(data['Gender'])
    features=feature_def()
    scaler=preprocess_data(data,features)
    X=data.drop(columns=['Customer_ID'])
    
    # pca = PCA(n_components=3)
    # X_pca = pca.fit_transform(X)
    # print(X_pca)
    # pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3'])
    # pca_df.to_csv('output/after_pca.csv', index=False)

    # Applying KMeans from scratch
    kmeans = KMeansScratch(n_clusters=5)
    kmeans.fit(scaler)
    with open('model/kmeans_model.pkl', 'wb') as model_file:
        pickle.dump(kmeans, model_file)


    data['Cluster'] = kmeans.labels_
    data.to_csv('output/customer_data_with_clusters.csv', index=False)
    visualize_clusters_3d(data, 'Annual_Income', 'Spending_Score', 'Transaction_Amount')


if __name__ == "__main__":
    main()
