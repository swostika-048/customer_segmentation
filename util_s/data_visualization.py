import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os,sys
from sklearn.preprocessing import LabelEncoder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util_s.utils import load_data,preprocess_data,determine_optimal_clusters,add_cluster_labels,feature_def

label_encoder = LabelEncoder()
def apply_kmeans_clustering(data_scaled, n_clusters):
    """Apply K-Means clustering to the standardized data."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(data_scaled)
    return clusters




def visualize_clusters(data, features):
    """Visualize the clusters using a pair plot."""
    sns.pairplot(data, hue='Cluster', vars=features)
    plt.show()

def visualize_clusters(data, features):
    """Visualize the clusters by plotting each feature against each other one by one."""
    num_features = len(features)
    
    for i in range(num_features):
        for j in range(i + 1, num_features):
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=features[i], y=features[j], hue='Cluster', data=data, palette='viridis')
            plt.title(f'Clusters: {features[i]} vs {features[j]}')
            plt.xlabel(features[i])
            plt.ylabel(features[j])
            plt.show()






def visualize_clusters_3d(data, feature_x, feature_y, feature_z):
    """Visualize the clusters in 3D space using selected features and include a legend."""
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    colors = {0: 'red', 1: 'blue', 2: 'orange', 3: '#39FF14', 4: 'purple', 5: 'brown', 6: 'yellow', 7: 'gray'}

    
    for cluster_id in np.unique(data['Cluster']):
        cluster_data = data[data['Cluster'] == cluster_id]
        ax.scatter(cluster_data[feature_x], cluster_data[feature_y], cluster_data[feature_z],color=colors[cluster_id],
                   label=f'Cluster {cluster_id}', s=20)
    
    ax.set_xlabel(feature_x)
    ax.set_ylabel(feature_y)
    ax.set_zlabel(feature_z)
    plt.title(f'Clusters: {feature_x} vs {feature_y} vs {feature_z}')
    plt.legend(title='Cluster ID', loc='best')  
    plt.show()




def visualize_spending_score_by_cluster(data):
    """Visualize Spending Score by Cluster."""
    sns.boxplot(x='Cluster', y='Spending_Score', data=data)
    plt.title('Spending Score by Cluster')
    plt.show()



def plot_all_distributions(data):
    """Plot histograms, box plots, and KDE plots for each feature in the dataframe one at a time."""
    
    # Label encode the Gender column if it's categorical
    label_encoder = LabelEncoder()
    data['Gender'] = label_encoder.fit_transform(data['Gender'])
    data = data.drop(columns=['Customer_ID'])

    for column in data.columns:
        plt.figure(figsize=(8, 6))

        # Histogram
        plt.hist(data[column], bins=30, color='#007acc')
        plt.title(f'Histogram of {column}', fontsize=14)
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()


        # KDE Plot
        if data[column].nunique() > 1:  # Avoid KDE plot for columns with a single unique value
            plt.figure(figsize=(8, 6))
            sns.kdeplot(data[column], color='#007acc', shade=True)
            plt.title(f'KDE Plot of {column}', fontsize=14)
            plt.xlabel(column)
            plt.ylabel('Density')
            plt.show()
        else:
            print(f'KDE Plot of {column} is not applicable due to insufficient unique values.')


def visualize_correlation_matrix(data):
    """Visualize the correlation matrix using a heatmap."""
    data['Gender'] = label_encoder.fit_transform(data['Gender'])
    data = data.drop(columns=['Customer_ID'])
    # Compute the correlation matrix
    corr_matrix = data.corr()

    # Set up the matplotlib figure
    plt.figure(figsize=(10, 8))

    # Draw the heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f')

    # Add a title
    plt.title('Correlation Matrix', fontsize=16)

    # Show the plot
    plt.show()


def main():
    
    file_path = 'data/customer_data.csv' 
    data = load_data(file_path)

    # plot_all_distributions(data)

   
    features = feature_def()
    
    data_scaled = preprocess_data(data, features)

    visualize_correlation_matrix(data)
    
    determine_optimal_clusters(data_scaled)

    
    n_clusters = 8  # Update based on the optimal number of clusters
    clusters = apply_kmeans_clustering(data_scaled, n_clusters)

    # Step 6: Add the cluster labels to the original data
    data = add_cluster_labels(data, clusters)

    # # Step 7: Visualize the clusters
    # visualize_clusters(data, features)
    # # Visualize clusters with different feature combinations
    # visualize_clusters_3d(data, 'Annual_Income', 'Spending_Score', 'Transaction_Amount')
    # visualize_clusters_3d(data, 'Age', 'Web_Activity', 'Time_on_Website')
    # visualize_clusters_3d(data, 'Transaction_Amount', 'Transaction_Frequency', 'Product_Views')

    # visualize_spending_score_by_cluster(data)


if __name__ == "__main__":
    main()
