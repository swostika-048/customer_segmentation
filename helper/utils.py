import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def load_data(file_path):
    """
    Load the dataset from a CSV file.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: The loaded DataFrame.
    """
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    """
    Preprocess the data by encoding categorical variables and scaling the features.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    np.ndarray: The preprocessed and scaled feature matrix.
    """
    # Encode categorical variables (Gender: Male=0, Female=1)
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    if 'CustomerID' in df.columns:
        df = df.drop(columns=['CustomerID'])
    # Select features for clustering
    X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Gender']]
    
    # Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save the scaler to a file

        
    return X_scaled



def statistical_info(df):
    print("Display the first few rows")
    print(df.head())

    print("Summary of the dataset")
    print(df.info())

    print("Statistical summary of numerical columns")
    print(df.describe())

def check_null(df):
    # Check for missing values
    print(df.isnull().sum())



# def plot_clusters_and_centroids(X, Y, kmeans, title='Customer Groups', xlabel='Annual Income', ylabel='Spending Score'):
#     plt.figure(figsize=(8, 8))

#     # Define colors and labels for the clusters
#     colors = ['green', 'red', 'yellow', 'violet', 'blue']
#     labels = ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5']

#     # Plot each cluster
#     for i in range(len(colors)):
#         plt.scatter(X[Y == i, 0], X[Y == i, 1], s=50, c=colors[i], label=labels[i])

#     # Plot the centroids
#     # plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='cyan', label='Centroids')

#     # Add titles and labels
#     plt.title(title)
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.legend()

#     # Display the plot
#     plt.show()

def plot_clusters_and_centroids(X, labels, kmeans, title='Customer Groups', xlabel='Annual Income', ylabel='Spending Score', zlabel='Age'):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Define colors and labels for the clusters
    colors = ['green', 'red', 'yellow', 'violet', 'blue']
    cluster_labels = [f'Cluster {i+1}' for i in range(len(colors))]

    # Plot each cluster
    for i in range(len(colors)):
        ax.scatter(X[labels == i, 0], X[labels == i, 1], X[labels == i, 2], s=50, c=colors[i], label=cluster_labels[i])

    # # Plot the centroids
    # ax.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], kmeans.centroids[:, 2], s=100, c='cyan', label='Centroids', marker='X')

    # Add titles and labels
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.legend()

    # Show the plot
    plt.show()

# if __name__ == "__main__":
#     # Example usage
#     file_path = 'customer_data.csv'
    
#     # Load the data
#     df = load_data(file_path)
    
#     # Preprocess the data
#     X_scaled = preprocess_data(df)
    
#     # Now X_scaled can be used for clustering
#     print(X_scaled[:5])  # Print the first 5 rows of the scaled data as a sanity check
