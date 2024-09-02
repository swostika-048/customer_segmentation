import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

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


def plot_clusters(X_scaled, kmeans):
    """
    Plot the training data and cluster centers.

    Parameters:
    X_scaled (numpy.ndarray): The scaled training data used for clustering.
    kmeans (KMeans): The trained KMeans model.
    """
    # Plot the training data with cluster assignments
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans.labels, s=50, cmap='viridis', label='Data Points')
    
    # Plot the cluster centers
    # plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', s=200, alpha=0.75, marker='X', label='Cluster Centers')
    
    # Add title and legend
    plt.title('Cluster Centers and Data Points')
    plt.legend()
    
    # Display the plot
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
