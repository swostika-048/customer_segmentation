import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os,sys
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util_s.utils import load_data,preprocess_data,determine_optimal_clusters,add_cluster_labels,feature_def
from util_s.data_visualization import plot_all_distributions,visualize_clusters_3d,visualize_correlation_matrix,apply_kmeans_clustering,visualize_clusters,visualize_spending_score_by_cluster

def plot_facet_grid_summary_statistics(df):
    """
    Visualizes the summary statistics of each feature separately using a Facet Grid.
    
    Args:
    df (pd.DataFrame): DataFrame containing the features.
    
    Returns:
    None
    """
    description = df.describe().T
    description['feature'] = description.index
    description = description.reset_index().melt(id_vars='feature', var_name='statistic', value_name='value')

    g = sns.FacetGrid(description, col='feature', col_wrap=4, sharey=False)
    g.map_dataframe(sns.barplot, x='statistic', y='value')
    g.set_axis_labels('Statistic', 'Value')
    g.set_titles(col_template="{col_name}")
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle('Summary Statistics for Each Feature', fontsize=16)
    plt.show()


def plot_summary_statistics(description_df):

    
    if not isinstance(description_df, pd.DataFrame):
        description_df = pd.DataFrame(description_df)

    
    description_df = description_df.reset_index()
    description_df.columns = ['Statistic'] + list(description_df.columns[1:])

    
    fig = px.imshow(description_df.set_index('Statistic').T,
                    labels={'x': 'Statistics', 'y': 'Features'},
                    color_continuous_scale='Blues')
    fig.update_layout(title='Summary Statistics Table')
    fig.show()

def plot_pair_plot(df):
    """
    Visualizes pairwise relationships in a DataFrame using a pair plot.
    
    Args:
    df (pd.DataFrame): DataFrame to visualize.
    
    Returns:
    None
    """
    sns.pairplot(df)
    plt.suptitle('Pair Plot of Features', y=1.02)
    plt.show()

def plot_summary_statistics_heatmap(description_df):
    """
    Visualizes summary statistics using a heatmap.
    
    Args:
    description_df (pd.DataFrame): DataFrame containing summary statistics.
    
    Returns:
    None
    """
    # Ensure the index is 'Statistic' and convert it if necessary
    description_df.reset_index(inplace=True)
    description_df.rename(columns={'index': 'Statistic'}, inplace=True)

    # Set 'Statistic' as index for heatmap plotting
    description_df.set_index('Statistic', inplace=True)

    # Plotting
    plt.figure(figsize=(12, 8))
    sns.heatmap(description_df, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Summary Statistics Heatmap', fontsize=16)
    plt.show()

def before_visualization(df):
    print("checking for null values")
    has_nulls=df.isnull().any().any()

    print("describing data")
    description = df.describe().T
    print(description)
    plot_summary_statistics_heatmap(description)

    
    print(f"DataFrame has null values: {has_nulls}")
    plot_all_distributions(df)
    visualize_correlation_matrix(df)
   
def after_visualization(df,features):
    visualize_clusters_3d(df, 'Age', 'Web_Activity', 'Time_on_Website')
    visualize_clusters_3d(df, 'Annual_Income', 'Spending_Score', 'Transaction_Amount')
    visualize_clusters(df, features)
    visualize_spending_score_by_cluster(df)



    pass

def main():
    file_path = 'data/customer_data.csv'  
    data = load_data(file_path)
    before_visualization(data)

    
    features = feature_def()
    data_scaled = preprocess_data(data, features)
    determine_optimal_clusters(data_scaled)
    n_clusters = 4
    clusters = apply_kmeans_clustering(data_scaled, n_clusters)
    data=add_cluster_labels(data,clusters)
    after_visualization(data,features)

main()