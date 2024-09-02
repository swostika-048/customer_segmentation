import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os,sys
from sklearn.preprocessing import StandardScaler
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print(f"path:{os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))}")
from utils.utils import load_data,preprocess_data,statistical_info,check_null
print("Current Python Path:")
for p in sys.path:
    print(p)



'''Function to visualize the distribution of numeric columns'''
def visualize_distributions(df):
    plt.figure(figsize=(14,6))

    # Age Distribution
    plt.subplot(1, 3, 1)
    sns.histplot(df['Age'], kde=True, color='blue')
    plt.title('Distribution of Age')

    # Annual Income Distribution
    plt.subplot(1, 3, 2)
    sns.histplot(df['Annual Income (k$)'], kde=True, color='green')
    plt.title('Distribution of Annual Income')

    # Spending Score Distribution
    plt.subplot(1, 3, 3)
    sns.histplot(df['Spending Score (1-100)'], kde=True, color='red')
    plt.title('Distribution of Spending Score')

    plt.tight_layout()
    plt.show()

'''Function to create box plots to identify outliers'''
def visualize_box_plots(df):
    plt.figure(figsize=(12,6))

    # Box plot for Age
    plt.subplot(1, 3, 1)
    sns.boxplot(y='Age', data=df)
    plt.title('Box plot of Age')

    # Box plot for Annual Income
    plt.subplot(1, 3, 2)
    sns.boxplot(y='Annual Income (k$)', data=df)
    plt.title('Box plot of Annual Income')

    # Box plot for Spending Score
    plt.subplot(1, 3, 3)
    sns.boxplot(y='Spending Score (1-100)', data=df)
    plt.title('Box plot of Spending Score')

    plt.tight_layout()
    plt.show()


def scatter_plot_income_spending(df):
    plt.figure(figsize=(12,6))
    sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Gender', data=df)
    plt.title('Annual Income vs Spending Score')
    plt.show()


def correlation_matrix(df):
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    plt.figure(figsize=(8,6))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()


def count_plot_gender(df):
    plt.figure(figsize=(6,4))
    sns.countplot(x='Gender', data=df)
    plt.title('Count of Gender')
    plt.show()


def detect_outliers(df, column_name):
    z_scores = stats.zscore(df[column_name])
    outliers = df[(z_scores > 3) | (z_scores < -3)]
    print(f"Outliers in {column_name}:")
    print(outliers)



def save_data(df, file_name='data/processed_customer_data.csv'):
    df.to_csv(file_name, index=False)
    print(f"Data saved to {file_name}")


def main():

   
    df=load_data("data/archive/Mall_Customers.csv")
    statistical_info(df)
    check_null(df)
    visualize_distributions(df)
    visualize_box_plots(df)
    scatter_plot_income_spending(df)
    correlation_matrix(df)
    count_plot_gender(df)
    detect_outliers(df, 'Age')
    detect_outliers(df, 'Annual Income (k$)')
    detect_outliers(df, 'Spending Score (1-100)')
    save_data(df)

main()