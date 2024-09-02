import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

def train_and_save_logistic_regression(data_file='/home/swostika/Documents/datascience/ds_projects/market_basket/output/clustered_data.csv',model_file='model/logistic_regression_model.pkl'):

    
    df = pd.read_csv(data_file)
    X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Gender']]
    y = df['Cluster']  



   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    log_reg_model = LogisticRegression()
    log_reg_model.fit(X_train, y_train)

    # Save the trained Logistic Regression model
    with open(model_file, 'wb') as file:
        pickle.dump(log_reg_model, file)


    print(f"Model saved to {model_file}.")
  

# Example usage
train_and_save_logistic_regression(
    data_file='output/clustered_data.csv',
    model_file='model/logistic_regression_model.pkl'
    
)
