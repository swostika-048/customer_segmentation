import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle
import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util_s.utils import feature_def

def train_and_save_logistic_regression(data_file='output/customer_data_with_clusters.csv',model_file='model/logistic_regression_model.pkl'):

    
    df = pd.read_csv(data_file)
    features=feature_def()
    X = df[features]
    y=df['Cluster']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    log_reg_model = LogisticRegression()
    log_reg_model.fit(X_train, y_train)

    with open(model_file, 'wb') as file:
        pickle.dump(log_reg_model, file)


    print(f"Model saved to {model_file}.")
  

train_and_save_logistic_regression(
    data_file='output/customer_data_with_clusters.csv',
    model_file='model/logistic_regression_model.pkl'
    
)
