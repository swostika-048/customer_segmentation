import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from util_s.utils import load_data,preprocess_data,feature_def,load_model
from src.k_mean import KMeansScratch
import pandas as pd

def preprocess_user_input(user_data):
    
    features = feature_def()
    # user_data=preprocess_data(user_data,features)
    user_df = pd.DataFrame(user_data, index=[0])
    return user_df[features]

def predict_cluster(user_data, model_file='model/logistic_regression_model.pkl'):
    model = load_model(model_file)
    processed_data = preprocess_user_input(user_data)
    prediction = model.predict(processed_data)
    return prediction[0]  

def predict_classifcation():
    # age = int(input("Enter Age: "))
    # income = float(input("Enter Annual Income (k$): "))
    # spending_score = int(input("Enter Spending Score (1-100): "))
    # transaction_amount = float(input("Enter Transaction Amount: "))
    # transaction_frequency = int(input("Enter Transaction Frequency: "))
    # web_activity = int(input("Enter Web Activity: "))
    # product_views = int(input("Enter Product Views: "))
    # time_on_website = float(input("Enter Time on Website: "))

    user_data=  {
    'Age': 69,
    'Gender': 'Male',
    'Annual_Income': 143437.681075493,
    'Spending_Score': 43.3495402398504,
    'Transaction_Amount': 487.615632045607,
    'Transaction_Frequency': 28,
    'Web_Activity': 63.6902317454743,
    'Product_Views': 44,
    'Time_on_Website': 38.4437161327554
}
    cluster = predict_cluster(user_data)
    return cluster

if __name__ == "__main__":
    model_kmean = 'model/kmeans_model.pkl'
    model_classifcation='model/logistic_regression_model.pkl'
    cluster=predict_classifcation()
    print(cluster)
    # df=load_data('data/customer_data.csv')
    # features=feature_def()

    # processed_data = preprocess_data(df,features)
    # kmeans_scratch=load_model(model_kmean)

    # cluster = kmeans_scratch.predict(processed_data)
    # print(f"cluster:{cluster}")
    # print(f"The customer belongs to cluster {cluster[0]}.")







