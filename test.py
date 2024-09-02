import streamlit as st
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression


def preprocess_data(df):
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    if 'CustomerID' in df.columns:
        df = df.drop(columns=['CustomerID'])
    X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Gender']]
    return X


def load_model(model_file):
    with open(model_file, 'rb') as file:
        model = pickle.load(file)
    print(f"Model loaded from {model_file}.")
    return model

def predict_with_logistic_regression(model_file, input_data):
    log_reg_model = load_model(model_file)
    processed_data = preprocess_data(input_data)
    predicted_label = log_reg_model.predict(processed_data)
    return predicted_label


def main():
    st.title("Cluster Prediction App")


    age = st.number_input("Age", min_value=0, max_value=100, value=25)
    income = st.number_input("Annual Income (k$)", min_value=0, max_value=200, value=50)
    spending_score = st.number_input("Spending Score (1-100)", min_value=0, max_value=100, value=50)
    gender = st.selectbox("Gender", options=["Male", "Female"])

    
    if st.button("Predict Cluster"):
        
        input_data = pd.DataFrame({
            'Age': [age],
            'Annual Income (k$)': [income],
            'Spending Score (1-100)': [spending_score],
            'Gender': [gender]
        })

       
        model_file = 'model/logistic_regression_model.pkl'
        predicted_label = predict_with_logistic_regression(model_file, input_data)

        
        cluster_labels = {
            0: ("Type 1", "#FF4B4B"),  # Red
            1: ("Type 2", "#4BFF4B"),  # Green
            2: ("Type 3", "#4B4BFF"),  # Blue
            3: ("Type 4", "#FFB74B"),  # Orange
            4: ("Type 5", "#FF4BFF")   # Purple
        }

        label, color = cluster_labels[predicted_label[0]]

        
        st.markdown(f"<h2 style='color:{color};'>The customer belongs to {label}.</h2>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
