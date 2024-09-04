import streamlit as st
import pandas as pd
import pickle
from util_s.utils import feature_def

def load_model(model_file):
    with open(model_file, 'rb') as file:
        model = pickle.load(file)
    return model

def preprocess_user_input(user_data):
    features = feature_def()
    user_df = pd.DataFrame(user_data, index=[0])
    return user_df[features]

def predict_classification(model, input_data):
    processed_data = preprocess_user_input(input_data)
    prediction = model.predict(processed_data)
    return prediction[0]

def main():
    st.title("Customer Classification Prediction")

    st.markdown(
        """
        <div style="background-color: #d1ecf1; padding: 10px; border-radius: 5px; border-left: 6px solid #0c5460;">
            <h4 style="color: #0c5460;">💡 Information</h4>
            <p style="font-size: 16px; color: #0c5460;">
                Enter the customer information below to get the classification result.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

 
    default_age = 69.0
    default_income = 143437.68
    default_spending_score = 43.35
    default_transaction_amount = 487.62
    default_transaction_frequency = 28.0
    default_web_activity = 63.69
    default_product_views = 44.0
    default_time_on_website = 38.44
    default_gender = "Female"

    # Streamlit input fields with default values
    age = st.number_input("Age", min_value=0.0, max_value=100.0, value=default_age, format="%f")
    income = st.number_input("Annual Income (k$)", min_value=0.0, max_value=2000000.0, value=default_income, format="%f")
    spending_score = st.number_input("Spending Score (1-100)", min_value=0.0, max_value=100.0, value=default_spending_score, format="%f")
    transaction_amount = st.number_input("Transaction Amount", min_value=0.0, max_value=100000.0, value=default_transaction_amount, format="%f")
    transaction_frequency = st.number_input("Transaction Frequency (per month)", min_value=0.0, max_value=100.0, value=default_transaction_frequency, format="%f")
    web_activity = st.number_input("Web Activity", min_value=0.0, max_value=200.0, value=default_web_activity, format="%f")
    product_views = st.number_input("Product Views (per month)", min_value=0.0, max_value=100.0, value=default_product_views, format="%f")
    time_on_website = st.number_input("Time on Website (minutes per visit)", min_value=0.0, max_value=1000.0, value=default_time_on_website, format="%f")
    gender = st.selectbox("Gender", options=["Male", "Female"], index=0 if default_gender == "Male" else 1)

    if st.button("Predict Classification"):
        user_data = {
            'Age': age,
            'Annual_Income': income,
            'Spending_Score': spending_score,
            'Transaction_Amount': transaction_amount,
            'Transaction_Frequency': transaction_frequency,
            'Web_Activity': web_activity,
            'Product_Views': product_views,
            'Time_on_Website': time_on_website,
            'Gender': gender
        }

        model_file = 'model/logistic_regression_model.pkl'  # Path to your trained model
        model = load_model(model_file)
        predicted_class = predict_classification(model, user_data)

        # Display result
        st.markdown(f"<h2 style='color:#4B4BFF;'>The predicted class is: {predicted_class}</h2>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
