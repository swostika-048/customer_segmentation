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

if __name__ == "__main__":
    input_data = pd.DataFrame({
        'Age': [58],
        'Annual Income (k$)': [20],
        'Spending Score (1-100)': [15],
        'Gender': ['Female']
    })

    model_file = 'model/logistic_regression_model.pkl'

  
    predicted_label = predict_with_logistic_regression(
        model_file=model_file,
        input_data=input_data
    )

    print(f"Predicted label: {predicted_label[0]}")
