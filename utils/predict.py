import pandas as pd
import os
import joblib

# Get the absolute path to the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Load attrition model files
attrition_model = joblib.load(os.path.join(MODEL_DIR, "attrition_model.pkl"))
attrition_scaler = joblib.load(os.path.join(MODEL_DIR, "attrition_scaler.pkl"))
attrition_encoders = joblib.load(os.path.join(MODEL_DIR, "attrition_encoders.pkl"))

# Load promotion model files
promotion_model = joblib.load(os.path.join(MODEL_DIR, "promotion_model.pkl"))
promotion_scaler = joblib.load(os.path.join(MODEL_DIR, "promotion_scaler.pkl"))
promotion_encoders = joblib.load(os.path.join(MODEL_DIR, "promotion_encoders.pkl"))


def preprocess_input(input_data, encoders, scaler):
    df = pd.DataFrame([input_data])
    
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col])
    
    df_scaled = scaler.transform(df)
    return df_scaled

def predict_attrition(input_dict):
    processed = preprocess_input(input_dict, attrition_encoders, attrition_scaler)
    prediction = attrition_model.predict(processed)[0]
    return "Yes" if prediction == 1 else "No"

def predict_promotion(input_dict):
    processed = preprocess_input(input_dict, promotion_encoders, promotion_scaler)
    prediction = promotion_model.predict(processed)[0]
    return "Yes" if prediction == 1 else "No"
