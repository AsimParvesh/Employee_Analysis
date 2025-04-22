import joblib
import pandas as pd

# Load models and preprocessing objects
attrition_model = joblib.load("models/attrition_model.pkl")
promotion_model = joblib.load("models/promotion_model.pkl")

attrition_scaler = joblib.load("models/attrition_scaler.pkl")
promotion_scaler = joblib.load("models/promotion_scaler.pkl")

attrition_encoders = joblib.load("models/attrition_encoders.pkl")
promotion_encoders = joblib.load("models/promotion_encoders.pkl")

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
