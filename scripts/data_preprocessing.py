# data_preprocessing.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_features(df, features, label_encoders=None, scaler=None, is_train=True):
    df = df.copy()
    df = df[features].copy()
    
    if label_encoders is None:
        label_encoders = {}

    for col in df.select_dtypes(include='object').columns:
        if is_train:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
        else:
            le = label_encoders.get(col)
            if le:
                df[col] = le.transform(df[col])

    if is_train:
        scaler = StandardScaler()
        df[df.columns] = scaler.fit_transform(df)
    else:
        df[df.columns] = scaler.transform(df)

    return df, label_encoders, scaler
