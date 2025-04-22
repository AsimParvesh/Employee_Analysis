# Employee Analysis Project

This project uses machine learning to predict:

- **Employee Attrition** – Whether an employee is likely to leave the company.
- **Employee Promotion** – Whether an employee is likely to get promoted.

Built using:
- Python
- Streamlit
- scikit-learn
- imbalanced-learn


------------------------------------------------------------------------------------------------------------------------------------


## 📁 Project Structure

Employee Analysis Project/
│
├── data/
│   └── Employee_Attrition.csv
│
├── models/
│   ├── attrition_model.pkl
│   ├── attrition_scaler.pkl
│   ├── attrition_encoders.pkl
│   ├── promotion_model.pkl
│   ├── promotion_scaler.pkl
│   └── promotion_encoders.pkl
│
├── scripts/
│   ├── data_preprocessing.py
│   ├── train_attrition_model.py
│   └── train_promotion_model.py
│
├── utils/
│   └── predict.py
│
├── app.py
├── requirements.txt
└── README.md


------------------------------------------------------------------------------------------------------------------------------------


## 🚀 How to Run

1. Install Dependencies

```bash
pip install -r requirements.txt


2. Train the Models

python scripts/train_attrition_model.py
python scripts/train_promotion_model.py

3. Launch the App

streamlit run app.py



------------------------------------------------------------------------------------------------------------------------------------

🛠️ Features -->

Simple Streamlit interface with two tabs:

Attrition Prediction

Promotion Prediction

Models trained using:

Label encoding

Standard scaling

SMOTE for class balancing

Accurate prediction results using RandomForestClassifier


------------------------------------------------------------------------------------------------------------------------------------

📌 Notes
The dataset is located in data/Employee_Attrition.csv

Trained models and preprocessors are saved in models/

