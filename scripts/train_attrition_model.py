import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load data
df = pd.read_csv("data/Employee_Attrition.csv")

# Finalized features for attrition
attrition_features = [
    'Age', 'BusinessTravel', 'DistanceFromHome', 'EnvironmentSatisfaction', 'Gender',
    'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MaritalStatus', 'MonthlyIncome',
    'NumCompaniesWorked', 'OverTime', 'WorkLifeBalance', 'YearsAtCompany'
]
target_col = 'Attrition'

# Encode categorical columns
df = df.copy()
encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Scale numerical features (excluding target)
scaler = StandardScaler()
num_cols = [col for col in attrition_features if df[col].dtype in ['int64', 'float64']]
df[num_cols] = scaler.fit_transform(df[num_cols])

# Prepare data
X = df[attrition_features]
y = df[target_col]

# Balance using SMOTE
X_res, y_res = SMOTE(random_state=42).fit_resample(X, y)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("\n📊 Attrition Classification Report:\n")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "models/attrition_model.pkl")
joblib.dump(encoders, "models/attrition_encoders.pkl")
joblib.dump(scaler, "models/attrition_scaler.pkl")
print("\n✅ Attrition model saved.")
