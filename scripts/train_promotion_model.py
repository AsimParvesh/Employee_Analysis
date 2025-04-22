import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load data
df = pd.read_csv("data/Employee_Attrition.csv")

# Finalized features for promotion
promotion_features = [
    'Age', 'Gender', 'JobInvolvement', 'JobLevel', 'JobRole',
    'MonthlyIncome', 'OverTime', 'PercentSalaryHike', 'PerformanceRating',
    'TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsWithCurrManager'
]

# Create binary target: Promoted (1 if YearsSinceLastPromotion == 0, else 0)
df['Promoted'] = df['YearsSinceLastPromotion'].apply(lambda x: 1 if x == 0 else 0)

# Encode categorical columns
df = df.copy()
encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Scale numerical features
scaler = StandardScaler()
num_cols = [col for col in promotion_features if df[col].dtype in ['int64', 'float64']]
df[num_cols] = scaler.fit_transform(df[num_cols])

# Prepare data
X = df[promotion_features]
y = df['Promoted']

# Balance using SMOTE
X_res, y_res = SMOTE(random_state=42).fit_resample(X, y)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("\n📊 Promotion Classification Report:\n")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "models/promotion_model.pkl")
joblib.dump(encoders, "models/promotion_encoders.pkl")
joblib.dump(scaler, "models/promotion_scaler.pkl")
print("\n✅ Promotion model saved.")
