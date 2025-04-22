import streamlit as st
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from utils.predict import predict_attrition, predict_promotion

# Set page title
st.set_page_config(page_title="Employee Prediction App", layout="centered")

st.title("🧠 Employee Prediction App")
st.markdown("Use this app to predict **Attrition** or **Promotion** based on employee data.")

# Define common form elements for categorical choices
gender_options = ['Male', 'Female']
business_travel_options = ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel']
marital_status_options = ['Single', 'Married', 'Divorced']
overtime_options = ['Yes', 'No']
job_role_options = [
    'Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director',
    'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources'
]

# ATTRITION FEATURES
attrition_fields = [
    'Age', 'BusinessTravel', 'DistanceFromHome', 'EnvironmentSatisfaction', 'Gender',
    'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MaritalStatus', 'MonthlyIncome',
    'NumCompaniesWorked', 'OverTime', 'WorkLifeBalance', 'YearsAtCompany'
]

# PROMOTION FEATURES
promotion_fields = [
    'Age', 'Gender', 'JobInvolvement', 'JobLevel', 'JobRole', 'MonthlyIncome', 'OverTime',
    'PercentSalaryHike', 'PerformanceRating', 'TotalWorkingYears',
    'YearsAtCompany', 'YearsInCurrentRole', 'YearsWithCurrManager'
]

tab1, tab2 = st.tabs(["👤 Attrition Prediction", "📈 Promotion Prediction"])

with tab1:
    st.header("Attrition Prediction")
    with st.form("attrition_form"):
        attrition_input = {
            "Age": st.number_input("Age", 18, 65, 30),
            "BusinessTravel": st.selectbox("Business Travel", business_travel_options),
            "DistanceFromHome": st.number_input("Distance From Home", 0, 100, 10),
            "EnvironmentSatisfaction": st.slider("Environment Satisfaction", 1, 4, 3),
            "Gender": st.selectbox("Gender", gender_options),
            "JobInvolvement": st.slider("Job Involvement", 1, 4, 3),
            "JobLevel": st.slider("Job Level", 1, 5, 2),
            "JobSatisfaction": st.slider("Job Satisfaction", 1, 4, 3),
            "MaritalStatus": st.selectbox("Marital Status", marital_status_options),
            "MonthlyIncome": st.number_input("Monthly Income", 1000, 100000, 5000),
            "NumCompaniesWorked": st.slider("Number of Companies Worked", 0, 10, 2),
            "OverTime": st.selectbox("OverTime", overtime_options),
            "WorkLifeBalance": st.slider("Work Life Balance", 1, 4, 3),
            "YearsAtCompany": st.number_input("Years at Company", 0, 40, 5)
        }
        submitted = st.form_submit_button("Predict Attrition")
        if submitted:
            result = predict_attrition(attrition_input)
            if result=="Yes":
                st.success(f"Prediction: Employee will likely to leave the company.")
            else:
                st.success(f"Prediction: Employee will Not likely to leave the company.")
                
            




with tab2:
    st.header("Promotion Prediction")
    with st.form("promotion_form"):
        promotion_input = {
            "Age": st.number_input("Age", 18, 65, 30, key="p_age"),
            "Gender": st.selectbox("Gender", gender_options, key="p_gender"),
            "JobInvolvement": st.slider("Job Involvement", 1, 4, 3, key="p_ji"),
            "JobLevel": st.slider("Job Level", 1, 5, 2, key="p_jl"),
            "JobRole": st.selectbox("Job Role", job_role_options, key="p_jr"),
            "MonthlyIncome": st.number_input("Monthly Income", 1000, 100000, 5000, key="p_income"),
            "OverTime": st.selectbox("OverTime", overtime_options, key="p_ot"),
            "PercentSalaryHike": st.slider("Percent Salary Hike", 0, 100, 15, key="p_hike"),
            "PerformanceRating": st.slider("Performance Rating", 1, 4, 3, key="p_perf"),
            "TotalWorkingYears": st.slider("Total Working Years", 0, 40, 10, key="p_tw"),
            "YearsAtCompany": st.slider("Years at Company", 0, 40, 5, key="p_yc"),
            "YearsInCurrentRole": st.slider("Years in Current Role", 0, 20, 3, key="p_yir"),
            "YearsWithCurrManager": st.slider("Years with Current Manager", 0, 20, 2, key="p_ywm")
        }
        submitted = st.form_submit_button("Predict Promotion")
        if submitted:
            result = predict_promotion(promotion_input)
            if result=="Yes":
                st.success(f"Prediction: Employee is likely to be promoted.")
            else:
                st.success(f"Prediction: Employee is Not likely to be promoted.")
            