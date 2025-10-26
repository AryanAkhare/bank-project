import streamlit as st
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")


st.set_page_config(page_title="Bank Term Deposit Prediction (Naïve Bayes)", layout="centered")
st.title("🏦 Bank Term Deposit Prediction App (Naïve Bayes)")

# Load saved model & preprocessor
preprocessor_file = "gnb_preprocessor.pkl"
model_file = "tuned_gnb_model.pkl"
columns_file = "feature_columns_gnb.pkl"

preprocessor = joblib.load(preprocessor_file)
model = joblib.load(model_file)
feature_columns = joblib.load(columns_file)

# Attribute options
jobs = ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management',
        'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown']
marital = ['divorced', 'married', 'single', 'unknown']
education = ['basic.4y','basic.6y','basic.9y','high.school','illiterate',
             'professional.course','university.degree','unknown']
binary_unknown = ['no','yes','unknown']
contact_type = ['cellular','telephone']
months = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
days = ['mon','tue','wed','thu','fri']
poutcome = ['failure','nonexistent','success']

# Input form
st.subheader("👤 Client Info")
age = st.number_input("Age", 18,90,30)
job = st.selectbox("Job", jobs)
marital_status = st.selectbox("Marital Status", marital)
education_status = st.selectbox("Education", education)
default = st.selectbox("Has Credit in Default?", binary_unknown)
housing = st.selectbox("Housing Loan?", binary_unknown)
loan = st.selectbox("Personal Loan?", binary_unknown)

st.subheader("📞 Contact Info")
contact = st.selectbox("Contact Type", contact_type)
month = st.selectbox("Last Contact Month", months)
day_of_week = st.selectbox("Last Contact Day of Week", days)

st.subheader("📊 Campaign & Economic Info")
campaign = st.number_input("Number of Contacts (this campaign)", 1)
pdays = st.number_input("Days Since Last Contact (999=never)", 999)
previous = st.number_input("Previous Contacts",0)
poutcome_sel = st.selectbox("Previous Campaign Outcome", poutcome)
emp_var_rate = st.number_input("Employment Variation Rate",0.5)
cons_price_idx = st.number_input("Consumer Price Index",93.0)
cons_conf_idx = st.number_input("Consumer Confidence Index",-35.0)
euribor3m = st.number_input("Euribor 3 Month Rate",2.0)
nr_employed = st.number_input("Number of Employees",5000.0)

# Prepare dataframe
input_data = pd.DataFrame([{
    "age": age, "job": job, "marital": marital_status, "education": education_status,
    "default": default, "housing": housing, "loan": loan, "contact": contact,
    "month": month, "day_of_week": day_of_week, "campaign": campaign,
    "pdays": pdays, "previous": previous, "poutcome": poutcome_sel,
    "emp.var.rate": emp_var_rate, "cons.price.idx": cons_price_idx,
    "cons.conf.idx": cons_conf_idx, "euribor3m": euribor3m, "nr.employed": nr_employed
}])

# Predict button
if st.button("🔍 Predict"):
    try:
        input_transformed = preprocessor.transform(input_data)
        pred = model.predict(input_transformed)[0]
        proba = model.predict_proba(input_transformed)[0][1]

        if pred==1:
            st.success("✅ The client is **likely** to subscribe to a term deposit.")
        else:
            st.error("⚠️ The client is **NOT likely** to subscribe.")

        
    except Exception as e:
        st.error(f"Prediction error: {e}")

# Button for guaranteed likely subscribe
if st.button("🎯 Sample Likely to Subscribe"):
    yes_sample = pd.DataFrame([{
        "age": 35, "job": "management", "marital": "married", "education": "university.degree",
        "default": "no", "housing": "no", "loan": "no", "contact": "cellular",
        "month":"may", "day_of_week":"mon", "campaign":1, "pdays":999, "previous":0,
        "poutcome":"success", "emp.var.rate":1.0, "cons.price.idx":93.5, "cons.conf.idx":-30,
        "euribor3m":2.5, "nr.employed":5100.0
    }])
    yes_transformed = preprocessor.transform(yes_sample)
    pred = model.predict(yes_transformed)[0]
    proba = model.predict_proba(yes_transformed)[0][1]

    if pred==1:
        st.success("✅ The client is **likely** to subscribe to a term deposit.")
    else:
        st.error("⚠️ The client is **NOT likely** to subscribe.")

    
    st.subheader("Why this sample is likely to subscribe:")
    st.markdown("""
    - **Job:** Management → higher income & stability  
    - **Education:** University degree → more financial literacy  
    - **No loans / default:** financially reliable  
    - **Previous campaign outcome:** Success  
    - **Economic indicators favorable:** High employment, positive euribor  
    - **Low campaign number & pdays=999:** first contact / fresh lead
    """)
