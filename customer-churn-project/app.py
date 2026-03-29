import streamlit as st 
import pickle
import pandas as pd

st.set_page_config(page_title="AI Churn Predictor", layout="centered")

st.markdown("## Customer Churn Prediction System")
st.markdown("Powered by Machine Learning")
st.markdown("---")
model=pickle.load(open("model.pkl","rb"))
st.write("Enter the customer details below:")

tenure=st.slider("Tenure (months)",0,72)
monthly=st.number_input("Monthly Charges")
total=st.number_input("Total Charges")

contract=st.selectbox("contract Type",["Month-to-month","One year", "Two year"])
internet=st.selectbox("internet Services",["DSL","Fiber optic","No"])
security=st.selectbox("online Security",["Yes","No"])
tech=st.selectbox("Tech Support",["Yes","No"])
payment=st.selectbox("Payment Method",[
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)"
])

if st.button("predict"):
    if monthly <= 0 or total <= 0:
        st.warning("please enter valid charges")

    else:
        data = pd.DataFrame([{
            "tenure": tenure,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
            "Contract": contract,
            "InternetService": internet,
            "OnlineSecurity": security,
            "TechSupport": tech,
            "PaymentMethod": payment
        }])
        result=model.predict(data)
        prob=model.predict_proba(data)[0][1]

        if result[0]==1:
            st.error(f"Customer is likely to Churn with a probability of {prob:.2f}")
        else:
            st.success(f"Customer is not likely to Churn with a probability of {prob:.2f}")

        st.info("Customer with month-to-month contracts are more likely to churn.")