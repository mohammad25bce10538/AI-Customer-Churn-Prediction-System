import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder 
from sklearn.metrics import accuracy_score

df=pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
print(df.head())

df.drop("customerID",axis=1,inplace=True)
df["TotalCharges"]=pd.to_numeric(df["TotalCharges"],errors="coerce")
df.dropna(inplace=True)

x=df[[
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Contract",
    "InternetService",
    "OnlineSecurity",
    "TechSupport",
    "PaymentMethod"
]]
y=df["Churn"].map({"Yes":1,"No":0})

cat_cols=["Contract","InternetService","OnlineSecurity","TechSupport","PaymentMethod"]
num_cols=["tenure","MonthlyCharges","TotalCharges"]

preprocessor=ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"),cat_cols),
    ],
    remainder="passthrough"
)
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(n_estimators=500, random_state=42))
])

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

pipeline.fit(x_train,y_train)

y_pred = pipeline.predict(x_test)
print("Accuracy:",accuracy_score(y_test, y_pred))
pickle.dump(pipeline, open("model.pkl", "wb"))
print("Model saved successfully!")