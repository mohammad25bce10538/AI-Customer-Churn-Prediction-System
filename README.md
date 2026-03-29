AI Customer Churn Prediction System


Overview-

Here's a hands-on project where I built a complete machine learning app to figure out if a customer might leave a service or stick around.
It runs on Python, using scikit-learn and Streamlit. The web interface lets you play with predictions in real time.


Problem Statement-

When customers leave, businesses lose money—and it’s usually cheaper to keep a current customer than to go out and find a new one.
This project digs into customer data, predicts who’s likely to churn, and gives businesses a heads-up so they can act before it’s too late.


Machine Learning Approach- 

- Algorithm: Random Forest Classifier
- For categorical features, I used OneHot Encoding
- Everything runs through a pipeline that connects preprocessing and model training
- Dealt with class imbalance by using class_weight="balanced"


Dataset

- Telco Customer Churn Dataset
    - Includes info like:
        - Customer demographics
        - Account details
        - Services they’ve signed up for
        - Billing information


Tech Stack-

- Python
- Pandas & NumPy
- Scikit-learn
- Streamlit


Project Structure-

customer-churn-project/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── model.py          # Trains the model
├── app.py            # Runs the Streamlit app
├── model.pkl         # Saved model file
├── README.md         # Project documentation


Installation and Setup-

1. Clone the repo:
git clone https://github.com/your-username/customer-churn-project.git
cd customer-churn-project


2. Install the requirements:

pip install pandas numpy scikit-learn streamlit


How to Run the Project-

Step 1: Train the model:
python model.py
Step 2: Start the web app:
streamlit run app.py


Input Features-

The model takes these key features:
- Tenure
- Monthly Charges
- Total Charges
- Contract Type
- Internet Service
- Online Security
- Tech Support
- Payment Method


Output-

- Prediction: Churn or Not Churn
- Plus, a probability score showing how confident the model feels


Model Performance-

- Accuracy sits around 78%–82%
- Parameters are tuned to balance predictions between classes


Key Highlights-

- End-to-end workflow, from preprocessing to prediction
- Clean structure using ColumnTransformer
- User-friendly web interface—make predictions on the fly
- Real-time results


Future Improvements-

- Try advanced models like XGBoost
- Pull in more features for better predictions
- Set up cloud deployment so it’s always available
- Add dashboards for visualizing trends


Conclusion-

This project brings machine learning to life by tackling a real business challenge—figuring out which customers are likely to leave. You get data prep, model training, and an interactive tool, all wrapped in one package.
