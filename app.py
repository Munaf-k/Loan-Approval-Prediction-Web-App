from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# Load the trained model and set the threshold
model = pickle.load(open('rf_model.pkl', 'rb'))
threshold = 0.42

@app.route('/', methods=['GET'])
def Home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        # Raw input from form
        Gender = request.form['Gender']  # 'Male' or 'Female'
        Married = request.form['Married']  # 'Yes' or 'No'
        Dependents = request.form['Dependents']  # '0', '1', '2', '3+'
        Education = request.form['Education']  # 'Graduate' or 'Not Graduate'
        Self_Employed = request.form['Self_Employed']  # 'Yes' or 'No'
        ApplicantIncome = int(request.form['ApplicantIncome'])
        CoapplicantIncome = float(request.form['CoapplicantIncome'])
        LoanAmount = float(request.form['LoanAmount'])
        Loan_Amount_Term = float(request.form['Loan_Amount_Term'])
        Credit_History = int(request.form['Credit_History'])
        Property_Area = request.form['Property_Area']  # 'Urban', 'Rural', 'Semiurban'

        # One-hot encoding manually
        input_data = {
            'ApplicantIncome': ApplicantIncome,
            'CoapplicantIncome': CoapplicantIncome,
            'LoanAmount': LoanAmount,
            'Loan_Amount_Term': Loan_Amount_Term,
            'Credit_History': Credit_History,
            'Gender_Male': Gender == 'Male',
            'Married_Yes': Married == 'Yes',
            'Dependents_1': Dependents == '1',
            'Dependents_2': Dependents == '2',
            'Dependents_3+': Dependents == '3+',
            'Education_Not Graduate': Education == 'Not Graduate',
            'Self_Employed_Yes': Self_Employed == 'Yes',
            'Property_Area_Semiurban': Property_Area == 'Semiurban',
            'Property_Area_Urban': Property_Area == 'Urban'
        }

        # Convert to DataFrame
        final_input = pd.DataFrame([input_data])

        # Predict probability
        proba = model.predict_proba(final_input)[:, 1][0]

        # Apply threshold
        prediction_text = "🎉 Congratulations! Your loan is likely to be approved!" if proba >= threshold else "Unfortunately, your loan may not be approved at this time."

        # prediction as 'prediction' to match HTML template
        return render_template("index.html", prediction=prediction_text)

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
