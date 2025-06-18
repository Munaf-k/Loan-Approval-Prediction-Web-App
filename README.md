# Loan Approval Prediction using Random Forest

This project presents a machine learning-based approach to predict whether a loan application should be approved. A Random Forest model is trained on a cleaned and preprocessed dataset, and deployed using Flask for user interaction via a web interface.

## Project Overview

- Objective: Predict loan approval status based on applicant data.
- Model: Random Forest Classifier
- Custom Threshold: 0.42 (to fine-tune business logic)
- Deployment: Flask Web App

## Features Used

- Gender
- Married
- Education
- ApplicantIncome
- LoanAmount
- Credit_History
- Property_Area
- Dependents
- Self_Employed
- CoapplicantIncome
- Loan_Amount_Term

## Model Pipeline

1. Data Cleaning and Preprocessing
2. Feature Engineering and Encoding
3. Train/Test Split
4. Random Forest Model Training
5. Threshold Tuning
6. Evaluation using Confusion Matrix, Classification Report, ROC Curve
7. Deployment using Flask


## Web App Preview


### Prediction Result Examples

**Example 1 – Loan Approved:**

![Loan Approved](images/Loan%20approved.png)

**Example 2 – Loan Not Approved:**

![Loan Not Approved](images/Loan%20not%20approved.png)


