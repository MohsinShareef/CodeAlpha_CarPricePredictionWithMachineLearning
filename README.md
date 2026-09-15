# CodeAlpha_CarPricePredictionWithMachineLearning
- Car Price Prediction with Machine Learning.
# 1. Project Overview
## 1.1 Title

Car Price Prediction with Machine Learning, An end-to-end regression pipeline that predicts used-car selling prices and serves them through an interactive Streamlit web application.

## 1.2 Objective
Build a machine learning system that:
- Collects and preprocesses car-related features (brand, age, mileage, fuel type, etc.).
- Trains and evaluates multiple regression models.
- Selects the best-performing model based on R², MAE, and RMSE.
- Deploys the model as a user-friendly web app for real-time price prediction.

## 1.3 Domain
Auto & Used-Car Marketplace,  Applied in platforms like Cars24, CarDekho, OLX Auto, and Spinny for automated price quoting.

## 1.4 Tech Stack
- Language = Python 3.10+.
- Data Analysis = Pandas, NumPy.
- ML = Scikit-learn.
- Visualization = Matplotlib, Seaborn.
- Model save =  Joblib.
- Web App = Streamlit.
- Dataset Source = Kaggle.

# 2. Problem Statement

## 2.1 Business Context
Determining a fair selling price for a used car is difficult because value depends on many interacting factors: brand reputation, vehicle age, kilometers driven, fuel type, transmission, and ownership history. Manual valuation is slow and inconsistent.

## 2.2 Formal Definition
Given a feature vector x = (Present_Price, Driven_kms, Fuel_Type, Selling_type, Transmission, Owner, Car_Age, Brand), predict the continuous target y = Selling_Price (in lakhs INR).

# 3. Dataset Documentation

## 3.1 Source
Kaggle : https://www.kaggle.com/datasets/vijayaadithyanvg/car-price-predictionused-cars/data
