import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from data_loader import load_data
import numpy as np
import os

st.set_page_config(layout="wide")
st.title("Module V: Regression and Price Prediction")

# ---------------- THEORY ----------------
st.header("Regression Theory")
st.markdown("""
Linear Regression is used to predict a continuous dependent variable based on one or more independent variables.
In this case, we predict **house prices** using structural features.
""")

st.latex(r"y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_n x_n")

# ---------------- DATA ----------------
df = load_data()

features = [
    'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea',
    'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd'
]

X = df[features]
y = df['SalePrice']

# ---------------- MODEL ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# ---------------- METRICS ----------------
st.header("Model Performance")

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

m1, m2, m3 = st.columns(3)
m1.metric("R² Score", f"{r2:.4f}")
m2.metric("Mean Absolute Error", f"${mae:,.2f}")
m3.metric("Training Samples", len(X_train))

st.divider()

# ---------------- VISUALIZATION ----------------
st.header("Actual vs Predicted Prices")

fig = px.scatter(
    x=y_test,
    y=y_pred,
    labels={'x': 'Actual Price', 'y': 'Predicted Price'},
    title="Actual vs Predicted House Prices"
)

fig.add_shape(
    type="line",
    x0=y_test.min(),
    y0=y_test.min(),
    x1=y_test.max(),
    y1=y_test.max(),
    line=dict(dash="dash")
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------------- USER INPUT PREDICTION ----------------
st.header("Predict House Price")

st.markdown("Enter house features below:")

col1, col2, col3 = st.columns(3)

with col1:
    overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 5)
    gr_liv_area = st.number_input("Living Area (sqft)", 500, 5000, 1500)

with col2:
    garage_cars = st.slider("Garage Capacity", 0, 4, 2)
    garage_area = st.number_input("Garage Area", 0, 1500, 500)

with col3:
    bsmt = st.number_input("Basement Area", 0, 3000, 800)
    first_flr = st.number_input("1st Floor Area", 300, 3000, 1000)

full_bath = st.slider("Full Bathrooms", 0, 4, 2)
rooms = st.slider("Total Rooms Above Ground", 2, 15, 6)

if st.button("Predict Price"):
    input_data = np.array([[ 
        overall_qual, gr_liv_area, garage_cars, garage_area,
        bsmt, first_flr, full_bath, rooms
    ]])

    prediction = model.predict(input_data)[0]

    st.success(f"Estimated House Price: ${prediction:,.2f}")

# ---------------- EXPORT ----------------
if st.button("Export Regression Results"):
    with open('bi_project_results.txt', 'a') as f:
        f.write("\n" + "="*50 + "\n")
        f.write("MODULE V: REGRESSION RESULTS\n")
        f.write(f"R2 Score: {r2:.4f}\n")
        f.write(f"MAE: {mae:.2f}\n")
        f.write("="*50 + "\n")

    st.success("Regression results saved!")