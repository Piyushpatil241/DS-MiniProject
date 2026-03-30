import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from data_loader import load_data

st.set_page_config(layout="wide")
st.title("Module III: Classification and Prediction")

# ---------------- THEORY ----------------
st.header("Bayesian Classification Theory")
st.markdown("""
Naïve Bayes is a probabilistic classifier based on Bayes' Theorem.
Here, it is used to classify houses into **Low Price** and **High Price**
based on features like area, quality, and number of rooms.
""")

st.latex(r"P(Class | Features) = \frac{P(Features | Class) \times P(Class)}{P(Features)}")

# ---------------- DATA ----------------
df = load_data()

# Create classification target (binary)
median_price = df['SalePrice'].median()
df['Price_Class'] = (df['SalePrice'] > median_price).astype(int)

# Features
X = df[[
    'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea',
    'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd'
]]
y = df['Price_Class']

# ---------------- MODEL ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred = nb_model.predict(X_test)

# ---------------- METRICS ----------------
st.header("Model Performance Metrics")
acc = accuracy_score(y_test, y_pred)

m1, m2, m3 = st.columns(3)
m1.metric("Classification Accuracy", f"{acc:.2%}")
m2.metric("Total Houses", f"{len(df)}")
m3.metric("Test Samples", f"{len(y_test)}")

st.divider()

# ---------------- VISUALS ----------------
st.header("Visual Performance Analysis")

# 📊 Price Distribution by Class
st.markdown("#### Price Class Distribution")
fig_hist = px.histogram(
    df,
    x="SalePrice",
    color="Price_Class",
    barmode="overlay",
    labels={'Price_Class': 'Price Category'}
)
st.plotly_chart(fig_hist, use_container_width=True)

# 📊 Confusion Matrix
st.markdown("#### Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)

fig_cm = ff.create_annotated_heatmap(
    cm,
    x=['Predicted Low', 'Predicted High'],
    y=['Actual Low', 'Actual High'],
    colorscale='Blues'
)
st.plotly_chart(fig_cm, use_container_width=True)

st.divider()

# ---------------- REPORT ----------------
st.header("Analytical Documentation")

report_dict = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()

report_display = report_df.drop('accuracy', errors='ignore')
report_display.index = [
    'Low Price (0)', 'High Price (1)', 'Macro Average', 'Weighted Average'
]

st.dataframe(
    report_display.style.format(precision=4),
    use_container_width=True
)

st.divider()

# ---------------- EXPORT ----------------
if st.button("Archive Classification Report"):

    report_text = classification_report(y_test, y_pred)

    with open('bi_project_results.txt', 'a') as f:
        f.write("\n" + "="*50 + "\n")
        f.write("MODULE III: NAIVE BAYES CLASSIFICATION (HOUSING)\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write("-" * 50 + "\n")
        f.write(report_text + "\n")
        f.write("="*50 + "\n")

    st.success("Analysis report successfully appended to bi_project_results.txt")