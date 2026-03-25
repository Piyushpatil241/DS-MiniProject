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

st.header("Bayesian Classification Theory")
st.markdown("""
The Naïve Bayes classifier is a probabilistic model based on Bayes' Theorem. 
It predicts the probability of a transaction being profitable based on historical data patterns 
of Sales, Quantity, and Discounts.
""")

st.latex(r"P(Class | Features) = \frac{P(Features | Class) \times P(Class)}{P(Features)}")


df = load_data()

X = df[['Sales', 'Quantity', 'Discount']]
y = df['Is_Profitable']

# Model Training and Evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred = nb_model.predict(X_test)

st.header("Model Performance Metrics")
acc = accuracy_score(y_test, y_pred)

m1, m2, m3 = st.columns(3)
m1.metric("Classification Accuracy", f"{acc:.2%}")
m2.metric("Total Sample Size", f"{len(df)} Rows")
m3.metric("Test Population", f"{len(y_test)} Rows")

st.divider()

st.header("Visual Performance Analysis")

st.markdown("#### Discount Threshold Impact")
st.write("This distribution illustrates how different discount levels influence the final profitability class (0 = Loss, 1 = Profit).")
fig_hist = px.histogram(df, x="Discount", color="Is_Profitable", barmode="group",
                        color_discrete_map={0: "#e74c3c", 1: "#2ecc71"},
                        labels={'Is_Profitable': 'Profit Class'})
st.plotly_chart(fig_hist, use_container_width=True)

st.markdown("#### Confusion Matrix (Classification Error Measures)")
st.write("The matrix evaluates the accuracy of the classifier by comparing actual vs. predicted values.")
cm = confusion_matrix(y_test, y_pred)
fig_cm = ff.create_annotated_heatmap(cm, x=['Predicted Loss', 'Predicted Profit'], 
                                     y=['Actual Loss', 'Actual Profit'], 
                                     colorscale='Blues')
st.plotly_chart(fig_cm, use_container_width=True)

st.divider()

st.header("Analytical Documentation")
st.markdown("#### Detailed Performance Report")
st.write("""
The table below provides a granular breakdown of the model's performance. 
- **Precision**: The ability of the classifier not to label as positive a sample that is negative.
- **Recall**: The ability of the classifier to find all the positive samples.
- **F1-Score**: The weighted average of Precision and Recall.
""")

report_dict = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()

report_display = report_df.drop('accuracy', errors='ignore')
report_display.index = [
    'Unprofitable (0)', 'Profitable (1)', 'Macro Average', 'Weighted Average'
]

st.dataframe(
    report_display.style.format(precision=4),
    use_container_width=True
)

st.divider()

if st.button("Archive Classification Report"):

    report_text = classification_report(y_test, y_pred)
    with open('bi_project_results.txt', 'a') as f:
        f.write("\n" + "="*50 + "\n")
        f.write(f"MODULE III: NAIVE BAYES CLASSIFICATION REPORT\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write("-" * 50 + "\n")
        f.write(f"{str(report_text)}\n")
        f.write("="*50 + "\n")
    st.success("Analysis report successfully appended to bi_project_results.txt")