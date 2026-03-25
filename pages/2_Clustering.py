import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from data_loader import load_data
import os

st.set_page_config(layout="wide")
st.title("Module IV: Clustering and Outlier Detection")

st.header("Clustering Theory")
st.markdown("#### K-Means Partitioning Method")
st.write("The algorithm minimizes the Within-Cluster Sum of Squares (WCSS):")
st.latex(r"WCSS = \sum_{i=1}^{k} \sum_{x \in C_i} \|x - \mu_i\|^2")



df = load_data()
cust_data = df.groupby('Customer ID').agg({
    'Sales': 'sum', 'Profit': 'sum', 'Quantity': 'sum'
}).reset_index()

scaler = StandardScaler()
scaled_features = scaler.fit_transform(cust_data[['Sales', 'Profit', 'Quantity']])

st.sidebar.header("Algorithm Parameters")
k_value = st.sidebar.slider("Number of Clusters (k)", 2, 6, 3)

model = KMeans(n_clusters=k_value, random_state=42, n_init=10)
cust_data['Cluster'] = model.fit_predict(scaled_features).astype(str)

st.header("Customer Segmentation Visuals")
st.markdown("#### 2D Sales vs Profit Distribution (Seaborn)")
fig, ax = plt.subplots(figsize=(10, 5))
sns.scatterplot(data=cust_data, x='Sales', y='Profit', hue='Cluster', palette='viridis', ax=ax)
st.pyplot(fig)

st.markdown("#### 3D Feature Space Mapping (Plotly)")
fig_3d = px.scatter_3d(cust_data, x='Sales', y='Profit', z='Quantity', color='Cluster')
st.plotly_chart(fig_3d, use_container_width=True)

if st.button("Export Clustering Analytics"):
    summary = cust_data.groupby('Cluster').agg({'Sales': 'mean', 'Profit': 'mean', 'Customer ID': 'count'})
    with open('bi_project_results.txt', 'a') as f:
        f.write("\n" + "="*40 + "\n")
        f.write(f"CLUSTERING RESULTS (K={k_value})\n")
        f.write("="*40 + "\n")
        f.write(summary.to_string() + "\n")
    st.success("Analytics appended to bi_project_results.txt")