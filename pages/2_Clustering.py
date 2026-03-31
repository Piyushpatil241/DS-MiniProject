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
st.title("Clustering and Outlier Detection")

# ---------------- THEORY ----------------
st.header("Clustering Theory")
st.markdown("#### K-Means Partitioning Method")
st.write("The algorithm minimizes the Within-Cluster Sum of Squares (WCSS):")
st.latex(r"WCSS = \sum_{i=1}^{k} \sum_{x \in C_i} \|x - \mu_i\|^2")

# ---------------- DATA ----------------
df = load_data()

# Select relevant numerical features for clustering
features = df[[
    'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea',
    'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'SalePrice'
]]

# Scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# ---------------- SIDEBAR ----------------
st.sidebar.header("Algorithm Parameters")
k_value = st.sidebar.slider("Number of Clusters (k)", 2, 6, 3)

# ---------------- MODEL ----------------
model = KMeans(n_clusters=k_value, random_state=42, n_init=10)
df['Cluster'] = model.fit_predict(scaled_features).astype(str)

# ---------------- VISUALIZATION ----------------
st.header("House Segmentation Visuals")

st.markdown("#### Living Area vs Sale Price")
fig, ax = plt.subplots(figsize=(10, 5))
sns.scatterplot(
    data=df,
    x='GrLivArea',
    y='SalePrice',
    hue='Cluster',
    palette='viridis',
    ax=ax
)
st.pyplot(fig)

st.markdown("#### 3D Feature Space Mapping")
fig_3d = px.scatter_3d(
    df,
    x='GrLivArea',
    y='SalePrice',
    z='OverallQual',
    color='Cluster'
)
st.plotly_chart(fig_3d, use_container_width=True)

# ---------------- CLUSTER INSIGHTS ----------------
st.header("Cluster Insights")
summary = df.groupby('Cluster').agg({
    'SalePrice': 'mean',
    'GrLivArea': 'mean',
    'OverallQual': 'mean',
    'Cluster': 'count'
}).rename(columns={'Cluster': 'Count'})

st.dataframe(summary, use_container_width=True)

# ---------------- OUTLIER DETECTION ----------------
st.header("Outlier Detection (Simple Method)")
q1 = df['SalePrice'].quantile(0.25)
q3 = df['SalePrice'].quantile(0.75)
iqr = q3 - q1

outliers = df[(df['SalePrice'] < q1 - 1.5 * iqr) | (df['SalePrice'] > q3 + 1.5 * iqr)]

st.write(f"Number of Outliers (based on SalePrice): {len(outliers)}")

# ---------------- EXPORT ----------------
if st.button("Export Clustering Analytics"):
    with open('project_results.txt', 'a') as f:
        f.write("\n" + "="*40 + "\n")
        f.write(f"CLUSTERING RESULTS (K={k_value})\n")
        f.write("="*40 + "\n")
        f.write(summary.to_string() + "\n")
        f.write(f"\nOutliers Detected: {len(outliers)}\n")

    st.success("Analytics appended to project_results.txt")