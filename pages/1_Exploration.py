import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
from data_loader import load_data
import os

st.set_page_config(layout="wide")
st.title("Data Exploration and Preprocessing")

df = load_data()

# ---------------- METRICS ----------------
st.header("Housing Dataset Metrics")
m1, m2, m3, m4 = st.columns(4)

m1.metric("Average Sale Price", f"${df['SalePrice'].mean():,.2f}")
m2.metric("Maximum Sale Price", f"${df['SalePrice'].max():,.2f}")
m3.metric("Average Living Area", f"{df['GrLivArea'].mean():,.0f} sqft")
m4.metric("Total Houses", len(df))

st.divider()

# ---------------- THEORY ----------------
st.header("Statistical Analysis and Preprocessing")

st.markdown("#### Normalization Theory")
st.write("Normalization ensures features like area and price are on the same scale, improving model performance.")
st.latex(r"x_{normalized} = \frac{x - \min(x)}{\max(x) - \min(x)}")

# ---------------- STATS ----------------
st.markdown("#### Descriptive Statistics")
st.dataframe(df.describe(), use_container_width=True)

st.divider()

# ---------------- VISUALIZATION ----------------
st.header("Visual Exploration")

# Sale Price Distribution
st.markdown("#### Sale Price Distribution")
fig1, ax1 = plt.subplots()
df['SalePrice'].hist(bins=30, ax=ax1)
ax1.set_xlabel("Sale Price")
ax1.set_ylabel("Frequency")
st.pyplot(fig1)

# Neighborhood vs Price
st.markdown("#### Neighborhood vs Sale Price")
fig2 = px.box(df, x='Neighborhood', y='SalePrice')
st.plotly_chart(fig2, use_container_width=True)

# Living Area vs Price
st.markdown("#### Living Area vs Sale Price")
fig3 = px.scatter(df, x='GrLivArea', y='SalePrice',
                  color='OverallQual',
                  title="Living Area vs Price")
st.plotly_chart(fig3, use_container_width=True)

# Correlation Heatmap
st.markdown("#### Correlation Heatmap")
corr = df.corr(numeric_only=True)

fig4, ax4 = plt.subplots(figsize=(8, 5))
cax = ax4.matshow(corr)
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
fig4.colorbar(cax)
st.pyplot(fig4)

# ---------------- EXPORT ----------------
if st.button("Export Exploration Summary"):
    summary = df.describe().to_string()
    
    with open('project_results.txt', 'a') as f:
        f.write("\n" + "="*40 + "\n")
        f.write("EXPLORATION RESULTS (HOUSING DATA)\n")
        f.write("="*40 + "\n")
        f.write(summary + "\n")

    st.success("Summary appended to project_results.txt")