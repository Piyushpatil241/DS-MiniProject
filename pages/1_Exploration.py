import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
from data_loader import load_data
import os

st.set_page_config(layout="wide")
st.title("Module II: Data Exploration and Preprocessing")

df = load_data()

st.header("Performance Metrics")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Sales Volume", f"${df['Sales'].sum():,.2f}")
m2.metric("Total Net Profit", f"${df['Profit'].sum():,.2f}")
m3.metric("Average Discount Rate", f"{df['Discount'].mean():.2%}")
m4.metric("Total Transaction Count", len(df))

st.divider()

st.header("Statistical Analysis and Preprocessing")
st.markdown("#### Normalization Theory")
st.write("Data is normalized to ensure that large-scale features do not bias the mining algorithms.")
st.latex(r"x_{normalized} = \frac{x - \min(x)}{\max(x) - \min(x)}")

st.markdown("#### Descriptive Statistics")
st.dataframe(df.describe(), use_container_width=True)

st.divider()

st.header("Visual Exploration")
st.markdown("#### Regional Sales Distribution (Static 2D)")
fig, ax = plt.subplots(figsize=(10, 4))
df.groupby('Region')['Sales'].sum().plot(kind='bar', ax=ax, color='#2c3e50')
ax.set_ylabel("Total Sales ($)")
st.pyplot(fig)

st.markdown("#### Hierarchical Sales Analysis (Interactive Sunburst)")
fig_sun = px.sunburst(df, path=['Region', 'Category', 'Sub-Category'], 
                      values='Sales', color='Profit',
                      color_continuous_scale='RdYlGn')
st.plotly_chart(fig_sun, use_container_width=True)

if st.button("Export Exploration Summary"):
    summary = df.groupby('Region')['Sales'].sum().to_string()
    with open('bi_project_results.txt', 'a') as f:
        f.write("\n" + "="*40 + "\n")
        f.write("EXPLORATION RESULTS\n")
        f.write("="*40 + "\n")
        f.write(f"Summary Stats:\n{df.describe().to_string()}\n")
        f.write(f"Regional Sales:\n{summary}\n")
    st.success("Summary appended to bi_project_results.txt")