import streamlit as st

st.set_page_config(page_title="Business Intelligence System", layout="wide")

st.title("Business Intelligence Decision Support System")
st.markdown("### Business Intelligence Lab Project")

st.markdown("""
This application serves as a comprehensive Business Intelligence tool for analyzing 
retail performance using the Superstore dataset. It covers the end-to-end KDD 
(Knowledge Discovery in Databases) process.
""")

st.header("Data Warehouse and Dimensional Modeling")
st.markdown("""
A Star Schema has been designed for analytical processing of this project. 
It separates business process data into facts and dimensions.
""")

st.markdown("#### Schema Architecture")
st.markdown("""
* **Fact Table**: Sales_Fact (Sales, Profit, Quantity, Discount)
* **Dimension: Product**: Category, Sub-Category, Product Name
* **Dimension: Customer**: Customer ID, Name, Segment
* **Dimension: Location**: Region, State, City, Postal Code
* **Dimension: Time**: Order Date, Ship Date, Quarter, Year
""")



st.divider()
st.header("Project Scope")
st.markdown("""
1. **Data Exploration**: Statistical description and visual analysis.
2. **Preprocessing**: Data cleaning and transformation.
3. **Clustering**: Customer segmentation using K-Means.
4. **Classification**: Profitability prediction using Naïve Bayes.
5. **Association Mining**: Market Basket Analysis using Apriori.
""")