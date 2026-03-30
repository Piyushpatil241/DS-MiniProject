import streamlit as st

st.set_page_config(page_title="House Prices Business Intelligence System", layout="wide")

st.title("House Prices Business Intelligence Decision Support System")
st.markdown("### House Prices Business Intelligence Lab Project")

st.markdown("""
This application serves as a comprehensive Business Intelligence tool for analyzing 
house prices data using Housing Dataset. It covers the end-to-end KDD 
(Knowledge Discovery in Databases) process.
""")

st.header("Data Warehouse and Dimensional Modeling")
st.markdown("""
A Star Schema has been designed for analytical processing of this project. 
It separates business process data into facts and dimensions.
""")

st.markdown("#### Schema Architecture")

st.image("StarSchema.png", caption="Star Schema for House Prices Analysis")
st.markdown("""
* **Fact Table**: Fact_HouseSales (house_id, quality_id, garage_id, building_id, area_id, location_id, sales_price)
* **Dimension: Dim_Building**: building_id, year_built, full_bath, total_rooms
* **Dimension: Dim_Area**: area_id, gr_liv_area, total_bsmt_sf, first_flr_sf
* **Dimension: Dim_Location**: Region, State, City, Postal Code
* **Dimension: Dim_Quality**: quality_id, overall_qual
* **Dimension: Dim_Garage**: garage_id, garage_cars, garage_area
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