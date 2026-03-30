import pandas as pd
import streamlit as st

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('HousingData.csv', encoding='windows-1252')
    except:
        df = pd.read_csv('HousingData.csv', encoding='utf-8')

    # Select only required columns (safety check)
    required_cols = [
        'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea',
        'TotalBsmtSF', '1stFlrSF', 'YearBuilt', 'FullBath',
        'TotRmsAbvGrd', 'Neighborhood', 'HouseStyle', 'SalePrice'
    ]
    
    df = df[required_cols]

    # Handle missing values (basic cleaning)
    df = df.dropna()

    # Optional: Create a new feature (example)
    df['Price_per_sqft'] = df['SalePrice'] / df['GrLivArea']

    # Optional: Convert categorical columns to category type
    df['Neighborhood'] = df['Neighborhood'].astype('category')
    df['HouseStyle'] = df['HouseStyle'].astype('category')

    return df