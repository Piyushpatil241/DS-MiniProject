import pandas as pd
import streamlit as st

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('Superstore.csv', encoding='windows-1252')
    except:
        df = pd.read_csv('Superstore.csv', encoding='utf-8')

    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df['Ship Date'] = pd.to_datetime(df['Ship Date'])

    df['Is_Profitable'] = (df['Profit'] > 0).astype(int)
    
    return df