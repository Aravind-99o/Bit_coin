import streamlit as st
import numpy as np
import pandas as pd

st.title('Bitcoin Price Prediction')

st.info(' This project aims to predict the future price of Bitcoin using machine learning techniques.')

with st.expander('Data'):
  st.write('**Raw data**')
  df = pd.read_csv('BTC-USD_stock_data.csv')
  df

  st.write('**X**')
  x = df.drop(['Close'],axis=1)
  x

  st.write('**y**')
  y = df['Close']
  y




with st.expander('Data Visualization'):
    # Display a line chart for Bitcoin Close prices over time
    st.subheader("Bitcoin Close Price Over Time")
    chart_data = df[['Close']].copy()  # Select only 'Close' column for line chart
    st.line_chart(chart_data)
