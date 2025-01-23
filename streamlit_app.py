import streamlit as st
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

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

with st.expander('Data Visualization'):
    # Display a line chart for Bitcoin Close prices over time
    st.subheader("Bitcoin Close Price Over Time")
    chart_data = df[['Close']].copy()  # Select only 'Close' column for line chart
    st.line_chart(chart_data)

    # Alternatively, you can display the scatter plot for Open vs Close prices:
    st.subheader("Open vs Close Price")
    # plt.figure(figsize=(10, 5))
    sns.scatterplot(x=df['Open'], y=df['Close'], color='red')
    plt.title("Open vs Close Price")
    plt.xlabel("Open Price (USD)")
    plt.ylabel("Close Price (USD)")
    st.pyplot(plt)

    # Line plot for 'Volume' over time (optional)
    st.subheader("Bitcoin Trading Volume Over Time")
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=df.index, y=df['Volume'], label='Volume', color='g')
    plt.title("Bitcoin Trading Volume Over Time")
    plt.xlabel("Date")
    plt.ylabel("Volume")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)
