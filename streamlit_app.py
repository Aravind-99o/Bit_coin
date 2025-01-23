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




with st.expander('Data visualization'):
  import seaborn as sns
  import matplotlib.pyplot as plt



  # Plot a line graph of Bitcoin Closing prices over time
  st.subheader("Bitcoin Close Price Over Time")
  plt.figure(figsize=(10, 5))
  sns.lineplot(x=df.index, y=df['Close'], label='Close Price', color='b')
  plt.title("Bitcoin Close Price Over Time")
  plt.xlabel("Date")
  plt.ylabel("Close Price (USD)")
  plt.xticks(rotation=45)
  plt.tight_layout()
  st.pyplot(plt)
