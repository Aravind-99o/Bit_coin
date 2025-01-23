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



import matplotlib.pyplot as plt
import seaborn as sns
with st.expander('Data visualization'):
  numeric_df_train = df.select_dtypes(include=[np.number])

  # Create the heatmap plot
  plt.figure(figsize=(10, 8))
  sns.heatmap(numeric_df_train.corr(), annot=True, cmap='coolwarm', fmt='.2f')
  plt.title('Correlation Heatmap - Training Dataset')

  # Display the plot in Streamlit
  st.pyplot(plt)
