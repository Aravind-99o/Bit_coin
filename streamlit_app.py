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

# Assuming df is the DataFrame containing Bitcoin data
# For demonstration purposes, let's assume you already have the 'df' dataset loaded.

# Example: Create a simple DataFrame (replace with actual df)
df = pd.read_csv("bitcoin_data.csv")  # Replace with the actual CSV file or data

# Convert 'Date' column to datetime if it's not already
df['Date'] = pd.to_datetime(df['Date'])

# Set the 'Date' as the index for easy plotting
df.set_index('Date', inplace=True)

# Streamlit title
st.title('Bitcoin Price Prediction Visualization')

# Display basic information about the dataset
st.write("Dataset Preview:")
st.write(df.head())

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

# Plotting the scatter plot for Open vs Close prices
st.subheader("Open vs Close Price")
plt.figure(figsize=(10, 5))
sns.scatterplot(x=df['Open'], y=df['Close'], color='red')
plt.title("Open vs Close Price")
plt.xlabel("Open Price (USD)")
plt.ylabel("Close Price (USD)")
st.pyplot(plt)

# Plot a line graph for 'Volume' over time (optional)
st.subheader("Bitcoin Trading Volume Over Time")
plt.figure(figsize=(10, 5))
sns.lineplot(x=df.index, y=df['Volume'], label='Volume', color='g')
plt.title("Bitcoin Trading Volume Over Time")
plt.xlabel("Date")
plt.ylabel("Volume")
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(plt)
