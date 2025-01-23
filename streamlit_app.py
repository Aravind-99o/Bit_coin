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
    # st.plotly_chart(fig)



# Input features
with st.sidebar:
  st.header('Input features')
  Open = st.slider('Open', 32.1, 59.6, 43.9)
  High = st.slider('High', 32.1, 59.6, 43.9)
  Low = st.slider('Low', 13.1, 21.5, 17.2)
  Adj_Close = st.slider('Adj Close', 172.0, 231.0, 201.0)
  Volume = st.slider('Volume', 2700.0, 6300.0, 4207.0)

    # Create a DataFrame for the input features
  data = {'Open': Open,
          'High': High,
          'Adj Close': Adj_Close,
          'Volume': Volume}
  input_df = pd.DataFrame(data, index=[0])




with st.expander('Input features'):
  st.write('**Input**')
  input_df


x=df.drop(['Close'],axis=1)
x

y=df['Close']
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
x_train

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
y_pred

from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)
mse

r2 = r2_score(y_test,y_pred)
r2





# Display predicted species
st.subheader('Predicted Closing Price')
