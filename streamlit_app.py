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


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# # Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# Data cleaning
df = df.dropna()  # Drop rows with NaN values
x = df.drop(columns=['Close'])
y = df['Close']
    
# Ensure all data is numeric
x = x.apply(pd.to_numeric, errors='coerce')
y = pd.to_numeric(y, errors='coerce')
    
# Data cleaning
    df = df.dropna()  # Drop rows with NaN values
    
    # Ensure all data is numeric
    x = x.apply(pd.to_numeric, errors='coerce')
    y = pd.to_numeric(y, errors='coerce')
    
    # Check for NaN values after conversion
    if x.isnull().any().any() or y.isnull().any():
        st.write("Data contains NaN values after conversion to numeric. Please check your data.")
    else:
        # Split the data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        # Convert to numpy arrays
        x_train = x_train.values
        x_test = x_test.values
        y_train = y_train.values
        y_test = y_test.values

        # Create and train the model
        model = LinearRegression()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write(f'Mean Squared Error: {mse}')
        st.write(f'R^2 Score: {r2}')
else:
    st.write("Please upload a CSV file.")
# # Convert to numpy arrays
# x_train = x_train.values
# x_test = x_test.values
# y_train = y_train.values
# y_test = y_test.values

# # Create and train the model
# model = LinearRegression()
# model.fit(x_train, y_train)
# y_pred = model.predict(x_test)

# # Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print(f'Mean Squared Error: {mse}')
# print(f'R^2 Score: {r2}')




# # Display predicted species
# st.subheader('Predicted Closing Price')
