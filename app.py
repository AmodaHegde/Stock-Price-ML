# app.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

# Title of the application
st.title("Basic Stock Price Prediction with LSTM")

# Sidebar inputs
ticker = st.sidebar.text_input("Enter Stock Ticker", value="AAPL", max_chars=5)
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2023-01-01"))

# Fetch stock data from Yahoo Finance
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    return data

# Data preprocessing
def preprocess_data(df):
    df = df[['Close']]
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(df)
    return scaled_data, scaler

# Create training and test datasets
def create_datasets(scaled_data):
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]

    X_train, y_train = [], []
    for i in range(60, len(train_data)):
        X_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    X_test, y_test = [], []
    for i in range(60, len(test_data)):
        X_test.append(test_data[i-60:i, 0])
        y_test.append(test_data[i, 0])
    X_test, y_test = np.array(X_test), np.array(y_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    return X_train, y_train, X_test, y_test

# Build and train the LSTM model
def build_lstm_model(X_train):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Predict and inverse transform the scaled data
def predict_and_inverse(model, X_test, scaler):
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    return predictions

# Plot the predictions
def plot_predictions(train, valid, predictions):
    plt.figure(figsize=(16,8))
    plt.title('LSTM Stock Price Prediction')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.plot(train['Close'], label='Train Data')
    plt.plot(valid.index, valid['Close'], label='Actual Price')
    plt.plot(valid.index, valid['Predictions'], label='Predicted Price')
    plt.legend(loc='upper left')
    st.pyplot(plt)

# Main app functionality
data_load_state = st.text('Loading data...')
data = load_data(ticker, start_date, end_date)
data_load_state.text('Loading data...done!')

if st.checkbox('Show raw data'):
    st.subheader('Raw Data')
    st.write(data.tail())

# Preprocess the data
scaled_data, scaler = preprocess_data(data)

# Create datasets for training and testing
X_train, y_train, X_test, y_test = create_datasets(scaled_data)

# Build and train the LSTM model
model = build_lstm_model(X_train)
model.fit(X_train, y_train, batch_size=1, epochs=1)

# Make predictions and inverse transform the scaled data
predictions = predict_and_inverse(model, X_test, scaler)

# Prepare for plotting
train_data = data[:int(len(data)*0.8)]
valid_data = data[int(len(data)*0.8):]
valid_data = valid_data.iloc[60:].copy()  # Adjust valid_data to match the number of predictions
valid_data['Predictions'] = predictions

# Plot the results
plot_predictions(train_data, valid_data, predictions)
