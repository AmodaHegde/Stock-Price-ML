# app.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import timedelta

# Title of the Streamlit application
st.title("Basic Stock Price Prediction with LSTM")

# Streamlit Sidebar inputs
ticker = st.sidebar.text_input("Enter Stock Ticker", value="AAPL", max_chars=5)
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2023-01-01"))

# Fetch stock data from Yahoo Finance
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    return data

# Enhanced data preprocessing
def preprocess_data(df):
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(df)
    return scaled_data, scaler

# Modified create_datasets function
def create_datasets(scaled_data, time_steps=60):
    X, y = [], []
    for i in range(time_steps, len(scaled_data)):
        X.append(scaled_data[i-time_steps:i])
        y.append(scaled_data[i, 3])  # 3 is the index of 'Close' price
    X, y = np.array(X), np.array(y)
    
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return X_train, y_train, X_test, y_test

# Enhanced LSTM model
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Predict and inverse transform the scaled data
def predict_and_inverse(model, X_test, scaler):
    predictions = model.predict(X_test)
    predictions = np.column_stack((np.zeros((len(predictions), 3)), predictions, np.zeros((len(predictions), 1))))
    predictions = scaler.inverse_transform(predictions)[:, 3]
    return predictions

# Plotly interactive chart
def plot_predictions_plotly(train, valid):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index, y=train['Close'], name='Train Data'))
    fig.add_trace(go.Scatter(x=valid.index, y=valid['Close'], name='Actual Price'))
    fig.add_trace(go.Scatter(x=valid.index, y=valid['Predictions'], name='Predicted Price'))
    fig.update_layout(title='LSTM Stock Price Prediction',
                      xaxis_title='Date',
                      yaxis_title='Close Price USD ($)')
    st.plotly_chart(fig)

# Main app functionality
st.sidebar.header("Model Parameters")
time_steps = st.sidebar.slider("Time Steps", 30, 100, 60)
epochs = st.sidebar.slider("Epochs", 10, 100, 50)
batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64, 128])

data_load_state = st.text('Loading data...')
data = load_data(ticker, start_date, end_date)
data_load_state.text('Loading data...done!')

if st.checkbox('Show raw data'):
    st.subheader('Raw Data')
    st.write(data.tail())

# Preprocess the data
scaled_data, scaler = preprocess_data(data)

# Create datasets for training and testing
X_train, y_train, X_test, y_test = create_datasets(scaled_data, time_steps)

# Build and train the LSTM model
model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
with st.spinner('Training model...'):
    history = model.fit(X_train, y_train, validation_split=0.2, 
                        batch_size=batch_size, epochs=epochs, 
                        callbacks=[early_stopping], verbose=0)

# Plot training history
st.subheader('Training History')
fig, ax = plt.subplots()
ax.plot(history.history['loss'], label='Training Loss')
ax.plot(history.history['val_loss'], label='Validation Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend()
st.pyplot(fig)

# Make predictions and inverse transform the scaled data
predictions = predict_and_inverse(model, X_test, scaler)

# Prepare for plotting
train_data = data[:int(len(data)*0.8)]
valid_data = data[int(len(data)*0.8):]
valid_data = valid_data.iloc[time_steps:].copy()
valid_data['Predictions'] = predictions

# Plot the results using Plotly
plot_predictions_plotly(train_data, valid_data)

# Future predictions
st.subheader('Future Predictions')
days_to_predict = st.slider('Days to predict', 1, 30, 7)

last_60_days = scaled_data[-time_steps:]
future_predictions = []

for _ in range(days_to_predict):
    next_pred = model.predict(last_60_days.reshape(1, time_steps, 5))
    future_predictions.append(next_pred[0, 0])
    last_60_days = np.roll(last_60_days, -1, axis=0)
    last_60_days[-1] = np.array([0, 0, 0, next_pred[0, 0], 0])

future_predictions = np.array(future_predictions).reshape(-1, 1)
future_predictions = np.column_stack((np.zeros((len(future_predictions), 3)), future_predictions, np.zeros((len(future_predictions), 1))))
future_predictions = scaler.inverse_transform(future_predictions)[:, 3]

future_dates = pd.date_range(start=valid_data.index[-1] + timedelta(days=1), periods=days_to_predict)
future_df = pd.DataFrame(index=future_dates, data={'Predictions': future_predictions})

fig = go.Figure()
fig.add_trace(go.Scatter(x=valid_data.index, y=valid_data['Close'], name='Historical Price'))
fig.add_trace(go.Scatter(x=future_df.index, y=future_df['Predictions'], name='Future Predictions'))
fig.update_layout(title='Future Stock Price Predictions',
                  xaxis_title='Date',
                  yaxis_title='Close Price USD ($)')
st.plotly_chart(fig)

st.write(future_df)
