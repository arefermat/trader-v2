import yfinance as yf
import numpy as np
import pandas as pd
import alpaca_trade_api as tradeapi
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import schedule
import time
import os
import config
import keyboard

# Alpaca API Credentials
API_KEY = config.alpaca_api_key
API_SECRET = config.alpaca_secret_api
BASE_URL = 'https://paper-api.alpaca.markets'

# Initialize Alpaca API
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

money = 100
starting_money = money

# Fetch all historical stock data for a given symbol
def fetch_all_data(stock_symbol):
    data = yf.download(stock_symbol, period='max', interval='1d')  # Fetch full history
    data['SMA_20'] = data['Close'].rolling(window=20).mean()  # 20-day simple moving average
    data['SMA_50'] = data['Close'].rolling(window=50).mean()  # 50-day simple moving average
    data['RSI'] = compute_RSI(data['Close'], 14)  # 14-day RSI
    return data.dropna()[['Close', 'SMA_20', 'SMA_50', 'RSI']]  # Remove NaNs

# Compute RSI (Relative Strength Index)
def compute_RSI(data, window):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    RS = gain / loss
    RSI = 100 - (100 / (1 + RS))
    return RSI

# Prepare the data for machine learning (LSTM)
def prepare_full_data(data, time_step=60):
    scaler = MinMaxScaler(feature_range=(0, 1))  # Scale data to [0, 1]
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(time_step, len(scaled_data)):
        X.append(scaled_data[i-time_step:i, :])  # Last 60 time steps as input
        if scaled_data[i, 0] > scaled_data[i-1, 0]:
            y.append(1)  # Label as 'Buy' (1) if price is going up
        else:
            y.append(0)  # Label as 'Sell' (0) if price is going down

    return np.array(X), np.array(y), scaler

# Build and train the LSTM model
def build_and_train_model(X_train, y_train, epochs, batch_size, lstm_layer_one_neurons, lstm_layer_two_neurons, dropout, dense_layer_amount, output_layer_neurons, output_layer_mode="sigmoid"):
    model = Sequential()
    model.add(LSTM(units=lstm_layer_one_neurons, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(dropout))
    model.add(LSTM(units=lstm_layer_two_neurons, return_sequences=False))
    model.add(Dropout(dropout))
    layerAMT = 1
    for layer in range(dense_layer_amount):
        neurons = input(f"Dense Layer {layerAMT} neurons : ")
        model.add(Dense(units=neurons))
        layerAMT += 1
    model.add(Dense(units=1, activation='sigmoid'))  # Sigmoid for binary classification (Buy/Sell)
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    return model

def get_current_price(symbol):
    barset = api.get_barset(symbol, 'minute', 1)
    stock_bars = barset[symbol]
    return stock_bars[-1].c

# Predict Buy/Sell signals using the model
def predict_action(model, scaler, stock_symbol):
    # Fetch recent 60 days of stock data
    recent_data = yf.download(stock_symbol, period='60d', interval='1d')
    recent_data['SMA_20'] = recent_data['Close'].rolling(window=20).mean()
    recent_data['SMA_50'] = recent_data['Close'].rolling(window=50).mean()
    recent_data['RSI'] = compute_RSI(recent_data['Close'], 14)
    recent_data = recent_data.dropna()

    # Preprocess the data (scale it like the training data)
    scaled_recent_data = scaler.transform(recent_data[['Close', 'SMA_20', 'SMA_50', 'RSI']])
    X_recent = scaled_recent_data.reshape(1, scaled_recent_data.shape[0], scaled_recent_data.shape[1])
    
    # Predict (1: Buy, 0: Sell)
    action = model.predict(X_recent)
    
    if action > 0.5:
        return 'buy'
    else:
        return 'sell'

def get_profit(money, current_qty, starting_money):
    profit = money - starting_money
    profit_prcntg = profit/starting_money
# Execute trades based on the model's prediction
def trade_based_on_prediction(stock_symbol):
    action = predict_action(model, scaler, stock_symbol)
    # Get current stock position
    try:
        position = api.get_position(stock_symbol)
        current_qty = int(position.qty)
    except Exception as e:
        current_qty = 0
    
    # Execute trade based on the action
    if action == 'buy' and current_qty == 0 and money >= get_current_price(stcok_symbol):
        print(f"Model predicts buy for {stock_symbol}. Placing buy order.")
        buy_stock(stock_symbol, 1)
    elif action == 'sell' and current_qty > 0:
        print(f"Model predicts sell for {stock_symbol}. Placing sell order.")
        money += (get_current_price(stock_symbol) * current_qyt)
        sell_stock(stock_symbol, current_qty)

# Function to buy stock
def buy_stock(symbol, qty):
    api.submit_order(
        symbol=symbol,
        qty=qty,
        side='buy',
        type='market',
        time_in_force='gtc'
    )
    print(f"Bought {qty} share(s) of {symbol}")

# Function to sell stock
def sell_stock(symbol, qty):
    api.submit_order(
        symbol=symbol,
        qty=qty,
        side='sell',
        type='market',
        time_in_force='gtc'
    )
    print(f"Sold {qty} share(s) of {symbol}")

# Main program execution
if __name__ == "__main__":
    stock_symbol = 'AAPL'
    
    # Fetch and prepare all historical data
    data = fetch_all_data(stock_symbol)
    X_train, y_train, scaler = prepare_full_data(data)
    
    # Build and train the model
    model = build_and_train_model(X_train, y_train, epochs=10, batch_size=64)
    
    # Schedule trades every 5 minutes
    schedule.every(5).minutes.do(trade_based_on_prediction, stock_symbol)

    # Keep the script running
    while True:
        schedule.run_pending()
        time.sleep(1)
