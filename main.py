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

# .a and .b are Alpha and Beta extensions
CURRENT_VERSION = "2.0.b"

bought, sold, hold = 0

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
    model.add(Dense(units=output_layer_neurons, activation=output_layer_mode))  # Sigmoid for binary classification (Buy/Sell)

    start = time.perf_counter()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    end = time.perf_counter()
    time_taken = round((end - start), 2)
    if time_taken > 60:
        print(f"It took {time_taken} seconds to train your AI")
    else:
        print(f"It took {round(time_taken/60, 2)} minutes to train your AI")
    return model

def get_current_price(symbol):
    barset = api.get_barset(symbol, 'minute', 1)
    stock_bars = barset[symbol]
    return stock_bars[-1].c

# Predict Buy/Sell signals using the model
def predict_action(model, scaler, stock_symbol, error=0.3):
    global bought, sold, hold
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
    
    if action >= (0.5+error):
        bought += 1
        return 'buy'
    elif action <= (0.5-error):
        sold += 1
        return 'sell'
    else:
        hold += 1
        return 'hold'
    

def get_profit(money, starting_money):
    profit = money - starting_money
    profit_prcntg = (profit/starting-money) * 100
    return profit, profit_prcntg
    
    
# Execute trades based on the model's prediction
def trade_based_on_prediction(stock_symbol, error):
    action = predict_action(model, scaler, stock_symbol, error)
    # Get current stock position
    try:
        position = api.get_position(stock_symbol)
        current_qty = int(position.qty)
    except Exception as e:
        current_qty = 0
    
    # Execute trade based on the action
    if action == 'buy' and current_qty == 0 and money >= get_current_price(stock_symbol):
        print(f"Trader predicts buy for {stock_symbol}. Placing buy order.")
        buy_stock(stock_symbol, 1)
    elif action == 'sell' and current_qty > 0:
        print(f"Trader predicts sell for {stock_symbol}. Placing sell order.")
        money += (get_current_price(stock_symbol) * current_qty)
        sell_stock(stock_symbol, current_qty)
    elif action == 'hold' and current_qty != 0:
        print("Trader says to hold your stocks.")

def load_model(filename):
    tf.keras.models.load_model(filename)


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

def get_stats(money, original_money, current_qty, current_stock_price, start, end, bought, sold, hold):
    profit, profit_prcntg = get_profit(money, starting_money)
    time_taken = round(end - start, 0)
    if time_taken < 60:
        print(f"Trader has been running for {time_taken} seconds!")
    elif time_taken > 3600:
        hours = 0
        minutes = 0
        while True:
            time_taken -= 3600
            hours += 1
            if time_taken < 3600:
                break
        while True:
            time_taken -= 60
            minutes += 1
            if time_taken < 60:
                break
        print(f"Trader has been running for {hours} hours, {minutes} minutes, and {time_taken} seconds")
    else:
        minutes = 0
        while True:
            time_taken -= 60
            minutes += 1
            if time_taken < 60:
                break
        print(f"Trader has been running for {minutes} minutes, and {time_taken} seconds")
 
    print(f"Current Money : {money}")
    print(f"Current Amount of Stock : {current_qty}")
    print(f"Profit : {profit}")
    print(f"Profit Percentage : {profit_prcntg}")
    print(f"Current Stock Price : {current_stock_price}")
    print(f"Trader has bought {bought} times, sold {sell} times, and has held {hold} times")

# Main program execution
if __name__ == "__main__":
    clear()
    interval = input("Interval : ")
    model_decision = input("Do you want to train a new model or load an existing one")
    if model_decision == "new":
        clear()
        stock_symbol = input("Stock Symbol : ")
        clear()
        batch_size = input("Batch Size : ")
        clear()
        epoch_size = input("Epoch Size : ")
        clear()
        lstm_layer_one_units = input("Neurons For Layer 1 : ")
        clear()
        lstm_layer_two_units = input("Neurons For Layer 2  : ")
        clear()
        dense_layer_amount = input("Amount of Dense Layers : ")
        clear()
        dropout = (input("Drop Out Percentage : ")) / 100
        clear()
        error = input("Error : ")
    elif model_decision == "load":
        clear()
        file_name = f'trained-models/{input("File Directory : ")}'
        stock_symbol = input("Security: Stock Symbol : ")
        model = load_model(file_name)

    
    
    # Fetch and prepare all historical data
    data = fetch_all_data(stock_symbol)
    X_train, y_train, scaler = prepare_full_data(data)
    
    # Build and train the model
    model = build_and_train_model(X_train, y_train, epochs=epoch_size, batch_size=batch_size, lstm_layer_one_neurons=lstm_layer_one_units, lstm_layer_two_neurons=lstm_layer_two_neurons, dropout=dropout, dense_layer_amount=dense_layer_amount, output_layer_neurons=1)
    
    # Schedule trades every X minutes
    schedule.every(interval).minutes.do(trade_based_on_prediction, stock_symbol, error=error)
    start = time.perf_counter()
    # Keep the script running
    while True:
        schedule.run_pending()
        print("1. Save")
        print("2. Load")
        print("3. Stats")
        print("4. Change Error")
        print("5. Exit")
        # Add more like threading and more training and more complex stuff (v2.1)
        print(f"Current Version : {CURRENT_VERSION}")
        decision = input(": ")
        if decision == "1":
            clear()
            file_name = f'trained-models/{input("File Name : ")}' + ".h5"
            model.save(file_name)
            print("Model saved")
        elif decision == "2":
            clear()
            file_name = f'trained-models/{input("File Directory : ")}' + ".h5"
            model = load_model(file_name)
            print("New Model Loaded")
        elif decision == "3":
            clear()
            get_stats(money, starting_money, current_qty, get_current_price(stock_symbol), start, end=time.perf_counter(), bought, sold, hold
        elif decision == "4":
            new_error = input("New Error : ")
            error = new_error
        elif decision == "5":
            clear()
            quit("Exit succesful")


        
        if keyboard.is_pressed("ctrl + p") == True:
            profit, prcntg = get_profit(money, current_qty, starting_money)
            print(f"Profit : {profit}$ \nProfit Percentage : {prcntg}% increase/decrease")

        time.sleep(0.2)
