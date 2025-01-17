import yfinance as yf
import numpy as np
import pandas as pd
import alpaca_trade_api as tradeapi
import tensorflow as tf
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import schedule
import time
import os
import config
import keyboard
import threading as ted

# .a and .b are Alpha and Beta extensions
CURRENT_VERSION = "2.1.a"

# How many actions have been commited
bought, sold = 0

# Alpaca API Credentials
API_KEY = config.alpaca_api_key
API_SECRET = config.alpaca_secret_api
BASE_URL = 'https://paper-api.alpaca.markets'

# Initialize Alpaca API
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')
denseone, densetwo, densethree, densefour, densefive = 0

# Money and Stock Stats
money = 100
starting_money = money
current_qty = 0

# Clear the terminal
def clear():
    os.system("cls")

# Prep time data (look over, maybe delete)
def time_functions():
    global timed
    seconds, minutes, hours = get_stats()
    if minutes == -1 and hours == -1:
        timed = [seconds, -1, -1]
    elif hours == -1 and minutes != -1:
        timed = [seconds, minutes, -1]
    else:
        timed = [seconds, minutes, hours]


# Fetch all historical stock data for a given symbol
def fetch_all_data(stock_symbol):
    data = yf.download(stock_symbol, period='max', interval='1d')
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['RSI'] = compute_RSI(data['Close'], 14)
    return data.dropna()[['Close', 'SMA_20', 'SMA_50', 'RSI']] 

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
    scaler = MinMaxScaler(feature_range=(0, 1))  # Scale data to [0, 1] for the model
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(time_step, len(scaled_data)):
        X.append(scaled_data[i-time_step:i, :])
        if scaled_data[i, 0] > scaled_data[i-1, 0]:
            y.append(1) 
        else:
            y.append(0) 

    return np.array(X), np.array(y), scaler

# Build and train the LSTM model
def build_and_train_model(X_train, y_train, output_layer_neurons, output_layer_mode):
    global denseone, densetwo, densethree, densefour, densefive
    global lstm_layer_one_units, lstm_layer_two_units
    global epoch_size, batch_size, dropout
    model = Sequential()
    clear()
    batch_size = input("Batch Size : ")
    clear()
    epoch_size = input("Epoch Size : ")
    clear()
    dropout = (input("Drop Out Percentage : ")) / 100
    clear()
    lstm_layer_one_units = input("Neurons For Layer 1 : ")
    model.add(LSTM(units=lstm_layer_one_units, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(dropout))
    clear()
    lstm_layer_two_units = input("Neurons For Layer 2  : ")
    model.add(LSTM(units=lstm_layer_two_units, return_sequences=False))
    model.add(Dropout(dropout))
    clear()
    dense_layer_amount = input("Amount of Dense Layers (Max : 5) : ")
    clear()
    for layer in range(dense_layer_amount):
        neurons = int(input(f"Dense Layer {layer} neurons : "))
        if layer == 1:
            denseone = neurons
        elif layer == 2:
            densetwo = neurons
        elif layer == 3:
            densethree = neurons
        elif layer == 4:
            densefour = neurons
        elif layer == 5:
            densefive = neurons
        else:
            print("BRUH WTH")
        model.add(Dense(units=neurons))
        layerAMT += 1
    model.add(Dense(units=output_layer_neurons, activation=output_layer_mode))  # Sigmoid for binary classification (Buy/Sell)

    start = time.perf_counter()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=epoch_size, batch_size=batch_size)
    end = time.perf_counter()
    time_taken = round((end - start), 2)
    if time_taken > 60:
        print(f"It took {time_taken} seconds to train your AI")
    else:
        print(f"It took {round(time_taken/60, 2)} minutes to train your AI")
    return model, end

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
    
    if action >= (1):
        bought += 1
        return 'buy'
    elif action <= (0):
        sold += 1
        return 'sell'
    

def get_profit(money, starting_money):
    profit = money - starting_money
    profit_prcntg = (profit/starting_money) * 100
    return profit, profit_prcntg
    
    
# Execute trades based on the model's prediction
def trade_based_on_prediction(stock_symbol, error):
    action = predict_action(model, scaler, stock_symbol, error)
    # Get current stock position
    try:
        position = api.get_position(stock_symbol)
        current_qty =+ int(position.qty)
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

def get_stats(money, original_money, current_qty, current_stock_price,  bought, sold, hold, start, end,):
    profit, profit_prcntg = get_profit(money, starting_money)
    time_taken = round(end - start, 0)
    if time_taken < 60:
        print(f"Trader has been running for {time_taken} seconds!")
        return time_taken, -1, -1
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
        
        return time_taken, minutes, hours
    else:
        minutes = 0
        while True:
            time_taken -= 60
            minutes += 1
            if time_taken < 60:
                break
        print(f"Trader has been running for {minutes} minutes, and {time_taken} seconds")
        
        return  time_taken, minutes, -1
 
    print(f"Current Money : {money}")
    print(f"Current Amount of Stock : {current_qty}")
    print(f"Profit : {profit}")
    print(f"Profit Percentage : {profit_prcntg}")
    print(f"Current Stock Price : {current_stock_price}")
    print(f"Trader has bought {bought} times, sold {sell} times, and has held {hold} times")

def decision_picking():
    global decision
    print("1. Save")
    print("2. Load")
    print("3. Stats")
    print("4. Change Error")
    print("5. Save Data")
    print("6. Load Data Table")
    print("7. Live Graphing")
    print("8. Threading")
    print("9. More Training")
    print("Exit (ESC)")
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
        get_stats(money, starting_money, current_qty, get_current_price(stock_symbol), bought, sold, hold, start, end=time.perf_counter())
    elif decision == "4":
        clear()
        new_error = input("New Error : ")
        error = new_error
    elif decision == "5":
        clear()
        # Not done yet, add drop out to data table
        #dp.save_data_to_table(lstm_layer_one_units, lstm_layer_two_units)
        print("Saved Data To Table")
    elif decision == "6":
        clear()
        with open("data/data.md", "r") as data:
            for i in range(len(data.readlines)):
                print(data.readlines())
    elif decision == "7":
        clear()
        graph_dec = input("1. Line Plot \n2. Scatter Plot \3. 3D graph \n4. Back")
        if graph_dec == "1":
            x = input("What's your X-Axis? (L1N, L2N, D1N, D2N, D3N, D4N, D5N, DO, BS, ES, ER, TTB, TR, HELP) : ")
            if x == "HELP": 
                print("L1N = Layer One Neurons \nL2N = Layer Two Neurons \nD1N = Dense 1 Neurons")
            x_axis = graph_axis[x]
            y = input("What's your Y-Axis? (L1N, L2N, D1N, D2N, D3N, D4N, D5N, DO, BS, ES, ER, TTB, TR, HELP) : ")
            if y == "HELP": 
                print("L1N = Layer One Neurons \nL2N = Layer Two Neurons \nD1N = Dense 1 Neurons")
            #y_axis = graph_axis[y]
            #graph = dp.graph.line_plot(x_axis, y_axis)
            #graph_ted = ted.Thread(graph, x_axis, y_axis, graph_axis[y], graph_axis[x], target=dp.graph.appendAndUpdate.scatter_plot)
            #graph_ted.start()
        elif graph_dec == "2":
            x = input("What's your X-Axis? (L1N, L2N, D1N, D2N, D3N, D4N, D5N, DO, BS, ES, ER, TTB, TR, HELP) : ")
            if x == "HELP": 
                print("L1N = Layer One Neurons \nL2N = Layer Two Neurons \nD1N = Dense 1 Neurons")
            x_axis = graph_axis[x]
            y = input("What's your Y-Axis? (L1N, L2N, D1N, D2N, D3N, D4N, D5N, DO, BS, ES, ER, TTB, TR, HELP) : ")
            if y == "HELP": 
                print("L1N = Layer One Neurons \nL2N = Layer Two Neurons \nD1N = Dense 1 Neurons")
            #y_axis = graph_axis[y]
            #graph = dp.graph.scatter_plot(x_axis, y_axis)
            #graph_ted = ted.Thread( graph, x_axis, y_axis, graph_axis[y], graph_axis[x], target=dp.graph.appendAndUpdate.scatter_plot)
            #graph_ted.start()
                
        elif decision == "8":
            clear()
            print("Threading is only currently under Development. \n**Will be released in v2.2**")
        elif decision == "9":
            clear()
            print("More Training is only currently under Development. \n**Will be released in v2.2**")
        # When click ESC button, exit program
        elif decision == "ESC":
            clear()
            try:
                if keyboard.is_pressed("ctrl + s"):
                    quit("Exit succesful")
                else:
                    pass
            except:
                pass
                
       
graph_axis = {
    "L1N" : lstm_layer_one_units,
    "L2N" : lstm_layer_two_units, 
    "D1N" : denseone, 
    "D2N" : densetwo, 
    "D3N" : densethree,
    "D4N" : densefour, 
    "D5N" : densefive,
    "DO" : dropout,
    "BS" : batch_size,
    "ES" : epoch_size,  
}

# Main program execution
if __name__ == "__main__":
    clear()
    model_decision = input("Do you want to train a new model or load an existing one (new/load) : ")
    clear()
    interval = input("Interval : ")
    clear()
    stock_symbol = input("Stock Symbol : ")
    clear()
    error = input("Error : ")
    clear()
    if model_decision == "new":
        clear()
        data = fetch_all_data(stock_symbol)
        X_train, y_train, scaler = prepare_full_data(data)
        model, ttb = build_and_train_model(X_train, y_train)
    elif model_decision == "load":
        clear()
        file_name = f'trained-models/{input("File Directory : ")}'
        model = load_model(file_name)

    
    
    
    # Schedule trades every X minutes
    schedule.every(interval).minutes.do(trade_based_on_prediction, stock_symbol, error=error)
    dice = ted.Thread(target=decision_picking)
    start = time.perf_counter()
    # Keep the script running
    while True:
        schedule.run_pending()
        dice.start()
        if decision == "":
            pass
        clear()
        time.sleep(0.2)
