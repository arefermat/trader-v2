markdown
Copy code
# Stock Trading Bot with LSTM and Alpaca API

This repository contains a stock trading bot that uses LSTM (Long Short-Term Memory) neural networks for predicting stock price movements and automates trading using Alpaca's paper trading API.

## Features

- **Data Fetching**: Downloads historical stock data using Yahoo Finance (`yfinance`).
- **Technical Indicators**: Computes simple moving averages (SMA) and relative strength index (RSI) for feature engineering.
- **Machine Learning**: Uses LSTM neural networks for stock price prediction (buy/sell decisions).
- **Alpaca API**: Automatically executes trades based on predictions on Alpaca's paper trading platform.

## Prerequisites

Before running the project, make sure you have the following installed on your system:

- Python 3.8 or higher
- Pip (Python package manager)
- [Alpaca Paper Trading Account](https://alpaca.markets)

## Getting Started

### 1. Clone the Repository

```
git clone https://github.com/yourusername/yourrepository.git
cd yourrepository
```

### 2. Install Dependencies
First, ensure you have all required dependencies by running the following command:

```
pip install -r requirements.txt
```

Alternatively, install these packages manually:

```
pip install yfinance pandas numpy alpaca-trade-api tensorflow scikit-learn schedule keyboard
```

### 3. API Key Configuration
To access Alpaca's trading API, you'll need to provide your Alpaca API key and secret in the config.py file.

Create a config.py file in the root directory.
Add the following lines to your config.py:

```
alpaca_api_key = 'YOUR_ALPACA_API_KEY'
alpaca_secret_api = 'YOUR_ALPACA_SECRET_KEY'
```
Replace 'YOUR_ALPACA_API_KEY' and 'YOUR_ALPACA_SECRET_KEY' with the actual credentials from your Alpaca account.

### 4. Preparing to Run
Once you have installed the required libraries and set up your Alpaca API credentials, you can proceed to run the script.

### 5. Running the Script
To run the trading bot, execute:

```
python trading_bot.py
```

### 6. Training and Loading Models
When the script starts, it will prompt you to either train a new model or load an existing one:

Train a new model: Enter "new" and provide the required hyperparameters (e.g., stock symbol, batch size, epoch size, etc.)
Load an existing model: Enter "load" and specify the path to the saved model.

### 7. Automating Trades
Once the model is trained or loaded, the bot will automatically fetch stock data, make predictions, and place trades based on the decision (buy, sell, hold). The trading frequency is determined by the interval you provide when starting the bot.

### 8. Saving or Loading Models
You can save your trained models for future use or load previously trained models:

Save a model: Press 1 during runtime and provide a filename. The model will be saved in the trained-models/ directory.
Load a model: Press 2 during runtime and specify the model's filename to reload it.
### 9. Monitoring Performance
During runtime, press 3 to see current statistics like profit, percentage gain/loss, number of buy/sell actions, and the amount of stock held.

## Notes

The script is designed to work with Alpaca’s paper trading environment. Ensure that your Alpaca account is set to paper trading mode for safe testing.
Be cautious while dealing with real-money trading and backtest thoroughly before deploying.

## Troubleshooting

Common Errors : 

Missing config.py: Ensure you create a config.py file with your Alpaca API credentials.
Incorrect API Keys: Double-check that you have entered your Alpaca API key and secret correctly.
Library Installation Issues: Make sure all dependencies are installed using the commands in the setup section.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Contact
For any questions or issues, please open an issue or reach out to arefermat.


## Key Sections:

1. **Installation**: Guides users on installing necessary packages and setting up API credentials.
2. **Running the Script**: Explains how to run the trading bot, including saving/loading models.
3. **API Configuration**: Covers adding `config.py` for Alpaca API credentials.
4. **Troubleshooting**: Points out common issues and solutions.
