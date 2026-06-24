# Multi-Agent Financial Forecasting

This project implements an adaptive multi-agent financial forecasting framework for predicting SPY (S&P 500 ETF) returns using a combination of statistical, machine learning, and deep learning models.

The system consists of multiple forecasting agents:

* Ordinary Least Squares (OLS)
* XGBoost
* Long Short-Term Memory (LSTM)

Each agent generates daily market forecasts using SPY and VIX-based features. Instead of relying on a single model, predictions are combined through a no-regret online learning framework based on the Hedge algorithm and Fixed-Share updates.

The adaptive ensemble continuously adjusts model weights according to recent performance, enabling the system to respond to changing market regimes and non-stationary financial environments.

## Features

* Multi-agent forecasting architecture
* OLS, XGBoost, and LSTM prediction models
* Hedge-based adaptive ensemble weighting
* Fixed-Share mechanism for regime shifts
* SPY and VIX feature engineering
* Historical backtesting framework
* Performance evaluation using risk-adjusted metrics

## Technologies Used

* Python
* NumPy
* Pandas
* Scikit-Learn
* XGBoost
* TensorFlow / PyTorch
* Matplotlib

## Objective

To develop a robust forecasting framework that dynamically allocates trust to the best-performing model over time, improving prediction stability and adaptability compared to static ensemble approaches.
