import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import date_range
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime
from src.settings import INPUT_FOLDER, OUTPUT_FOLDER, PROCESSED_FOLDER



def sarima(df):

    # Convert 'Date_Time' to datetime
    df['Date_Time'] = pd.to_datetime(df['Date_Time'])

    # Set 'Date_Time' as the index for time series analysis
    df.set_index('Date_Time', inplace=True)

    # Select the 'Wait Time' column for forecasting
    wait_times = df['Wait Time']

    # Visualize the time series data
    plt.figure(figsize=(10, 6))
    plt.plot(wait_times)
    plt.title('Wait Time over Time')
    plt.xlabel('Time')
    plt.ylabel('Wait Time (minutes)')
    plt.show()

    # Fit a SARIMA model
    # We will assume a basic SARIMA(p,d,q)(P,D,Q,s) model
    # (p, d, q): p=autoregressive terms, d=differencing, q=moving average terms
    # (P, D, Q, s): seasonal autoregressive, differencing, moving average terms, s=seasonal period
    # You can tune (p, d, q, P, D, Q, s) based on AIC/BIC or domain knowledge
    model = SARIMAX(wait_times, order=(1, 1, 1),
                    seasonal_order=(1, 1, 1, 12))  # s=12 for seasonality, change based on domain knowledge

    # Fit the model
    sarima_result = model.fit()

    # Print model summary
    print(sarima_result.summary())

    # Forecast for the next day (since we have minute-level data, we forecast for next 1440 minutes)
    forecast_steps = 1440
    forecast = sarima_result.get_forecast(steps=forecast_steps)

    # Get confidence intervals
    forecast_ci = forecast.conf_int()

    # Plot the forecasted values
    plt.figure(figsize=(10, 6))
    plt.plot(wait_times.index, wait_times, label='Observed')
    plt.plot(forecast.predicted_mean.index, forecast.predicted_mean, label='Forecast')
    plt.fill_between(forecast_ci.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='k', alpha=0.2)
    plt.title('SARIMA Forecast for Wait Time')
    plt.xlabel('Time')
    plt.ylabel('Wait Time (minutes)')
    plt.legend()
    plt.show()

    # Print the forecasted values
    print(forecast.predicted_mean)

    return


if __name__ == "__main__":
    # This is where the training of the model will be performed

    model_data_file = os.path.join(PROCESSED_FOLDER, 'data_wth_features.csv')
    df = pd.read_csv(model_data_file)




    # prediction should be predicting the wait time for (at least
    # 30 mnin intervals) for the next day
    sarima(df)





