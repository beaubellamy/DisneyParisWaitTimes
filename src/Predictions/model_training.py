import os
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import time
import logging
from src.settings import INPUT_FOLDER, OUTPUT_FOLDER, PROCESSED_FOLDER

from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf


def sarima(df):


    # Step 1: Setup logging for progress tracking
    # logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    #
    # # Step 2: Load the Data
    # data = {
    #     'Ride': ['Avengers Assemble: Flight Force'] * 13,
    #     'Date_Time': ['2023-02-02 08:40:00', '2023-02-02 08:45:00', '2023-02-02 08:50:00', '2023-02-02 08:55:00',
    #                   '2023-02-02 09:00:00', '2023-02-02 09:05:00', '2023-02-02 09:10:00', '2023-02-02 09:15:00',
    #                   '2023-02-02 09:20:00', '2023-02-02 09:25:00', '2023-02-02 09:30:00', '2023-02-02 09:35:00',
    #                   '2023-02-02 09:40:00'],
    #     'Wait Time': [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 10, 15, 15]
    # }
    #
    # df = pd.DataFrame(data)

    # Convert the 'Date_Time' column to datetime and set as index
    df['Date_Time'] = pd.to_datetime(df['Date_Time'])
    df.set_index('Date_Time', inplace=True)

    # Step 3: Resample to 5-min intervals and handle missing values
    # df = df.resample('5T').mean()  # Ensure the data is in 5-min intervals
    # df['Wait Time'] = df['Wait Time'].ffill()  # Forward fill any missing values

    # Step 4: Track the training time
    # start_time = time.time()
    # logging.info("Training started...")

    # We assume daily seasonality (144 periods per day = 24 hours * 60 minutes / 5-minute intervals)
    # Weekly seasonality = 144 periods * 7 days
    seasonal_order = (1, 1, 1, 144)  # Daily seasonality
    order = (1, 0, 1)  # Standard SARIMA non-seasonal part

    # Step 5: Fit the SARIMA model and add performance tracking
    model = SARIMAX(df['Wait Time'], order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)

    # Fit the model and track training time with updates
    # sarima_result = model.fit(disp=False)  # Disable built-in output from SARIMAX fit
    sarima_result = model.fit(disp=True)  # Disable built-in output from SARIMAX fit

    # end_time = time.time()
    # total_training_time = end_time - start_time

    # logging.info(f"Training completed in {total_training_time:.2f} seconds.")
    # print(f'Training completed in {total_training_time:.2f} seconds.')

    # Step 6: Evaluate the model's performance using in-sample fit statistics
    # logging.info(f"Model Summary: \n{sarima_result.summary()}")
    print(f'Model Summary: \n{sarima_result.summary()}')

    # Step 7: Forecast the next day's wait times (next 24 hours at 5-min intervals)
    forecast_steps = 24 * 12  # 24 hours * 12 intervals per hour (5-min intervals)
    forecast = sarima_result.get_forecast(steps=forecast_steps)
    forecast_values = forecast.predicted_mean

    # Create a date range for the forecast period (next 24 hours)
    forecast_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(minutes=5), periods=forecast_steps, freq='5T')

    # Step 8: Plot the forecast
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['Wait Time'], label='Observed Wait Time', color='blue')
    plt.plot(forecast_dates, forecast_values, label='Forecasted Wait Time', color='red', linestyle='--')
    plt.title('Wait Time Forecast for Next Day (5-min Intervals)')
    plt.xlabel('Date Time')
    plt.ylabel('Wait Time (minutes)')
    plt.legend()
    plt.grid(True)
    plt.show()

    return


if __name__ == "__main__":
    # This is where the training of the model will be performed

    model_data_file = os.path.join(PROCESSED_FOLDER, 'data_wth_features.csv')
    df = pd.read_csv(model_data_file)

    # prediction should be predicting the wait time for (at least
    # 30 mnin intervals) for the next day
    ride1 = df[df['Ride'] == 'Avengers Assemble: Flight Force']

    plot_acf(ride1['Wait Time'])
    plt.show()

    autocorrelation_plot(ride1['Wait Time'])


    plot_pacf(ride1['Wait Time'])
    plt.show()

    sarima(ride1)





