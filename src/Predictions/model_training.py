import os
from datetime import datetime

import pandas as pd
import numpy as np
from datetime import timedelta

from keras.src.backend.jax.random import dropout
from sklearn.metrics import accuracy_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import time
import logging
from src.settings import MODELS_FOLDER, PROCESSED_FOLDER
import pandas as pd
import numpy as np

import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import optimizers

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.kernel_ridge import KernelRidge
# from sklearn.model_selection import KFold
from sklearn import metrics



def split_data(df, target, test_size=0.2):

    train_size = int(len(df) * (1- test_size))
    x_train = df[:train_size]
    x_test = df[train_size:]
    y_train = target[:train_size]
    y_test = target[train_size:]

    return x_train, x_test, y_train, y_test


def calculate_metrics(y_test, predictions):
    mea =  metrics.mean_absolute_error(y_test, predictions)
    mse = metrics.mean_squared_error(y_test, predictions)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, predictions))

    # Calculating Error
    errors = round(metrics.mean_absolute_error(y_test, predictions), 2)
    # mean Absolute Percentage Error
    mape = 100 * (errors / y_test)
    mape.replace([np.inf, -np.inf], 0, inplace=True)

    acc_linear = (100 - np.mean(mape))
    accuracy = round(acc_linear, 2)


def update_metrics(ride_metrics, model, mae, mse, rmse, mape, mape_acc, accuracy):
    ride_metrics['model'].append(model)
    ride_metrics['mae'].append(mae)
    ride_metrics['mse'].append(mse)
    ride_metrics['rmse'].append(rmse)
    ride_metrics['mape'].append(mape.mean())
    ride_metrics['mape_acc'].append(mape_acc)
    ride_metrics['accuracy05'].append(accuracy['accuracy05'])
    ride_metrics['accuracy10'].append(accuracy['accuracy10'])
    ride_metrics['accuracy15'].append(accuracy['accuracy15'])
    ride_metrics['accuracy20'].append(accuracy['accuracy20'])

    return ride_metrics


def Predict_LinearRegresion(x_train, x_test, y_train, y_test, threshold, model_name='LinearRegresion'):
    lm = LinearRegression()
    model = lm.fit(x_train, y_train)
    # Make Prediction on test set
    y_pred = lm.predict(x_test)

    model_file = os.path.join(MODELS_FOLDER, f'{model_name}.pkl')
    # Save the model
    with open(model_file, 'wb') as file:
        pickle.dump(model, file)

    # Calculate metricx
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    # accuracy_score(y_test, linear_predictions)

    # Calculating Error
    errors = round(metrics.mean_absolute_error(y_test, y_pred), 2)
    # mean Absolute Percentage Error
    mape = 100 * (errors / y_test)
    mape.replace([np.inf, -np.inf], 0, inplace=True)
    mape_acc = (100 - np.mean(mape))

    # adjusted mae
    mae2 = abs(y_pred - y_test)
    mae2.replace([np.inf, -np.inf], 0, inplace=True)
    accuracy = {
        'accuracy05': 1 - mae2[mae2 > 5].shape[0] / mae2.shape[0],
        'accuracy10': 1 - mae2[mae2 > 10].shape[0] / mae2.shape[0],
        'accuracy15': 1 - mae2[mae2 > 15].shape[0] / mae2.shape[0],
        'accuracy20':  1 - mae2[mae2 > 20].shape[0] / mae2.shape[0]
    }

    return mae, mse, rmse, mape, mape_acc, accuracy


def Predict_RidgeRegresion(x_train, x_test, y_train, y_test, threshold, model_name='RidgeRegresssion'):
    # ridgeReg = KernelRidge(alpha=1.0, kernel='linear')
    model = Ridge(alpha=1.0)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    model_file = os.path.join(MODELS_FOLDER, f'{model_name}.pkl')
    # Save the model
    with open(model_file, 'wb') as file:
        pickle.dump(model, file)

    # Calculate metricx
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    # accuracy_score(y_test, linear_predictions)

    # Calculating Error
    errors = round(metrics.mean_absolute_error(y_test, y_pred), 2)
    # mean Absolute Percentage Error
    mape = 100 * (errors / y_test)
    mape.replace([np.inf, -np.inf], 0, inplace=True)
    mape_acc = (100 - np.mean(mape))

    # adjusted mae
    mae2 = abs(y_pred - y_test)
    mae2.replace([np.inf, -np.inf], 0, inplace=True)
    accuracy = {
        'accuracy05': 1 - mae2[mae2 > 5].shape[0] / mae2.shape[0],
        'accuracy10': 1 - mae2[mae2 > 10].shape[0] / mae2.shape[0],
        'accuracy15': 1 - mae2[mae2 > 15].shape[0] / mae2.shape[0],
        'accuracy20': 1 - mae2[mae2 > 20].shape[0] / mae2.shape[0]
    }

    return mae, mse, rmse, mape, mape_acc, accuracy


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


def randomforestregressor(X_train, X_test, y_train, y_test, threshold, model_name='RFRegressor'):
    """
    Predicts the wait time for the same time on the next day.

    Parameters:
        data (pd.DataFrame): Historical data with columns "Date", "Time", and "Wait Time".
                             "Date" should be in 'YYYY-MM-DD' format and "Time" in 'HH:MM' format.

    Returns:
        model (RandomForestRegressor): Trained model.
        next_day_prediction (float): Predicted wait time for the next day at the same time.
    """

    # Train a Random Forest Regressor
    model = RandomForestRegressor(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)

    model_file = os.path.join(MODELS_FOLDER, f'{model_name}.pkl')
    # Save the model
    with open(model_file, 'wb') as file:
        pickle.dump(model, file)


    # Test the model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    # Calculate metricx
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    # accuracy_score(y_test, linear_predictions)

    # Calculating Error
    errors = round(metrics.mean_absolute_error(y_test, y_pred), 2)
    # mean Absolute Percentage Error
    mape = 100 * (errors / y_test)
    mape.replace([np.inf, -np.inf], 0, inplace=True)
    mape_acc = (100 - np.mean(mape))

    # adjusted mae
    mae2 = abs(y_pred - y_test)
    mae2.replace([np.inf, -np.inf], 0, inplace=True)
    accuracy = {
        'accuracy05': 1 - mae2[mae2 > 5].shape[0] / mae2.shape[0],
        'accuracy10': 1 - mae2[mae2 > 10].shape[0] / mae2.shape[0],
        'accuracy15': 1 - mae2[mae2 > 15].shape[0] / mae2.shape[0],
        'accuracy20': 1 - mae2[mae2 > 20].shape[0] / mae2.shape[0]
    }

    return mae, mse, rmse, mape, mape_acc, accuracy


def sequential_model(layers=5, units=64, dropout=0.3, input_shape=(1,)):
    model = Sequential()

    # Add the first layer with input shape
    model.add(Dense(units, activation='relu', input_shape=input_shape))
    model.add(Dropout(dropout))

    # Dynamically add hidden layers
    for _ in range(layers - 1):
        model.add(Dense(units, activation='relu'))
        model.add(Dropout(dropout))

    # Output layer (regression)
    model.add(Dense(1))


    return model


def Predict_NeuralNetwork(X_train, X_test, y_train, y_test, layers=5, epochs=50, batch_size=32, model_name='NeuralNetowrk'):

    # Scale the data
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)

    # Build the model
    model = sequential_model(layers=layers, units=batch_size, dropout=0.3, input_shape=(X_train.shape[1],))

    # Compile the model
    optimizer = optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

    # Train the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)

    model_file = os.path.join(MODELS_FOLDER, f'{model_name}.pkl')
    # Save the model
    with open(model_file, 'wb') as file:
        pickle.dump(model, file)

    # Evaluate the model
    test_loss, test_mae = model.evaluate(X_test, y_test)
    print(f'Test Mean Absolute Error: {test_mae:.4f}')

    # Test the model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    # Calculate metricx
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    # accuracy_score(y_test, linear_predictions)

    # Calculating Error
    errors = round(metrics.mean_absolute_error(y_test, y_pred), 2)
    # mean Absolute Percentage Error
    mape = 100 * (errors / y_test)
    mape.replace([np.inf, -np.inf], 0, inplace=True)
    mape_acc = (100 - np.mean(mape))

    # adjusted mae
    mae2 = abs(y_pred[:,0] - y_test)
    mae2.replace([np.inf, -np.inf], 0, inplace=True)
    accuracy = {
        'accuracy05': 1 - mae2[mae2 > 5].shape[0] / mae2.shape[0],
        'accuracy10': 1 - mae2[mae2 > 10].shape[0] / mae2.shape[0],
        'accuracy15': 1 - mae2[mae2 > 15].shape[0] / mae2.shape[0],
        'accuracy20': 1 - mae2[mae2 > 20].shape[0] / mae2.shape[0]
    }

    return mae, mse, rmse, mape, mape_acc, accuracy


def run_5min_training(df):
    results = pd.DataFrame()

    # prediction will be predicting the wait time for each 5 min interval of the next day
    # loop over each ride
    for ride in df['Ride'].unique():

        ride_df = df[df['Ride'] == ride]
        ride_df.drop(columns=['Date_Time', 'Ride'], inplace=True)

        ride_metrics = {'model': [], 'mae': [], 'mse': [], 'rmse': [], 'mape': [], 'accuracy': []}
        # loop over each 5 min interval of the day
        for timepoint in ride_df['Time'].unique():

            timedata = ride_df[ride_df['Time'] == timepoint]

            # reduce variability of targets
            target = timedata['Wait Time']
            # target = np.log(timedata['Wait Time'])
            # target.replace([np.inf, -np.inf], 0, inplace=True)

            timedata.drop(columns=['Wait Time'], inplace=True)
            # 80% training & 25% testing - ensure order is maintained
            x_train, x_test, y_train, y_test = split_data(timedata, target, test_size=0.2)

            model = 'Linear Regression'
            mae, mse, rmse, mape, mape_acc, accuracy = (
                Predict_LinearRegresion(x_train, x_test, y_train, y_test, 'Linear5min'))
            ride_metrics = update_metrics(ride_metrics, model, mae, mse, rmse, mape, mape_acc, accuracy)

            model = 'Ridge Regression'
            mae, mse, rmse, mape, mape_acc, accuracy = (
                Predict_RidgeRegresion(x_train, x_test, y_train, y_test, 'Ridge5min'))
            ride_metrics = update_metrics(ride_metrics, model, mae, mse, rmse, mape, mape_acc, accuracy)

            model = 'Random Forest Regressor'
            mae, mse, rmse, mape, mape_acc, accuracy = (
                randomforestregressor(x_train, x_test, y_train, y_test, 'RandomForest5min'))
            ride_metrics = update_metrics(ride_metrics, model, mae, mse, rmse, mape, mape_acc, accuracy)

            print(f'Ride: {ride}: Time: {timepoint}')

        metric_df = pd.DataFrame(ride_metrics)
        ride_results = metric_df.groupby(by=['model']).mean()
        ride_results['ride'] = ride

        results = pd.concat([results, ride_results])
    results.to_csv(os.path.join(PROCESSED_FOLDER, 'avg_resuts-5min.csv'))
    return

def run_daily_training(df):
    results = pd.DataFrame()
    for ride in df['Ride'].unique():

        ride_df = df[df['Ride'] == ride]
        ride_df.drop(columns=['Date_Time', 'Ride'], inplace=True)

        ride_metrics = {'model': [], 'mae': [], 'mse': [], 'rmse': [], 'mape': [], 'accuracy': []}

        # reduce variability of targets
        target = ride_df['Wait Time']
        # target = np.log(timedata['Wait Time'])
        # target.replace([np.inf, -np.inf], 0, inplace=True)

        ride_df.drop(columns=['Wait Time'], inplace=True)
        # 80% training & 25% testing
        x_train, x_test, y_train, y_test = split_data(ride_df, target, test_size=0.2)

        model = 'Linear Regression'
        mae, mse, rmse, mape, mape_acc, accuracy = (
            Predict_LinearRegresion(x_train, x_test, y_train, y_test, 'LinearDaily'))
        ride_metrics = update_metrics(ride_metrics, model, mae, mse, rmse, mape, mape_acc, accuracy)

        model = 'Ridge Regression'
        mmae, mse, rmse, mape, mape_acc, accuracy = (
            Predict_RidgeRegresion(x_train, x_test, y_train, y_test, 'RidgeDaily'))
        ride_metrics = update_metrics(ride_metrics, model, mae, mse, rmse, mape, mape_acc, accuracy)

        model = 'Random Forest Regressor'
        mae, mse, rmse, mape, mape_acc, accuracy = (
            randomforestregressor(x_train, x_test, y_train, y_test, 'RandomForestDaily'))
        ride_metrics = update_metrics(ride_metrics, model, mae, mse, rmse, mape, mape_acc, accuracy)

        print(f'Ride: {ride}')

        metric_df = pd.DataFrame(ride_metrics)
        ride_results = metric_df.groupby(by=['model']).mean()
        ride_results['ride'] = ride

        results = pd.concat([results, ride_results])

    results.to_csv(os.path.join(PROCESSED_FOLDER, 'avg_resuts-daily.csv'))

    return


def run_training(df):

    ride_metrics = {'model': [], 'mae': [], 'mse': [], 'rmse': [],
                    'mape': [], 'mape_acc': [], 'accuracy05': [],
                    'accuracy10': [], 'accuracy15': [], 'accuracy20': []}
    threshold = 10

    # df = df.sample(frac=1).reset_index(drop=True)

    # target = np.log(timedata['Wait Time'])
    # target.replace([np.inf, -np.inf], 0, inplace=True)
    ride_encoded = pd.DataFrame()

    # detatch the ride column for later re-attachment
    if len(df['Ride'].unique()) > 1:
        ride_df = df['Ride']
        df.drop(columns='Ride', inplace=True)

        # onehot encode the ride labels
        ride_encoded = pd.get_dummies(ride_df)
    else:
        ride = df.loc[0,'Ride'].replace(':', '').replace(' ', '_')
        df.drop(columns=['Ride'], inplace=True)

    # Manually split data
    # df['Date_Time'] = pd.to_datetime(df['Date_Time'])
    train_date = df['Date'].min() + timedelta(days=587) # train: 587, test: +100,
    # test_date = df['Date'].min() + timedelta(days=587)
    train_idx = df[df['Date'] <= pd.to_datetime(train_date)].index
    # train_df = df[df['Date'] <= pd.to_datetime(train_date)]
    test_idx = df[df['Date'] > pd.to_datetime(train_date)].index
    # test_idx = df[(df['Date'] > pd.to_datetime(train_date)) & (df['Date_Time'] <= pd.to_datetime(test_date))].index
    # val_idx = df[df['Date'] > pd.to_datetime(test_date)].index
    df.drop(columns=['Date'], inplace=True)

    # extract the target
    y_train = df.loc[train_idx, 'Wait Time']
    y_test = df.loc[test_idx, 'Wait Time']
    # y_val = df.loc[val_idx, 'Wait Time']
    df.drop(columns=['Wait Time'], inplace=True)

    x_train = df.loc[train_idx,:]
    x_test = df.loc[test_idx,:]
    # x_val = df.loc[val_idx,:]

    print (x_train.columns)
    # 70% training & 30% testing
    # x_train, x_test, y_train, y_test = split_data(df, target, test_size=0.3)
    # x_val, x_test, y_val, y_test = split_data(x_test, y_test, test_size=0.6)

    # Select MinMaxScaler for uniformly distributed features
    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    # x_val_scaled = scaler.transform(x_val)

    with open(os.path.join(MODELS_FOLDER, f'{ride}_scalar.pkl'), 'wb') as file:
        pickle.dump(scaler, file)

    # re-attatch the ride features - by design - normalised
    x_train_scaled = pd.DataFrame(columns=x_train.columns, data=x_train_scaled)
    x_test_scaled = pd.DataFrame(columns=x_test.columns, data=x_test_scaled)

    if not ride_encoded.empty:
        x_train_scaled.loc[:, ride_encoded.columns] = ride_encoded.loc[train_idx,:].reset_index().astype(int)
        x_test_scaled.loc[:, ride_encoded.columns] = ride_encoded.loc[test_idx,:].reset_index().astype(int)

    # x_train_scaled.to_csv(os.path.join(PROCESSED_FOLDER, 'scaled_train_set.csv'))
    # x_test_scaled.to_csv(os.path.join(PROCESSED_FOLDER, 'scaled_test_set.csv'))



    # model = 'Linear Regression'
    # mae, mse, rmse, mape, mape_acc, accuracy = (
    #     Predict_LinearRegresion(x_train_scaled, x_test_scaled, y_train, y_test, threshold, 'Linear'))
    # ride_metrics = update_metrics(ride_metrics, model, mae, mse, rmse, mape, mape_acc, accuracy)
    #
    # model = 'Ridge Regression'
    # mae, mse, rmse, mape, mape_acc, accuracy = (
    #     Predict_RidgeRegresion(x_train_scaled, x_test_scaled, y_train, y_test, threshold, 'Ridge'))
    # ride_metrics = update_metrics(ride_metrics, model, mae, mse, rmse, mape, mape_acc, accuracy)
    #
    # model = 'Random Forest Regressor'
    # mmae, mse, rmse, mape, mape_acc, accuracy = (
    #     randomforestregressor(x_train_scaled, x_test_scaled, y_train, y_test, threshold,'RandomForest'))
    # ride_metrics = update_metrics(ride_metrics, model, mae, mse, rmse, mape, mape_acc, accuracy)

    for layers in range(2,6):
        for batch in [32, 64, 128]:
            print (f'{ride}:: Training layers: {layers}, batch size: {batch}')
            model = f'Neural Network {layers}-{batch}'
            model_name = f'NeuralNetwork_{layers}-{batch}_{ride}'
            mae, mse, rmse, mape, mape_acc, accuracy = (
                Predict_NeuralNetwork(x_train_scaled, x_test_scaled, y_train, y_test, layers=layers, epochs=50,
                                      batch_size=batch, model_name=model_name))
            ride_metrics = update_metrics(ride_metrics, model, mae, mse, rmse, mape, mape_acc, accuracy)



    metric_df = pd.DataFrame(ride_metrics)
    metric_df.to_csv(os.path.join(PROCESSED_FOLDER, f'{ride}_results.csv'))

    results = metric_df.groupby(by=['model']).mean().reset_index(drop=True)

    # results = pd.concat([results, ride_results])
    results.to_csv(os.path.join(PROCESSED_FOLDER, f'avg_model_results_{ride}.csv'))
    print(results.shape)

    return metric_df


if __name__ == "__main__":
    # This is where the training of the model will be performed

    model_data_file = os.path.join(PROCESSED_FOLDER, 'data_wth_features.csv')
    df = pd.read_csv(model_data_file)
    df.drop(columns='Date_Time', inplace=True)

    df['Date'] = pd.to_datetime(df['Date'])
    data_period = (df['Date'].max() - df['Date'].min()).days
    print (f'period of data {data_period}')

    all_results = pd.DataFrame()
    for ride in df['Ride'].unique().tolist():
        ride_df = df[df['Ride'] == ride].reset_index(drop=True)

        result_df = run_training(ride_df)
        result_df['Ride'] = ride

        all_results = pd.concat([all_results, result_df])

    all_results.reset_index(inplace=True, drop=True)
    all_results.to_csv(os.path.join(PROCESSED_FOLDER, 'all_ride_results.csv'))

    # run_5min_training(df)
    # run_daily_training(df)
    # run_training(df)
    # todo: make sure im normalising the data
    # todo: build different modelling architectures - NNs




