import os.path
import joblib
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone
from src.settings import PROCESSED_FOLDER, MODELS_FOLDER, OUTPUT_FOLDER
from src.Processing.processing_main import(
    convert_datetime_to_min_since_midnight,
    convert_unix_date_to_datetime, convert_datetime_to_unix_date,
    feature_yesterday, feature_past7day_average,
    feature_sametime_lastweek, feature_sametime_lastmonth,
    n_hourly_trend)


def create_prediction_data(historical_df):
    # create a copy of 1 day to emulate the prediction day data
    prediction_df = historical_df[historical_df['Date'] == historical_df['Date'].max()].reset_index(drop=True)

    # update the date information
    # prediction_date = pd.to_datetime('28/01/2025')
    # prediction_datetime = pd.to_datetime(prediction_date)
    prediction_df['Date'] = prediction_datetime.date()
    prediction_df['Date_Time'] = pd.to_datetime(prediction_df['Date_Time'])

    prediction_df['Date_Time'] = prediction_df['Date_Time'].apply(
        lambda x: x.replace(year=prediction_datetime.year,
                            month=prediction_datetime.month,
                            day=prediction_datetime.day))

    # prediction_df['Time'] = prediction_df['Date_Time'].dt.time
    # # prediction_df['Date'] = pd.to_datetime(prediction_datetime)
    # prediction_df['Day'] = prediction_datetime.weekday()
    # prediction_df['is_weekday'] = int(prediction_datetime.weekday() < 5)

    # rolling measures
    # prediction_df['Yesterday_wait_time'] = -9
    # prediction_df['Rolling_Avg_7_Days'] = -9
    # prediction_df['LastWeek_wait_time'] = -9
    # prediction_df['LastMonth_wait_time'] = -9

    # append the prediction day to the hisorical data to include the hstorical rolling averages
    prediction_df = pd.concat([historical_df, prediction_df]).reset_index(drop=True)
    prediction_df['is_weekday'] = (prediction_df['Date_Time'].dt.weekday < 5).astype(int)

    return prediction_df


def build_prediction_features(prediction_df):
    feature_df = prediction_df[['Date_Time', 'Date', 'Ride', 'is_weekday', 'Wait Time']]
    # feature_df = prediction_df[['Date', 'Time', 'Ride', 'Day', 'is_weekday', 'Wait Time']]
    # convert back to time, or process the cycle before calcaulting the new features in processing step
    feature_df['Date'] = pd.to_datetime(feature_df['Date'])
    feature_df['Date_Time'] = pd.to_datetime(feature_df['Date_Time'])
    feature_df['Time'] = feature_df['Date_Time'].dt.time
    feature_df['Day'] = feature_df['Date_Time'].dt.weekday

    feature_df['Ride_closed'] = 0
    feature_df.loc[feature_df['Wait Time'] == 0, 'Ride_closed'] = 1

    yesterday_df = feature_yesterday(feature_df)
    past7day = feature_past7day_average(feature_df)
    df7days_ago = feature_sametime_lastweek(feature_df)
    df28days_ago = feature_sametime_lastmonth(feature_df)
    hourly_trend_df = n_hourly_trend(feature_df)
    three_hourly_trend_df = n_hourly_trend(feature_df, hours=3, new_feature='Rolling_28D_3hr_trend')

    # Added all features to the dataframe
    feature_df = pd.merge(feature_df, yesterday_df, how='left', on=['Ride', 'Date_Time'])
    feature_df = pd.merge(feature_df, past7day, how='left', on=['Ride', 'Date_Time'])
    feature_df = pd.merge(feature_df, df7days_ago, how='left', on=['Ride', 'Date_Time'])
    feature_df = pd.merge(feature_df, df28days_ago, how='left', on=['Ride', 'Date_Time'])
    feature_df = pd.merge(feature_df, hourly_trend_df, how='left', on=['Ride', 'Date_Time'])
    feature_df = pd.merge(feature_df, three_hourly_trend_df, how='left', on=['Ride', 'Date_Time'])

    return feature_df


def add_time_feaures(feature_df):
    feature_df['Time'] = convert_datetime_to_min_since_midnight(feature_df['Time'])

    feature_df['Day_sin'] = np.sin(2 * np.pi * feature_df['Day'] / 7).round(5)
    feature_df['Day_cos'] = np.cos(2 * np.pi * feature_df['Day'] / 7).round(5)
    feature_df['Time_sin'] = np.sin(2 * np.pi * feature_df['Time'] / 1440).round(5)
    feature_df['Time_cos'] = np.cos(2 * np.pi * feature_df['Time'] / 1440).round(5)

    feature_df.drop(columns=['Date', 'Day', 'Time'], inplace=True)

    return feature_df

def predict_ride_wait_times(feature_df, model_name):
    # feature_df = pd.get_dummies(feature_df, columns=['Ride'])
    # Loop through each ride and load the required model
    prediction = pd.DataFrame()
    for ride in feature_df['Ride'].unique().tolist():
        ride_label = ride.replace(':', '').replace(' ', '_')

        # load the prefered model
        model = joblib.load(os.path.join(MODELS_FOLDER, f'{model_name}_{ride_label}.pkl'))
        scaler = joblib.load(os.path.join(MODELS_FOLDER, f'{ride_label}_scalar.pkl'))

        prediction_df = feature_df[feature_df['Ride'] == ride]
        prediction_df = prediction_df[['is_weekday', 'Ride_closed', 'Max Temp', 'Avg Temp',
                                       'Yesterday_wait_time', 'Rolling_Avg_7_Days',
                                       'LastWeek_wait_time', 'LastMonth_wait_time',
                                       'Rolling_28D_hr_trend', 'Rolling_28D_3hr_trend',
                                       'Day_sin', 'Day_cos', 'Time_sin', 'Time_cos']]

        # 'Ride_Avengers Assemble: Flight Force',
        # 'Ride_Buzz Lightyear Laser Blast', 'Ride_Cars Quatre Roues Rallye',
        # 'Ride_Cars ROAD TRIP', "Ride_Crush's Coaster", 'Ride_Indiana Jones and the Temple of Peril',
        # 'Ride_Les Mysteres du Nautilus', 'Ride_Orbitron', 'Ride_Phantom Manor',
        # 'Ride_Pirates of the Caribbean', 'Ride_Ratatouille: The Adventure',
        # 'Ride_Slinky Dog Zigzag Spin', 'Ride_Spider-Man W.E.B. Adventure',
        # 'Ride_Star Tours - The Adventure Continues',
        # 'Ride_Star Wars Hyperspace Mountain',
        # 'Ride_The Twilight Zone Tower of Terror',
        # 'Ride_Toy Soldiers Parachute Drop']]
        features = scaler.transform(prediction_df)

        # make the predictions
        prediction_df['pred_wait_time'] = model.predict(features)
        prediction_df['Ride'] = ride

        prediction = pd.concat([prediction, prediction_df])

    return prediction


def plot_wait_times(df, prediction_date):
    """
    Visualizes the actual vs predicted wait times for each ride throughout the day.

    Parameters:
    df (pd.DataFrame): A DataFrame containing columns: ['timestamp', 'ride', 'actual_wait', 'predicted_wait']
    """
    rides = df['Ride'].unique()

    # plt.figure(figsize=(12, 6 ))

    for i, ride in enumerate(rides, 1):
        ride_data = df[df['Ride'] == ride]
        ride_data = ride_data.sort_values(by='Time')

        plt.figure(figsize=(12, 6))
        plt.plot(ride_data['Time'], ride_data['Wait Time'], label='Actual Wait Time', marker='o')
        plt.plot(ride_data['Time'], ride_data['pred_wait_time'], label='Predicted Wait Time', linestyle='--',
                 marker='x')

        plt.xlabel('Time')
        plt.ylabel('Wait Time (minutes)')
        plt.title(f'Wait Times for {ride}')
        plt.suptitle(f'{prediction_date.strftime(format='%d-%m-%Y')}')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        filename = f'{ride.replace(" ", "_").replace(":", "")}_{prediction_date.strftime(format="%Y%m%d")}.png'
        plt.savefig(os.path.join(OUTPUT_FOLDER, filename))
        plt.close()

    return

if __name__ == "__main__":

    prediction_date = '2024-12-31'
    prediction_datetime = pd.to_datetime(prediction_date)
    # model_name = 'Linear'
    model_name = 'NeuralNetwork_2-32'

    # Weather data for the day
    # min_temp = 34
    avg_temp = 37
    max_temp = 39
    # precipitation = 0

    # alternate pipeline
    # - run processing on the day of predictions
    # - run the prediction on the last day of prepocessed data
    #   - modify the weather parameters
    #   - rolling metrics will have already been calculated

    historical_data = os.path.join(PROCESSED_FOLDER, 'data_wth_features.csv')
    historical_df = pd.read_csv(historical_data)
    historical_df['Date_Time'] = pd.to_datetime(historical_df['Date_Time'])

    validation_df = historical_df[historical_df['Date'] == prediction_date]
    historical_df = historical_df[historical_df['Date'] < prediction_date]

    prediction_df = create_prediction_data(historical_df)

    feature_df = build_prediction_features(prediction_df)

    # extract the data for the day of prediction
    feature_df = feature_df[feature_df['Date_Time'] > prediction_datetime]
    feature_df.drop(columns=['Date_Time', 'Yesterday', 'LastWeek', 'LastMonth', 'Wait Time'], inplace=True)
    # may need to drop 'Wait Time'

    # weather from public forecasts
    feature_df['Max Temp'] = max_temp
    feature_df['Avg Temp'] = avg_temp

    feature_df = add_time_feaures(feature_df)
    feature_df.reset_index(inplace=True, drop=True)
    prediction = predict_ride_wait_times(feature_df, model_name)
    validation_df[['Day_sin', 'Day_cos', 'Time_sin', 'Time_cos']] = validation_df[
        ['Day_sin', 'Day_cos', 'Time_sin', 'Time_cos']].round(5)

    prediction = pd.merge(prediction, validation_df[['Ride', 'Day_sin', 'Day_cos', 'Time_sin', 'Time_cos', 'Wait Time']],
                   on=['Ride', 'Day_sin', 'Day_cos', 'Time_sin', 'Time_cos'])

    prediction['Time'] = (np.arctan2(prediction['Time_sin'], prediction['Time_cos']) / (2 * np.pi)) * 1440
    prediction['Time'] %= 1440

    output_filename = f'predictions_{prediction_date.replace('-','')}.csv'
    prediction.to_csv(os.path.join(OUTPUT_FOLDER, output_filename))

    # todo: create visualisations of each ride.
    # for loop over each ride
    # ride = prediction[prediction['Ride'] == 'Avengers Assemble: Flight Force']
    plot_wait_times(prediction, prediction_datetime.date())