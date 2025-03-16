import os.path
import joblib
import pickle
import pandas as pd
from datetime import datetime, timezone
from src.settings import PROCESSED_FOLDER, MODELS_FOLDER, OUTPUT_FOLDER
from src.Processing.processing_main import(
    convert_unix_date_to_datetime, convert_datetime_to_unix_date,
    feature_yesterday, feature_past7day_average,
    feature_sametime_lastweek, feature_sametime_lastmonth,
    n_hourly_trend)


# todo: create df to generate the predictions for the next day for each ride

if __name__ == "__main__":

    prediction_date = '2024-12-31'
    model_name = 'Linear'

    min_temp = 34
    avg_temp = 37
    max_temp = 39
    precipitation = 0

    # alternate pipeline
    # - run processing on the day of predictions
    # - run the prediction on the last day of prepocessed data
    #   - modify the weather parameters
    #   - rolling metrics will have already been calculated

    # prediction_file = os.path.join(PROCESSED_FOLDER, 'prediction_template2.csv')
    # prediction_df = pd.read_csv(prediction_file)
    historical_data = os.path.join(PROCESSED_FOLDER, 'data_wth_features.csv')
    historical_df = pd.read_csv(historical_data)
    historical_df['Date_Time'] = pd.to_datetime(historical_df['Date_Time'])
    historical_df['Date'] = convert_unix_date_to_datetime(historical_df['Date'])

    # create a copy of 1 day to emulate the prediction day data
    prediction_df = historical_df[historical_df['Date'] == historical_df['Date'].max()].reset_index(drop=True)

     # update the date information
    # prediction_date = pd.to_datetime('28/01/2025')
    prediction_datetime = pd.to_datetime(prediction_date)
    # prediction_df['Date'] = prediction_date
    prediction_df['Date_Time'] = pd.to_datetime(prediction_df['Date_Time'])

    prediction_df['Date_Time'] = prediction_df['Date_Time'].apply(
        lambda x: x.replace(year=prediction_datetime.year,
                            month=prediction_datetime.month,
                            day=prediction_datetime.day))

    prediction_df['Date'] = pd.to_datetime(prediction_datetime)
    prediction_df['Day'] = pd.to_datetime(prediction_datetime).weekday()
    prediction_df['is_weekday'] = int(pd.to_datetime(prediction_datetime).weekday() < 5)


    # rolling measures
    # prediction_df['Yesterday_wait_time'] = -9
    # prediction_df['Rolling_Avg_7_Days'] = -9
    # prediction_df['LastWeek_wait_time'] = -9
    # prediction_df['LastMonth_wait_time'] = -9

    # append the prediction day to the hisorical data to include the hstorical rolling averages
    prediction_df = pd.concat([historical_df, prediction_df]).reset_index(drop=True)
    feature_df = prediction_df[['Date_Time', 'Date', 'Time', 'Ride', 'Day', 'is_weekday', 'Wait Time']]

    # todo: 'Date' field is not a common tye.
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

    feature_df['Date'] = convert_datetime_to_unix_date(feature_df['Date'])

    # prediction_df['Date'] = int(pd.to_datetime(prediction_date).timestamp())

    # extract the data for the day of prediction
    feature_df = feature_df[feature_df['Date_Time'] > prediction_datetime]
    feature_df.drop(columns=['Date_Time', 'Yesterday', 'LastWeek', 'LastMonth', 'Wait Time'], inplace=True)
    # may need to drop 'Wait Time'

    # weather from public forecasts
    feature_df['Max Temp'] = max_temp
    feature_df['Avg Temp'] = avg_temp
    feature_df['min Temp'] = min_temp
    feature_df['Precipitation'] = precipitation
    # todo: merge in current weather data

    feature_df = pd.get_dummies(feature_df, columns=['Ride'])

    # load the prefered model
    model = joblib.load(os.path.join(MODELS_FOLDER, f'{model_name}.pkl'))

    scaler = joblib.load(os.path.join(MODELS_FOLDER, 'scalar.pkl'))
    feature_df = feature_df[['Date', 'Time', 'Day', 'is_weekday', 'Max Temp', 'Avg Temp', 'min Temp',
     'Precipitation', 'Yesterday_wait_time', 'Rolling_Avg_7_Days',
     'LastWeek_wait_time', 'LastMonth_wait_time',
     'Ride_Avengers Assemble: Flight Force',
     'Ride_Buzz Lightyear Laser Blast', 'Ride_Cars Quatre Roues Rallye',
     'Ride_Cars ROAD TRIP', "Ride_Crush's Coaster", 'Ride_Indiana Jones and the Temple of Peril',
     'Ride_Les Mysteres du Nautilus', 'Ride_Orbitron', 'Ride_Phantom Manor',
     'Ride_Pirates of the Caribbean', 'Ride_Ratatouille: The Adventure',
     'Ride_Slinky Dog Zigzag Spin', 'Ride_Spider-Man W.E.B. Adventure',
     'Ride_Star Tours - The Adventure Continues',
     'Ride_Star Wars Hyperspace Mountain',
     'Ride_The Twilight Zone Tower of Terror',
     'Ride_Toy Soldiers Parachute Drop']]
    features = scaler.transform(feature_df)

    # make the predictions
    feature_df['pred_wait_time'] = model.predict(features)

    output_filename = f'predictions_{prediction_date.replace('-','')}.csv'
    feature_df.to_csv(os.path.join(OUTPUT_FOLDER, output_filename))
