import os.path
import pandas as pd
from datetime import datetime, timezone
from src.settings import PROCESSED_FOLDER
from src.Processing.processing_main import(
    convert_unix_date_to_datetime, convert_datetime_to_unix_date,
    feature_yesterday, feature_past7day_average,
    feature_sametime_lastweek, feature_sametime_lastmonth)


# todo: create df to generate the predictions for the next day for each ride

if __name__ == "__main__":

    # alternate pipeline
    # - run processing on the day of predictions
    # - run the prediction on the last day of preocessed data
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
    prediction_date = pd.to_datetime('2024-12-31')
    # prediction_df['Date'] = prediction_date
    prediction_df['Date_Time'] = pd.to_datetime(prediction_df['Date_Time'])

    prediction_df['Date_Time'] = prediction_df['Date_Time'].apply(
        lambda x: x.replace(year=prediction_date.year,
                            month=prediction_date.month,
                            day=prediction_date.day))

    prediction_df['Date'] = pd.to_datetime(prediction_date)
    prediction_df['Day'] = pd.to_datetime(prediction_date).weekday()
    prediction_df['is_weekday'] = int(pd.to_datetime(prediction_date).weekday() < 5)


    # rolling measures
    # prediction_df['Yesterday_wait_time'] = -9
    # prediction_df['Rolling_Avg_7_Days'] = -9
    # prediction_df['LastWeek_wait_time'] = -9
    # prediction_df['LastMonth_wait_time'] = -9

    # append the prediction day to the hisorical data to include the hstorical rolling averages
    prediction_df = pd.concat([historical_df, prediction_df]).reset_index(drop=True)
    feature_df = prediction_df[['Date_Time', 'Date', 'Time', 'Ride', 'Wait Time']]

    # todo: 'Date' field is not a common tye.
    yesterday_df = feature_yesterday(feature_df)
    past7day = feature_past7day_average(feature_df)
    df7days_ago = feature_sametime_lastweek(feature_df)
    df28days_ago = feature_sametime_lastmonth(feature_df)

    # Added all features to the dataframe
    feature_df = pd.merge(feature_df, yesterday_df, how='left', on=['Ride', 'Date_Time'])
    feature_df = pd.merge(feature_df, past7day, how='left', on=['Ride', 'Date_Time'])
    feature_df = pd.merge(feature_df, df7days_ago, how='left', on=['Ride', 'Date_Time'])
    feature_df = pd.merge(feature_df, df28days_ago, how='left', on=['Ride', 'Date_Time'])

    feature_df['Date'] = convert_datetime_to_unix_date(feature_df['Date'])

    # prediction_df['Date'] = int(pd.to_datetime(prediction_date).timestamp())

    # extract the data for the day of prediction
    feature_df = feature_df[feature_df['Date_Time'] > prediction_date]

    # weather from public forecasts
    feature_df['Max Temp'] = -9
    feature_df['Avg Temp'] = -9
    feature_df['min Temp'] = -9
    feature_df['Precipitation'] = -9
    # todo: merge in current weather data

    # load the prefered model

    # make the predictions
    # y_pred = lm.predict(x_test)

    feature_df.columns