import os
import glob
import pandas as pd
import numpy as np
from datetime import timedelta

from src.settings import INPUT_FOLDER, PROCESSED_FOLDER

def convert_datetime_to_sec_since_midnight(datetime):
    return datetime.dt.hour * 3600 + datetime.dt.minute * 60 + datetime.dt.second


def convert_datetime_to_unix_date(datetime):
    date_part = datetime.dt.floor('D')
    return date_part.astype('int64') // 10 ** 9

def convert_unix_date_to_datetime(unix_timestamp):
    return pd.to_datetime(unix_timestamp, unit='s')

def processDisneyRideWaitTimes(file_list):

    # Initialize a list to hold the DataFrames
    dataframes = []

    # Loop through the list of files and read each one into a DataFrame
    for file in file_list:
        df = pd.read_csv(file)
        df['Date/Time'] = pd.to_datetime(df['Date/Time'], format='mixed').dt.strftime('%Y-%m-%d %H:%M:%S')
        dataframes.append(df)

    # Combine all DataFrames into a single DataFrame
    ride_wait_times = pd.concat(dataframes, ignore_index=True)

    # drop duplicates
    ride_wait_times.drop_duplicates(inplace=True)

    # convert 'Data/Time' to date time field and seperate into date and time field
    # round to nearest 5 min interval to keep times consistent
    ride_wait_times['Date_Time'] = pd.to_datetime(ride_wait_times['Date/Time']).dt.round('5 min')
    # ride_wait_times['Date'] = ride_wait_times['Date_Time'].dt.date
    # ride_wait_times['Time'] = ride_wait_times['Date_Time'].dt.time

    # sort by ride name and date time
    ride_wait_times.sort_values(by=['Ride', 'Date_Time'], inplace=True)

    # return ride_wait_times[['Ride', 'Date_Time', 'Date', 'Time', 'Wait Time']].reset_index(drop=True)
    return ride_wait_times[['Ride', 'Date_Time', 'Wait Time']].reset_index(drop=True)

# Ride	Date	Time	Wait Time	Max Temp	Avg Temp	min Temp	Precipitation
def feature_previous_n_day(df, new_feature='Yesterday', n_days=1):

    df_copy = df.copy()
    df[new_feature] = df['Date_Time'] - pd.Timedelta(days=n_days)

    feature_df = pd.merge(df,
                          df_copy[['Ride', 'Date_Time', 'Wait Time']],
                          how='left', left_on=['Ride', new_feature], right_on=['Ride', 'Date_Time'])

    feature_df.drop(columns=[new_feature, 'Date_Time_y'], inplace=True)
    feature_df.rename(columns={'Date_Time_x': 'Date_Time', 'Wait Time_x': 'Wait Time',
                               'Wait Time_y': f'{new_feature}_wait_time'}, inplace=True)

    return feature_df[['Ride', 'Date_Time', f'{new_feature}_wait_time']]


def feature_yesterday(df):
    return feature_previous_n_day(df,new_feature='Yesterday', n_days=1)


def feature_past7day_average(df):
    # rolling 7 day average for the specific time of the measurement

    past7day_list = []

    for ride in df['Ride'].unique().tolist():
        for ride_time in df['Time'].unique().tolist():

            ride_data = df[(df['Ride'] == ride) & (df['Time'] == ride_time)]
            ride_data = ride_data.set_index('Date')

            # Note: The shifting is to ensure we dont have a data leakage - only considering the past 7
            # days of measurements (exclusive of "today")
            # The data occassionally has missing data, so when this shifting occurs, it can bring in data
            # for a date that is outside the intended 7 day range
            # Todo: ensure the df has a measurment for everyday and every time interval (NaN), so this
            #  window function will be correct - this should be done after processDisneyRideWaitTimes()
            ride_data['Shifted_Wait_Time'] = ride_data['Wait Time'].shift(1)

            # ride_data['Rolling_Avg_7_Days'] = ride_data['Wait Time'].rolling(window=7, min_periods=1).mean()
            ride_data['Rolling_Avg_7_Days'] = ride_data['Shifted_Wait_Time'].rolling(window='7D', min_periods=1).mean()

            past7day_list.append(ride_data[['Ride', 'Date_Time', 'Rolling_Avg_7_Days']])

    # Construct the final df with all rides and resulting average 7 day wate time
    final_df = pd.concat(past7day_list, ignore_index=True)

    final_df.sort_values(by=['Ride', 'Date_Time'], inplace=True)
    final_df.reset_index(inplace=True, drop=True)

    return final_df


def feature_past7day_average(df):
    # rolling 7 day average for the specific time of the measurement
    return rolling_window(df, on_feature='Wait Time', window='7D', new_feature='Rolling_Avg_7_Days')
    # past7day_list = []
    #
    # for ride in df['Ride'].unique().tolist():
    #     for ride_time in df['Time'].unique().tolist():
    #
    #         ride_data = df[(df['Ride'] == ride) & (df['Time'] == ride_time)]
    #         ride_data = ride_data.set_index('Date')
    #
    #         # Note: The shifting is to ensure we dont have a data leakage - only considering the past 7
    #         # days of measurements (exclusive of "today")
    #         # The data occassionally has missing data, so when this shifting occurs, it can bring in data
    #         # for a date that is outside the intended 7 day range
    #         # Todo: ensure the df has a measurment for everyday and every time interval (NaN), so this
    #         #  window function will be correct - this should be done after processDisneyRideWaitTimes()
    #         ride_data['Shifted_Wait_Time'] = ride_data['Wait Time'].shift(1)
    #
    #         # ride_data['Rolling_Avg_7_Days'] = ride_data['Wait Time'].rolling(window=7, min_periods=1).mean()
    #         ride_data['Rolling_Avg_7_Days'] = ride_data['Shifted_Wait_Time'].rolling(window='7D', min_periods=1).mean()
    #
    #         past7day_list.append(ride_data[['Ride', 'Date_Time', 'Rolling_Avg_7_Days']])
    #
    # # Construct the final df with all rides and resulting average 7 day wate time
    # final_df = pd.concat(past7day_list, ignore_index=True)
    #
    # final_df.sort_values(by=['Ride', 'Date_Time'], inplace=True)
    # final_df.reset_index(inplace=True, drop=True)
    #
    # return final_df


def feature_sametime_lastweek(df):
    # measurment for the same time of day, 7 days ago
    return feature_previous_n_day(df, new_feature='LastWeek', n_days=7)


def feature_sametime_lastmonth(df):
    # measurment for the same time of day, 4 weeks ago
    return feature_previous_n_day(df, new_feature='LastMonth', n_days=28)


def n_hourly_trend(df, hours=1, new_feature='Rolling_28D_hr_trend'):

    n_rows = hours * 12  # convert hours to n rows
    # Constant 5 min interval (12 rows) between each row, except when the date and/or ride changes
    rise = df['Wait Time'] - df['Wait Time'].shift(n_rows)
    run = hours * 60    # minutes
    slope = rise/run
    df[f'{hours}_hourly_trend'] = slope
    mask = (df['Date'] == df['Date'].shift(n_rows)) & (df['Ride'] == df['Ride'].shift(n_rows))
    df.loc[~mask, f'{hours}_hourly_trend'] = 0
    # calcaulte the average slopw for the last month - assing it to the current time
    # return df[['Date_Time', 'Ride', f'{hours}_hourly_trend']]
    new_df = rolling_window(df, on_feature=f'{hours}_hourly_trend', window='28D', new_feature=new_feature)
    # check that only data time, ride and new feature is returned
    return new_df

def resample_data(df):

    # ensure data has consistent intervals
    min_date = min(df['Date_Time'].dt.date)
    max_date = max(df['Date_Time'].dt.date)
    # set the data interval
    reference_data_time = pd.date_range(start=min_date, end=max_date, freq='5 min')

    reference = pd.DataFrame(reference_data_time)
    reference.rename(columns={0: 'Date_Time'}, inplace=True)
    reference['Time'] = reference['Date_Time'].dt.time
    # Only interested in times when the park was open
    reference = reference[(reference['Time'] > pd.to_datetime('08:00:00').time()) &
                          (reference['Time'] < pd.to_datetime('20:00:00').time())]

    # resample the data for each ride
    resampled_list = []
    for ride in df['Ride'].unique().tolist():
        ride_df = df[df['Ride'] == ride]
        ride_df = pd.merge(reference['Date_Time'],
                           ride_df[['Ride', 'Date_Time', 'Wait Time']],
                           on=['Date_Time'], how='left')

        # Assume missing data was a result of ride closure
        ride_df['Ride'] = ride_df['Ride'].ffill()
        ride_df['Ride'] = ride_df['Ride'].bfill()
        ride_df['Wait Time'] = ride_df['Wait Time'].fillna(value=0)

        resampled_list.append(ride_df)

    # Combine all DataFrames into a single DataFrame
    resampled_df = pd.concat(resampled_list, ignore_index=True)

    # drop duplicates - some files contain multiple values for the resampled time
    resampled_df.sort_values(by=['Ride', 'Date_Time', 'Wait Time'], inplace=True)
    resampled_df.drop_duplicates(subset=['Date_Time', 'Ride'], keep='last', inplace=True)


    return  resampled_df.reset_index(drop=True)


if __name__ == "__main__":

    # https://www.thrill-data.com/users/login
    # csv's that contain the wait times for each ride for the past 2 years
    wait_time_downloads = glob.glob(os.path.join(INPUT_FOLDER, 'download*.csv'))

    wait_time_df = processDisneyRideWaitTimes(wait_time_downloads)

    # Resample data and fill in missing measurements
    wait_time_df = resample_data(wait_time_df)

    # Load weather data and combine
    weather_file = os.path.join(INPUT_FOLDER, 'Paris weather.csv')
    weather = pd.read_csv(weather_file)
    weather['Date'] = pd.to_datetime(weather['Date'], format='%d/%m/%Y').dt.date

    # Additional feature engineering
    wait_time_df['Date'] = wait_time_df['Date_Time'].dt.date
    wait_time_df['Time'] = wait_time_df['Date_Time'].dt.time
    wait_time_df['Day'] = pd.to_datetime(wait_time_df['Date']).dt.weekday
    wait_time_df['is_weekday'] = (pd.to_datetime(wait_time_df['Date']).dt.weekday < 5).astype(int)
    # combined_df['PublicHoliday'] = combined_df['Date_Time'].dt.time

    # merge weather data with ride wait times
    combined_df = pd.merge(wait_time_df,
                           weather[['Date', 'Max Temp', 'Avg Temp', 'min Temp', 'Precipitation']],
                           on='Date')
    combined_df['Date'] = pd.to_datetime(combined_df['Date'])

    yesterday_df = feature_yesterday(combined_df)
    past7day = feature_past7day_average(combined_df)
    df7days_ago = feature_sametime_lastweek(combined_df)
    df28days_ago = feature_sametime_lastmonth(combined_df)
    hourly_trend_df = n_hourly_trend(combined_df)
    three_hourly_trend_df = n_hourly_trend(combined_df, hours=3, new_feature='Rolling_28D_3hr_trend')

    # Added all features to the dataframe
    combined_df2 = pd.merge(combined_df, yesterday_df, how='left', on=['Ride', 'Date_Time'])
    combined_df2 = pd.merge(combined_df2, past7day, how='left', on=['Ride', 'Date_Time'])
    combined_df2 = pd.merge(combined_df2, df7days_ago, how='left', on=['Ride', 'Date_Time'])
    combined_df2 = pd.merge(combined_df2, df28days_ago, how='left', on=['Ride', 'Date_Time'])
    combined_df2 = pd.merge(combined_df2, hourly_trend_df, how='left', on=['Ride', 'Date_Time'])
    combined_df2 = pd.merge(combined_df2, three_hourly_trend_df, how='left', on=['Ride', 'Date_Time'])

    # Ignore the first month of data - mostly null values for new features
    combined_df3 = combined_df2[combined_df2['Date'] > '2023-02-01']
    combined_df3.reset_index(inplace=True, drop=True)

    # interpolate the missing data
    # combined_df3.interpolate(method='linear', inplace=True)

    # Convert data types for modelling
    # combined_df3['Time2'] = combined_df3['Time'].map(lambda x: convert_datetime_sec_since_midnight(x))
    # combined_df3['Date'] = convert_datetime_to_unix_date(combined_df3['Date_Time'])
    combined_df3['Time'] = convert_datetime_to_sec_since_midnight(combined_df3['Date_Time'])
    # additional cyclic features
    combined_df3['Day_sin'] = np.sin(2 * np.pi * combined_df3['Day'] / 7)
    combined_df3['Day_cos'] = np.cos(2 * np.pi * combined_df3['Day'] / 7)
    combined_df3['Time_sin'] = np.sin(2 * np.pi * combined_df3['Time'] / 1440)
    combined_df3['Time_cos'] = np.cos(2 * np.pi * combined_df3['Time'] / 1440)
    combined_df3.drop(columns=['Date_Time', 'Day', 'Time'], inplace=True)

    combined_df3.drop(columns=['Yesterday', 'LastWeek', 'LastMonth'], inplace=True)
    # Save the combined DataFrame to a new CSV file
    combined_df3.to_csv(os.path.join(PROCESSED_FOLDER, 'data_wth_features.csv'), index=False)




