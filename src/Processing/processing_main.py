import os
import glob
import pandas as pd
# from datetime import timedelta

from src.settings import INPUT_FOLDER, OUTPUT_FOLDER


def processDisneyRideWaitTimes(file_list):

    # Initialize a list to hold the DataFrames
    dataframes = []

    # Loop through the list of files and read each one into a DataFrame
    for file in file_list:
        df = pd.read_csv(file)
        dataframes.append(df)

    # Combine all DataFrames into a single DataFrame
    ride_wait_times = pd.concat(dataframes, ignore_index=True)

    # convert 'Data/Time' to date time field and seperate into date and time field
    # round to nearest 5 min interval to keep times consistent
    ride_wait_times['Date_Time'] = pd.to_datetime(ride_wait_times['Date/Time']).dt.round('5 min')
    ride_wait_times['Date'] = ride_wait_times['Date_Time'].dt.date
    ride_wait_times['Time'] = ride_wait_times['Date_Time'].dt.time

    # sort by ride name and date time
    ride_wait_times.sort_values(by=['Ride', 'Date', 'Time'], inplace=True)

    return ride_wait_times[['Ride', 'Date_Time', 'Date', 'Time', 'Wait Time']].reset_index(drop=True)

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

    return feature_df

def feature_yesterday(df):
    return feature_previous_n_day(df,new_feature='Yesterday', n_days=1)


def feature_past7day_average(df):
    # rolling 7 day average for the specific time of the measurement

    past7day_list = []

    for t in df['Time'].unique().tolist():

        ride_data = df[(df['Ride'] == 'Avengers Assemble: Flight Force') & (df['Time'] == t)]
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

        # df2 = pd.merge(df, ride_data[['Ride', 'Date', 'Time', 'Rolling_Avg_7_Days']], on=['Ride', 'Date', 'Time'])
        # df2.to_csv(os.path.join(OUTPUT_FOLDER, f'avengers_window7_t_{t.strftime("%H%M")}.csv'))

    final_df = pd.concat(past7day_list, ignore_index=True)

    return final_df

def feature_sametime_lastweek(df):
    # measurment for the same time of day, 7 days ago
    return feature_previous_n_day(df, new_feature='LastWeek', n_days=7)


def feature_sametime_lastmonth(df):
    return feature_previous_n_day(df, new_feature='LastMonth', n_days=28)



if __name__ == "__main__":

    # csv's that contain the wait times for each ride for the past 2 years
    wait_time_downloads = glob.glob(os.path.join(INPUT_FOLDER, 'download*.csv'))

    wait_time_df = processDisneyRideWaitTimes(wait_time_downloads)

    # Todo: insert data for missing days (NaN)

    # Load weather data and combine
    weather_file = os.path.join(INPUT_FOLDER, 'Paris weather.csv')
    weather = pd.read_csv(weather_file)
    weather['Date'] = pd.to_datetime(weather['Date'], format='%d/%m/%Y').dt.date

    # merge weather data with ride wait times
    combined_df = pd.merge(wait_time_df,
                           weather[['Date', 'Max Temp', 'Avg Temp', 'min Temp', 'Precipitation']],
                           on='Date')
    combined_df['Date'] = pd.to_datetime(combined_df['Date'])

    # Todo: add feature engineering
    combined_df2 = feature_yesterday(combined_df)
    past7day = feature_past7day_average(combined_df2)
    df7days_ago = feature_sametime_lastweek(combined_df2)
    df28days_ago = feature_sametime_lastmonth(combined_df2)

    combined_df3 = pd.merge(combined_df2, past7day, how='left', on=['Ride', 'Date_Time'])

    # Save the combined DataFrame to a new CSV file
    combined_df3.to_csv(os.path.join(OUTPUT_FOLDER, 'data_wth_7day_average3.csv'), index=False)




