import os
import glob
import pandas as pd

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
    ride_wait_times['dt'] = pd.to_datetime(ride_wait_times['Date/Time']).dt.round('5T')
    ride_wait_times['Date'] = ride_wait_times['dt'].dt.date
    ride_wait_times['Time'] = ride_wait_times['dt'].dt.time

    # sort by ride name and date time
    ride_wait_times.sort_values(by=['Ride', 'Date', 'Time'], inplace=True)

    return ride_wait_times[['Ride', 'Date', 'Time', 'Wait Time']].reset_index(drop=True)

if __name__ == "__main__":

    # csv's that contain the wait times for each ride for the past 2 years
    wait_time_downloads = glob.glob(os.path.join(INPUT_FOLDER, 'download*.csv'))

    wait_time_df = processDisneyRideWaitTimes(wait_time_downloads)

    # combine with weather data
    # load weather data
    weather_file = os.path.join(INPUT_FOLDER, 'Paris weather.csv')
    weather = pd.read_csv(weather_file)
    weather['Date'] = pd.to_datetime(weather['Date'], format='%d/%m/%Y').dt.date

    # merge weather data with ride wait times
    combined_df = pd.merge(wait_time_df,
                           weather[['Date', 'Max Temp', 'Avg Temp', 'min Temp', 'Precipitation']],
                           on='Date')

    # Todo: add feature engineering

    # Save the combined DataFrame to a new CSV file
    combined_df.to_csv(os.path.join(OUTPUT_FOLDER, 'ML_input_data.csv'), index=False)




