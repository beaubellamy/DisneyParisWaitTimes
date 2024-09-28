import os
import pytz
import random
import yaml

from requests import get, post
from requests.exceptions import RequestException
from contextlib import closing
from bs4 import BeautifulSoup
import pandas as pd
import time as sleep_time
from datetime import datetime, time, timedelta

from src.settings import INPUT_FOLDER, OUTPUT_FOLDER
from src.utilities import (read_yaml, post_message_to_slack,
                           time_to_seconds, convert_to_float)


def is_good_response(resp):
    """
    Ensures that the response is a html object.
    """
    content_type = resp.headers['Content-Type'].lower()
    return (resp.status_code == 200 and 
            content_type is not None 
            and content_type.find('html') > -1)


def get_html_content(url, multiplier=1):
    """
    Retrieve the contents of the url, using the proxy and user 
    agent to help identify who you are.
    """

    # Responisble scraper: including multiplier to exponentially
    # increase the delay when there are several attempts at
    # connecting to the url
    randomSleep = random.uniform(2,10)
    sleep_time.sleep(randomSleep*multiplier)

    # specify the browser header
    headers = ({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36'})
   
    # Get the html from the url
    try:
        with closing(get(url, headers=headers)) as resp:

            # Check the response status
            if is_good_response(resp):
                print ("Success: ",url)
                return resp.content
            else:
                # Unable to get the url response
                return None

    except RequestException as e:
        print('Error during requests to {0} : {1}'.format(url, str(e)))


if __name__ == "__main__":

    # monitor the live wait times

    # intended to by run on the day of attendance to monitor the wait
    # times live and notify us of any dip below thresholds

    # Todo: reconfigure the ride list to maintain a ride specific threshold
    # Todo: add in check for ride threshold and send slack notification if below the threshold

    credentials = read_yaml('credentials.yml', INPUT_FOLDER)

    # master list
    Url_list = ['https://www.thrill-data.com/waits/attraction/disneyland-paris/buzzlightyearlaserblast/',
                'https://www.thrill-data.com/waits/attraction/disneyland-paris/indianajonesandthetempleofperil/'
                'https://www.thrill-data.com/waits/attraction/disneyland-paris/lesmysteresdunautilus/',
                'https://www.thrill-data.com/waits/attraction/disneyland-paris/orbitron/',
                'https://www.thrill-data.com/waits/attraction/disneyland-paris/phantommanor/',
                'https://www.thrill-data.com/waits/attraction/disneyland-paris/piratesofthecaribbean/',
                'https://www.thrill-data.com/waits/attraction/disneyland-paris/startourstheadventurescontinue/',
                'https://www.thrill-data.com/waits/attraction/disneyland-paris/starwarshyperspacemountain/',
                'https://www.thrill-data.com/waits/attraction/walt-disney-studios/avengersassembleflightforce/',
                'https://www.thrill-data.com/waits/attraction/walt-disney-studios/carsquatrerouesrallye/',
                'https://www.thrill-data.com/waits/attraction/walt-disney-studios/carsroadtrip/',
                'https://www.thrill-data.com/waits/attraction/walt-disney-studios/crushscoaster/',
                'https://www.thrill-data.com/waits/attraction/walt-disney-studios/ratatouilletheadventure/',
                'https://www.thrill-data.com/waits/attraction/walt-disney-studios/slinkydogzigzagspin/',
                'https://www.thrill-data.com/waits/attraction/walt-disney-studios/spidermanwebadventure/',
                'https://www.thrill-data.com/waits/attraction/walt-disney-studios/thetwilightzonetowerofterror/',
                'https://www.thrill-data.com/waits/attraction/walt-disney-studios/toysoldiersparachutedrop/'
                ]

    # test list
    baseUrl = ['https://www.thrill-data.com/waits/attraction/disneyland-paris/itsasmallworld/',
              'https://www.thrill-data.com/waits/attraction/disneyland-paris/buzzlightyearlaserblast/',
              'https://www.thrill-data.com/waits/attraction/disneyland-paris/indianajonesandthetempleofperil/',
              'https://www.thrill-data.com/waits/attraction/disneyland-paris/starwarshyperspacemountain/'
    ]

    # Define the Paris timezone
    paris_timezone = pytz.timezone('Europe/Paris')

    # Get the current time in Paris
    current_paris_datetime = datetime.now(paris_timezone)

    # Set the opening and closing times of the park
    opening_time = time(hour=9,minute=0,second=0)
    closing_time = time(hour=23,minute=0,second=0)

    wait_time_data = []


    if current_paris_datetime.time() < opening_time:
        # wait until the park opens
        # opening times may vary

        # Calculate the time difference - to wait
        opening_time_sec = time_to_seconds(opening_time)
        current_paris_time_sec = time_to_seconds(current_paris_datetime.time())

        time_difference = opening_time_sec - current_paris_time_sec

        hours, remainder = divmod(time_difference, 3600)
        minutes, _ = divmod(remainder, 60)
        print (f'waiting {hours}:{minutes} until the park is open')
        sleep_time.sleep(time_difference)

    # Monitor the wait times
    while opening_time < current_paris_datetime.time() < closing_time:
       print (f'scrape the wait times: {current_paris_datetime.time()}')

       for url in baseUrl:
           print (url)
           content = get_html_content(url)
           html = BeautifulSoup(content, 'html.parser')
           wait_time_div = html.find("div", {'id': "wait-menu"})
    
           data_element = {'ride': url.split('/')[-2],
                           'date': current_paris_datetime.date(),
                           'time': current_paris_datetime.time(),
                           }

           wait_div = wait_time_div.find_all('div', {'class': 'data-number'})
           wait_time = convert_to_float(wait_div[0].get_text(strip=True))
           data_element['Wait_time'] = wait_time
           data_element['daily_average'] = convert_to_float(wait_div[1].get_text(strip=True))

           wait_time_data.append(data_element)

           # if wait_time is below ride threshold

           # notify my via slack

       # Wait 5 minutes and repeat
       print ('waiting')
       sleep_time.sleep(300)
       current_paris_datetime = datetime.now(paris_timezone)

    # Create a DataFrame from the dictionary
    df = pd.DataFrame(wait_time_data)

    # Save the DataFrame to a CSV file
    paris_date = current_paris_datetime.date().strftime('%Y-%m-%d')
    csv_file = f'wait_time_{paris_date}.csv'
    df.to_csv(csv_file, index=False)

    message = f'saved wait time file for {paris_date}'
    post_message_to_slack(credentials['DISNEY_SLACK_BOT']['SLACK_CHANNEL'],
                          message,
                          credentials['DISNEY_SLACK_BOT']['SLACK_TOKEN'],
                          credentials['DISNEY_SLACK_BOT']['SLACK_URL'])
    

