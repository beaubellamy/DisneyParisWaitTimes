from requests import get
from requests.exceptions import RequestException
from contextlib import closing
from bs4 import BeautifulSoup
import pandas as pd
import time as sleep_time
from datetime import datetime, time, timedelta

import pytz
import random

from twilio.rest import Client


def time_to_seconds(t):
    """
    Convert a datetime.time object to the number of seconds since midnight.

    :param t: A datetime.time object
    :return: The number of seconds since midnight
    """
    return t.hour * 3600 + t.minute * 60 + t.second



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

    # Be a responisble scraper.
    # The multiplier is used to exponentially increase the delay when 
    # there are several attempts at connecting to the url
    randomSleep = random.uniform(2,10)
    sleep_time.sleep(randomSleep*multiplier)

    #Choose the next proxy in the list
    #proxy = next(proxy_pool)
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
  


def test_sms():
    

    ## Your Account Sid and Auth Token from twilio.com/console
    account_sid = ''
    auth_token = ''
    client = Client(account_sid, auth_token)

    message = client.messages \
                    .create(
                         body="Join Earth's mightiest heroes. Like Kevin Bacon.",
                         from_='+61 488 853 988',
                         to='+61 414 916 458'
                     )

    print(message.sid)


        

if __name__ == "__main__":

    # test_sms()

    baseUrl = ['https://www.thrill-data.com/waits/attraction/disneyland-paris/itsasmallworld/',
              'https://www.thrill-data.com/waits/attraction/disneyland-paris/buzzlightyearlaserblast/',
              'https://www.thrill-data.com/waits/attraction/disneyland-paris/indianajonesandthetempleofperil/',
              'https://www.thrill-data.com/waits/attraction/disneyland-paris/starwarshyperspacemountain/'
    ]

    # Define the Paris timezone
    paris_timezone = pytz.timezone('Europe/Paris')

    # Get the current time in Paris
    current_paris_datetime = datetime.now(paris_timezone)

    opening_time = time(hour=8,minute=30,second=0)
    closing_time = time(hour=11,minute=5,second=0)

    wait_time_data = []


    if current_paris_datetime.time() < opening_time:
        # wait until the park opens

        # Calculate the time difference
        opening_time_sec = time_to_seconds(opening_time)
        current_paris_time_sec = time_to_seconds(current_paris_datetime.time())

        time_difference = opening_time_sec - current_paris_time_sec

        hours, remainder = divmod(time_difference, 3600)
        minutes, _ = divmod(remainder, 60)
        print (f'waiting {hours}:{minutes} until the park is open')
        sleep_time.sleep(time_difference)


    # if current_paris_datetime.time() > closing_time:
    #     # wait until the park opens tomorrow
    #
    #     # Calculate tomorrow's date
    #     tomorrow = current_paris_datetime.date() + timedelta(days=1)
    #
    #     # Create a datetime object for the opening time tomorrow
    #     opening_datetime_tomorrow = datetime.combine(tomorrow, opening_time, tzinfo=paris_timezone)
    #
    #     # Calculate the time difference
    #     time_difference = opening_datetime_tomorrow - current_paris_datetime
    #
    #     # Optional: Convert time_difference to hours and minutes
    #     hours, remainder = divmod(time_difference.seconds, 3600)
    #     minutes, _ = divmod(remainder, 60)
    #     print(f"Time until opening: {time_difference.days} days, {hours} hours, {minutes} minutes")

    while opening_time < current_paris_datetime.time() < closing_time:
       print (f'scrape the wait times: {current_paris_datetime.time()}')
       for url in baseUrl:
           print (url)
           content = get_html_content(url)
           html = BeautifulSoup(content, 'html.parser')
           wait_time_div = html.find("div", {'id': "wait-menu"})
    
           data_element = {'ride': url,
                           'date': current_paris_datetime.date(),
                           'time': current_paris_datetime.time(),
                           }

           wait_div = wait_time_div.find_all('div', {'class': 'data-number'})
           data_element['Wait_time'] = float(wait_div[0].get_text(strip=True))
           data_element['daily_average'] = float(wait_div[1].get_text(strip=True))

           wait_time_data.append(data_element)

       # Wait 5 minutes and repeat
       print ('waiting')
       sleep_time.sleep(60)
       current_paris_datetime = datetime.now(paris_timezone)
        


    ## Create a DataFrame from the dictionary
    #df = pd.DataFrame(wait_time_data)

    ## Save the DataFrame to a CSV file
    #csv_file = 'wait_time.csv'
    #df.to_csv(csv_file, index=False)
    

