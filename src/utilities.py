import os
import yaml

from requests import post


def read_yaml(file, input_folder):

    if not file.endswith('.yml'):
        file += '.yml'
    with open(os.path.join(input_folder, file), 'r') as data_file:
        return yaml.safe_load(data_file)


def time_to_seconds(t):
    """
    Convert a datetime.time object to the number of seconds since midnight.

    :param t: A datetime.time object
    :return: The number of seconds since midnight
    """
    return t.hour * 3600 + t.minute * 60 + t.second


def post_message_to_slack(channel, message, SLACK_TOKEN, SLACK_URL):


    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {SLACK_TOKEN}'
    }

    data = {
        'channel': channel,
        'text': message
    }

    response = post(SLACK_URL, headers=headers, json=data)

    if response.status_code == 200:
        print("Message posted successfully!")
    else:
        print(f"Failed to post message: {response.status_code}, {response.text}")


