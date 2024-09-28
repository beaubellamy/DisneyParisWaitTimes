# Workflow

## Wait Time Predictions
Under Construction

- Processing:

    - Processes the existing "downloaded" wait time files.
    - Merges this data with downloaded[ historical weather data](https://www.wunderground.com/history/monthly/fr/mauregard/LFPG/date/2024-8)

- Feature Engineering

  - generate different features that can be used to aid the training of the ML model

- Training (under construction)
  
  - training a ML model to predict the next days wait times
  

## Day of attendance
- Run `Live_Monitoring` 

    - Monitors the wait times for key rides
    - Each ride has a designated wait time threshold that we are willing to wait
    - If the wait time is below this threshold, a notification is sent to slack to provide us the opportunity to make a decision for the next ride we will go to.

