# Workflow

## Wait Time Predictions
Under Construction

- Processing:
    - Processes the existing "downloaded" wait time files.
    - [More data can be found here](https://www.thrill-data.com/users/login)
    - Merges this data with downloaded [historical weather data](https://www.wunderground.com/history/monthly/fr/mauregard/LFPG/date/2024-8)

- Feature Engineering
  - generate different features that can be used to aid the training of the ML model

- Training
  - training a ML model to predict the next days wait times

- Predictions
  - Select the prefered model and run `model_predictions` to produce the next days results
    - Must set the required date.

- Visualisation (under construction)
  - produce visualisation that can be viewed while at the park to:
    1. test predictions in live environment
    2. assess which rides we should be targeting throughout the day

### Workflow
1. `processing_main`
2. `model_training` - identify the preferred model for predictions
3. `model_predictions`
4. `visualise_predictions` - not yet constructed


## Day of attendance
- Run `Live_Monitoring` 

    - Monitors the wait times for key rides
    - Each ride has a designated wait time threshold that we are willing to wait
    - If the wait time is below this threshold, a notification is sent to slack to provide us the opportunity to make a decision for the next ride we will go to.

