# Kaggle_Favorita_forcast
This repository is my solution to Kaggle's Store Sales - Time Series Forecasting machine learning competition.

Getting a handle on the data:
![image](https://github.com/user-attachments/assets/a828eed7-e399-4c4f-84e5-e79a72a00a56)
![image](https://github.com/user-attachments/assets/cf20e2c2-b3d1-42f8-85e6-96f88f3c190f)

## Model assumptions
It seems like if a holiday was transfered we should treat the original day as a normal day, and so we will remove these rows from holidays_events when building our model. Each of the other type of event seems distinct enough that they should be treated seperately, and so we will use a onehot model for these when building our model. 

There is very little missing data, we only have missing oil prices. We will back-fill these values. while this seems like a bad estimate it should be good enough for our machine learning model.

transactions contains no new information and so will not be included in the initial model.

Not clear what type in store metadata is, but including it is not too much more data and it might improve the model. Will do some statistical testing to find out if it is correlated with sales. 

Everything else seems important and worth including in our model.

Based on the graph of total sales over time:
![image](https://github.com/user-attachments/assets/dd4e8f9e-f5ba-4bec-b7ba-24c2eb35d605)
with the exception of times around December 25th the sales are relatively constant over time and so it does not make sense to try and model the (log of) the total sales as a sum of random variables given the time. We will instead use a machine learning algorithm xGBoost.

## Data pipeline

For testing we one hot encoded everything which was categorical except the date.

Since there are more than 3 million observations this caused a large bottleneck in testing and implemintation. The most obvious code took our laptop more than 8 hours to process just 1 million rows.

We fixed this by doing 2 things. Note that for the pipeline we do not need to worry about data leakage since the pipeline only involves one hot encoding and averaging local values. Thus we can fit the pipeline on a much smaller data set (we used the last 1000 rows of our feature list table) and then transfor the data acording to that fit. This sped up the process to run in less than an hour, a significant improvement but still too long to run every time we wanted to run a test. To get around this, for testing, we printed the preprocessed data to a csv file and loaded it for tests instead of processing the data every time.
