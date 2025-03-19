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
