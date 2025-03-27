# Kaggle_Favorita_forcast
This repository is my solution to Kaggle's Store Sales - Time Series Forecasting machine learning competition. There are 12 files, 5 python files and 7 csv files. 

The following is a printout of some of the basic details of the csv files which we are given by Kaggle:
![image](https://github.com/user-attachments/assets/a828eed7-e399-4c4f-84e5-e79a72a00a56)
![image](https://github.com/user-attachments/assets/cf20e2c2-b3d1-42f8-85e6-96f88f3c190f)

## Model assumptions
It seems like if a holiday was transfered we should treat the original day as a normal day, and so we will remove these rows from the table when building our model. Each of the other type of event seems distinct enough that they should be treated seperately, and so we will use a one-hot encoder for these when building our model. 

There is very little missing data, we only have missing oil prices. We will average the nearest 5 prices in order to fill in the missing values. while this seems like a rough estimate it should be good enough for our machine learning model.

transactions contains no new information and so will not be included in the initial model.

It is not clear what type in store metadata is, but including it is not too much more data and it might improve the model. Upon running a t-test there is a statistically signifigant correlation between these values and sales, so we will include these in our model as one-hot encoded data. 

Everything else seems important and worth including in our model.

Based on the graph of total sales over time:
![image](https://github.com/user-attachments/assets/dd4e8f9e-f5ba-4bec-b7ba-24c2eb35d605)
with the exception of times around December 25th the sales are relatively constant over time and so it does not make sense to try and model the (log of) the total sales as a sum of random variables given the time. We will instead use a machine learning algorithm xGBoost.

## Data pipeline

For testing we one-hot encoded everything which was categorical except the date.

Since there are more than 3 million observations this caused a large bottleneck in testing and implemintation. The most obvious code took our laptop more than an hour to process just 1 million rows, and runing a fit on the entire table was still running after 24 hours when we canceled the code and started looking for optimizations. 

We fixed this by doing 2 things. Note that for the pipeline we do not need to worry about data leakage since the pipeline only involves one hot encoding and averaging local values. Thus we can fit the pipeline on a smaller data set (we used the last 1 million rows of our feature list table) and then transform the data according to that fit. This sped up the process to run in 9 hours, a significant improvement but still too long to run every time we wanted to run a test. To get around this, for testing, we printed the preprocessed data to a csv file and loaded it for tests instead of processing the data every time.

## Statistics
Before running and optimizing our solution we want to first run t-tests to see which of our potential features are in fact relevant and worth including in our model. We use the favorita_statistics.py code to output the t-statistics which result in the following:
![image](https://github.com/user-attachments/assets/966abefa-e888-4c27-a564-9ca097ea4448)
![image](https://github.com/user-attachments/assets/44ba640f-6e5e-403e-80c8-ebadbee0b1a3)\
From this we see that city holidays, city transfers, national holidays, and national work days have very little linear effect on the total sales. This however seems at odds with the graph of total sales which spike around December 25th. It seems more likely that there is a linear relationship between these columns and the logarithm of total sales, which we compute the t-statistics of below:
![image](https://github.com/user-attachments/assets/bfd302c8-b337-466f-b493-5238078f2f09)
![image](https://github.com/user-attachments/assets/9291e1ee-2e37-4c33-8748-eb52de79ee3f)
We still see that city holidays, city transfer days, and national work days do not have a statistically signifigant correlation on the logarithm of total sales, but that national holidays do. This also suggests that we should train our model on the logarithm of sales and then exponentiate our results.

## Model Specifications
We now turn to using XGBoost. Without setting any training parameters we find that the model trained on 2014 data has a root mean squared error of 2.16773890287724. This leaves a lot of room for improvement. One source of imporvements could come from the number of estimators, but we see from the following graph of estimators vs log mean squared error that 100 is optimal:
![image](https://github.com/user-attachments/assets/61cb9d9e-7a81-4931-a59d-39f6dda00fbc)

We also removed the sales data from April 16, 2016 through June 30, 2016 since this data is effected by the earthquake.

## Results
Using the above we run XGBRegressor to the training data an print to the file favorita_sales_predictions.csv our predicted sales numbers. Upon submitting we find that the root mean squared log error of our prediction as compared to the real world data is 1.089.
![image](https://github.com/user-attachments/assets/195cf0ed-38c2-451e-98bc-f7914fad2886)

