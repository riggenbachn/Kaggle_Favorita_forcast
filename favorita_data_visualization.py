import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import datetime
import matplotlib.pyplot as plt
import datetime
import matplotlib.dates as mdates
local_file_path=r'C:\Users\noahr\OneDrive\Documents\Kaggle\Intro to machine learning\Favorita_forcast'

def create_data_frame(filename):
    slash = "\\"
    path=local_file_path+slash+filename
    result = pd.read_csv(path)
    return result
    
holiday_data=create_data_frame(r'holidays_events.csv')
oil_prices = create_data_frame(r'oil.csv')
oil_prices.interpolate(method='nearest',inplace=True)
oil_prices.interpolate(method='backfill',inplace=True)
store_data = create_data_frame(r'stores.csv')
training_data=create_data_frame(r'train.csv')
transaction_data=create_data_frame(r'transactions.csv')
test_data = create_data_frame(r'test.csv')

print(f"Holiday columns consist of {holiday_data.columns}\n")
print("The first few rows are:")
print(holiday_data.head())
print('The missing entries are:')
missing_holiday_data=holiday_data.isnull().sum()
print(missing_holiday_data[missing_holiday_data>0])
print(f"Oil columns consist of {oil_prices.columns}\n")
print("The first few rows are:")
print(oil_prices.head())
print('The missing entries are:')
missing_oil_data=oil_prices.isnull().sum()
print(missing_oil_data[missing_oil_data>0])
print(f"Store columns consist of {store_data.columns}")
print("The first few rows are:")
print(store_data.head())
print('The missing entries are:')
missing_store_data=store_data.isnull().sum()
print(missing_store_data[missing_store_data>0])
print(f"Training columns consist of {training_data.columns}")
print("The first few rows are:")
print(training_data)
print('The missing entries are:')
missing_training_data=training_data.isnull().sum()
print(missing_training_data[missing_training_data>0])
print(f"Transactions columns consist of {transaction_data.columns}")
print("The first few rows are:")
print(transaction_data.head())
print('The missing entries are:')
missing_transaction_data=transaction_data.isnull().sum()
print(missing_transaction_data[missing_transaction_data>0])
print(f"Test columns consist of {test_data.columns}")
print("The first few rows are:")
print(test_data)
print('The missing entries are:')
missing_test_data=test_data.isnull().sum()
print(missing_test_data[missing_test_data>0])

dates = pd.to_datetime(transaction_data.date).to_numpy()
total_sales = transaction_data['transactions'].to_numpy()
ticks_to_show = dates[::60]
plt.plot(dates, total_sales)
plt.xticks(ticks_to_show)

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator())
plt.gcf().autofmt_xdate() # Rotates dates for better readability

plt.show()