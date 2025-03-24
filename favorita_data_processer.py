import pandas as pd
import numpy as np
import datetime
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

local_file_path=r'C:\Users\noahr\OneDrive\Documents\Kaggle\Intro to machine learning\Favorita_forcast'
def create_data_frame(filename):
    slash = "\\"
    path=local_file_path+slash+filename
    result = pd.read_csv(path)
    return result

#import all of the relevant csv files into pandas dataframes.    
holiday_data=create_data_frame(r'holidays_events.csv')
oil_prices = create_data_frame(r'oil.csv')
store_data = create_data_frame(r'stores.csv')
training_data=create_data_frame(r'train.csv')
test_data = create_data_frame(r'test.csv')

#Fixing the holiday/event data. 
##We first remove the rows we will not be using
holiday_data=holiday_data[holiday_data.transferred==False]
holiday_data.drop(['description','transferred'],axis=1,inplace=True)
##We now break holiday_data into three tables which will help us build our feature data later.
city_events = holiday_data[holiday_data['locale']=='Local']
city_events=city_events.rename(columns={'type':'city_event_type'})
city_events=city_events.drop(['locale'],axis=1)
region_events = holiday_data[holiday_data['locale']=='Regional']
region_events=region_events.rename(columns={'type':'region_event_type'})
region_events = region_events.drop('locale',axis=1)
nation_events = holiday_data[holiday_data['locale']=='National']
nation_events=nation_events.drop(['locale','locale_name'],axis=1)
nation_events=nation_events.rename(columns={'type':'nation_event_type'})

#join relevant data
##first put all of the training data and testing data into one frame since all of the other tables contain information relevant to both tables.
X_train=training_data.drop('sales',axis=1, inplace=False).set_index('id')
cutoff_id=X_train.index.max()
X_test = test_data.set_index('id')
X=pd.concat([X_train,X_test])
X=X.reset_index()
## We will now add the other tables to X
###First we will add the store data to X
X=pd.merge(X, store_data,on='store_nbr', how='left')
###Now we will add the oil data
X=pd.merge(X,oil_prices, on='date',how='left')
###Now we will add the holiday_data. To do this we have three cases, either the event is local, 
X=pd.merge(X,city_events, left_on=['date','city'], right_on=['date','locale_name'],how='left')
X=X.drop('locale_name',axis=1)
X=pd.merge(X,region_events, left_on=['date','state'], right_on=['date','locale_name'],how='left')
X=X.drop('locale_name',axis=1)
X=pd.merge(X,nation_events, on='date',how='left')
X=X.drop('state',axis=1)

#Preprocess the data.
##In order to prevent data leakage we will first split our input data back into peices.
X_train=X[X['id']<=cutoff_id]
X_test=X[X['id']>cutoff_id]
##first we will fill in the missing values from the oil prices. We do this by first filling in the price with average of the nearest values
imputor = KNNImputer()
OH_encoder=OneHotEncoder(sparse_output=False)
cat_columns = [cname for cname in X.columns if X[cname].dtype == "object"]
cat_columns.pop(0)# date should not be onehotencoded.
preprocessor = ColumnTransformer(
    transformers=[
        #('date', OrdinalEncoder(),['date']),
        ('oil', imputor, ['dcoilwtico']),
        ('cat', OH_encoder, cat_columns)
    ], n_jobs=4, remainder='passthrough'
)
preprocessor.set_output(transform='pandas')
current_time = datetime.datetime.now()
print(f'Starting the preprocessing at {current_time.strftime("%H:%M:%S")}')
preprocessor.fit(X_train.tail(1000000))
current_time = datetime.datetime.now()
print(f'Now starting the real test at {current_time.strftime("%H:%M:%S")}:')
result = preprocessor.transform(X_train)
result = pd.concat([result, preprocessor.transform(X_test)])
result=result.drop(['cat__nation_event_type_nan', 'cat__region_event_type_nan','cat__city_event_type_nan'], axis=1)
current_time = datetime.datetime.now()
print(f'finished at {current_time.strftime("%H:%M:%S")}')
print(result)
result.to_csv('preprocessed_data.csv')
print(result.columns)


##We now restrict our attention to the 2013 and 2014 data to train our model.
