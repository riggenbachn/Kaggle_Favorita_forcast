import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

sales_data = pd.read_csv(r'C:\Users\noahr\OneDrive\Documents\Kaggle\Intro to machine learning\Favorita_forcast\train.csv')
processed_data = pd.read_csv(r'C:\Users\noahr\OneDrive\Documents\Kaggle\Intro to machine learning\Favorita_forcast\preprocessed_data.csv')

#processed_data['remainder__date']=pd.to_datetime(processed_data['remainder__date'])
#dc = ['cat__city_event_type_Holiday','cat__city_event_type_Transfer','cat__nation_event_type_Work Day']
#processed_data = processed_data.drop(dc, axis=1)

y=sales_data[['id','sales']]
y=y.set_index('id')
training_ids=y.index
X = processed_data.set_index('remainder__id')
X=X.iloc[training_ids]

X_train = X[X['remainder__date'].between("2013-01-01","2015-01-01")]
X_val = X_train[X_train['remainder__date'].between("2014-01-01","2014-12-31")]
X_train=X_train[X_train['remainder__date'].between("2013-01-01","2013-12-31")]
training_ids=X_train.index
val_ids = X_val.index
y_train = y
y_val = np.log(y_train.iloc[val_ids].reset_index()['sales']+1)
y_train=np.log(y_train.iloc[training_ids].reset_index()['sales']+1)
favorita_model=XGBRegressor(device='cuda')
X_train=X_train.drop(['remainder__date'],axis=1)
X_val=X_val.drop(['remainder__date'],axis=1)

model = XGBRegressor(random_state=0, device='cuda', n_estimators=500, early_stopping_rounds=5,eval_metric='rmse')
model.fit(X_train, y_train, eval_set=[(X_val, y_val)],verbose=False)
sales_predictions=model.predict(X_val)
model_mse = mean_squared_error(sales_predictions,y_val)
print(model_mse)