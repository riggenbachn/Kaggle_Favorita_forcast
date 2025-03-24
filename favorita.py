import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_squared_log_error


sales_data = pd.read_csv(r'C:\Users\noahr\OneDrive\Documents\Kaggle\Intro to machine learning\Favorita_forcast\train.csv')
processed_data = pd.read_csv(r'C:\Users\noahr\OneDrive\Documents\Kaggle\Intro to machine learning\Favorita_forcast\preprocessed_data.csv')

#processed_data['remainder__date']=pd.to_datetime(processed_data['remainder__date'])
dropped_columns = ['cat__city_event_type_Holiday','cat__city_event_type_Transfer','cat__nation_event_type_Work Day']
processed_data = processed_data.drop(dropped_columns, axis=1)

y=sales_data[['id','date','sales']]
y=y.set_index('id')
earthquake_indicies = y[y['date']<='2016-06-30'].index
y=y.drop(earthquake_indicies)
y=y.drop(['date'],axis=1)
training_ids=y.index
X_test = processed_data[processed_data['remainder__id'] >training_ids[len(training_ids)-1]].set_index('remainder__id')
X_test=X_test.drop(['remainder__date'],axis=1)
X = processed_data.set_index('remainder__id')
X_train=X.iloc[training_ids]

print('Now starting the fit')
X_train=X_train.drop(['remainder__date'],axis=1)
favorita_model = XGBRegressor(n_jobs=4, n_estimators=100, eval_metric='rmsle',verbose=True)
y=np.log(y+1)
favorita_model.fit(X_train,y)
print('Done fitting, now starting the predictions.')
sales_predictions=favorita_model.predict(X_test)
i=0
while i<len(sales_predictions):
    sales_predictions[i]=np.exp(sales_predictions[i])-1
    sales_predictions[i]=max(sales_predictions[i],0)
    i+=1
#Now we just need to print a cvs file with these predictions
X_test=X_test.reset_index()
output={'id':X_test['remainder__id'].to_numpy(), 'sales':sales_predictions}
result=pd.DataFrame(output)
result=result.set_index('id')
print(result)
result.to_csv('favorita_sales_predictions.csv')