import pandas as pd
import numpy as np
import datetime

sales_data = pd.read_csv(r'C:\Users\noahr\OneDrive\Documents\Kaggle\Intro to machine learning\Favorita_forcast\train.csv')
processed_data = pd.read_csv(r'C:\Users\noahr\OneDrive\Documents\Kaggle\Intro to machine learning\Favorita_forcast\preprocessed_data.csv')

#processed_data['remainder__date']=pd.to_datetime(processed_data['remainder__date'])
processed_data=processed_data.drop(['remainder__date'],axis=1)

y=sales_data[['id','sales']]
y=y.set_index('id')
training_ids=y.index
X = processed_data.set_index('remainder__id')
X=X.iloc[training_ids].reset_index()
columns = X.columns

for i in columns:
    mean=X[i].mean()
    X[i]=X[i]-mean
y=y['sales']-y['sales'].mean()
y=np.log(y)
y_std=y.std()
for x in columns:
    size=X[x].count()
    covariance = (X[x]*y).sum()/(size-1)
    current_std=X[x].std()
    correlation=covariance/(y_std*current_std)
    t_statistic=correlation*np.sqrt((size-2)/(1-(correlation**2)))
    #print(f"The t-statistic of {x} being correlated with sales is: {t_statistic}")
    if (t_statistic<3) and (t_statistic >-3):
        print(x)