import importlib
from datetime import timedelta, datetime

import dateutil
import pandas as pd
from sklearn.model_selection import train_test_split

import preprocessing as pr

pd.options.mode.chained_assignment = None  # to avoid the false postive warning of SettingWithCopyWarning:

def check_timeindex_subscribers(df):
    diff = (df.index[-1] - df.index[0])
    days = diff.days
    if(days == len(df)):

        return "All date are ok"
    else:
        range = pd.date_range(df.index[0], df.index[-1], freq='D')
        diff_range = [x for x in range if x not in df.index]
        return pr.pd.Series(diff_range)

dict_mont={
        'janvier' : 1,
        'fevrier' : 2,
        'mars' : 3,
        'avril' : 4,
        'mai' : 5,
        'juin' : 6,
        'juillet' : 7,
        'aout' : 8,
        'septembre' : 9,
        'octobre' : 10,
        'novembre' : 11,
        'decembre' : 12
}

def convert_to_date(x):
    day = x.split(" ")[1]
    month = dict_mont[x.split(" ")[2].lower()]
    year = x.split(" ")[3]
    import datetime
    date = datetime.date(int(year), int(month),int(day))
    return date

subs = pd.read_csv("subscribers/consomationBCMC.csv",sep=";")
subs["DateTime"] = subs["Date"].apply(convert_to_date)
subs['DateTime'] = pd.to_datetime(subs['DateTime'])
subs.set_index(subs["DateTime"],inplace=True)
subs.drop(["DateTime","Date","Moyenne consommation semaine MC","Moyenne consommation semaine BC"],axis=1,inplace=True)
subs["Energie consommee MC[kWh]"] = (subs["Energie consommee MC[kWh]"].str.split()).apply(lambda x: float(x[0].replace(',', '.')))
subs["Energie Consommee Total [kWh]"] = (subs["Energie Consommee Total [kWh]"].str.split()).apply(lambda x: float(x[0].replace(',', '.')))
subs["Energie consommee BC[kWh]"] = (subs["Energie consommee BC[kWh]"].str.split()).apply(lambda x: float(x[0].replace(',', '.')))

check_timeindex_subscribers(subs)


subs_BC = subs.drop_duplicates(['NB BC'], keep='last')
subs_MC = subs.drop_duplicates(['NB MC'], keep='last')
# pr.plot_data(subs_BC,subs_BC.columns,4,0)


data_site_hourly = pr.get_all_data_site()
# FOR BC
features_hourly_BC = ["Date", pr.PowerServiceBC]
data_prepared_hourly_BC = pr.prepare_date(data_site_hourly, features_hourly_BC)
data_cleaned_hourly_BC = pr.clean_data(data_prepared_hourly_BC, pr.PowerServiceBC)
pr.check_timeindex(data_cleaned_hourly_BC)
data_merged = data_cleaned_hourly_BC.join(subs['NB BC'], how='outer')
merged_backfill_BC = data_merged.fillna(method='backfill')

#FOR MC
features_hourly_MC = ["Date", pr.PowerServiceMC]
data_prepared_hourly_MC = pr.prepare_date(data_site_hourly, features_hourly_MC)
data_cleaned_hourly_MC = pr.clean_data(data_prepared_hourly_MC, pr.PowerServiceMC)
pr.check_timeindex(data_cleaned_hourly_MC)
data_merged = data_cleaned_hourly_MC.join(subs['NB MC'], how='outer')
data_merged = data_cleaned_hourly_MC.join(subs['NB MC'], how='outer')
merged_backfill_MC = data_merged.fillna(method='backfill')

pr.plot_data(merged_backfill_BC,merged_backfill_BC.columns,len(merged_backfill_BC.columns),0)
pr.plot_data(merged_backfill_MC,merged_backfill_MC.columns,len(merged_backfill_MC.columns),0)





df_model = pr.pd.DataFrame(index=merged_backfill_MC.index,data={
                        pr.PowerServiceMC:merged_backfill_MC[pr.PowerServiceMC]})

import fbprophet as ph
# Prophete Facebok
data_train = df_model.copy()
data_train.reset_index(level=0, inplace=True)
# # Prophet requires columns ds (Date) and y (value)
data_train = data_train.rename(columns={data_train.columns[0]: 'ds', data_train.columns[1]: 'y'})
data_train.drop([x for x in data_train.columns.tolist() if x not in ['ds','y']],axis=1,inplace=True)
# Make the prophet model and fit on the data
periode = 24*30
model = ph.Prophet(changepoint_prior_scale=0.05)
mask = (data_train['ds'] > '2017') & (data_train['ds'] < '2018-02')
model.fit(data_train.loc[mask])
# # Make a future dataframe for 1 month = 30 Day
future_consomation = model.make_future_dataframe(periods=periode, freq='H')
# Make predictions
forecast_consomation = model.predict(future_consomation)

# model.plot_components(forecast_consomation)
# pr.plt.show()

forecast = forecast_consomation['yhat'].iloc[-periode:]
observed = df_model[forecast_consomation.iloc[forecast.index[0],0].__str__():forecast_consomation.iloc[forecast.index[-1],0].__str__()][pr.PowerServiceMC]

#Plot observed vs predicted
prediction_to_plot = pd.DataFrame({'observed':observed.values, 'predicted': forecast.values}, index=observed.index)
pr.plot_data(prediction_to_plot,prediction_to_plot.columns,1)

feature_to_predict = pr.PowerServiceBC
data_model = pd.DataFrame(index=data_cleaned_hourly_BC.index)
data_model[feature_to_predict] = data_cleaned_hourly_BC[feature_to_predict].copy()
data_model['temperature'] = data_weather_darksky.loc[data_cleaned_hourly_BC.index]['apparentTemperature']
data_model['API_darksky_dewPoint'] = data_weather_darksky.loc[data_cleaned_hourly_BC.index]['dewPoint']
data_model['API_darksky_humidity'] = data_weather_darksky.loc[data_cleaned_hourly_BC.index]['humidity']
data_model['hours'] = data_model.index.hour

from sklearn.model_selection import train_test_split
def create_dataset(dataframe,feature_to_predict,portion=0.33):
    '''
    create the dataset for training the regression problem
    :param dataframe:
    :return:  X_train, X_test, y_train, y_test
    '''
    # scaler = MinMaxScaler()
    df_test = dataframe.copy()
    df_test.reset_index(inplace=True)
    df_test.drop([df_test.columns[0]], axis=1, inplace=True)
    # fix random seed for reproducibility
    pd.np.random.seed(7)
    X = df_test.drop([feature_to_predict], axis=1)
    y = df_test.drop([x for x in df_test.columns if x not in [feature_to_predict]], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=portion, random_state=42)
    # scaler = MinMaxScaler().fit(X_train)
    # X_train = scaler.transform(X_train)
    # X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

def test_all_classfiers(dataframe,feature_to_predict):
    X_train, X_test, y_train, y_test = create_dataset(dataframe,feature_to_predict)
    classifiers = [
        svm.SVR(),
        RandomForestRegressor(max_depth=2, random_state=0),
        linear_model.BayesianRidge(),
        linear_model.LassoLars(),
        linear_model.PassiveAggressiveRegressor(),
        linear_model.TheilSenRegressor()]
    df_result_metric = pd.DataFrame(index= [ item.__str__().split("(")[0] for item in classifiers],columns=['mse','rmse','r2','correlation'])
    for item in classifiers:
        clf = item
        clf.fit(pd.np.array(X_train), pd.np.array(y_train))
        pred = clf.predict(pd.np.array(X_test))
        predicted_df = pd.DataFrame({'observed':pd.np.array(y_test[feature_to_predict]), 'predicted': pred})
        #Metrics for regression
        mse = mean_squared_error(predicted_df.observed, predicted_df.predicted)
        rmse = sqrt(mean_squared_error(predicted_df.observed, predicted_df.predicted))
        r2 = r2_score(predicted_df.observed, predicted_df.predicted)
        correlation = pd.np.corrcoef(pd.np.array(y_test[feature_to_predict]),pred)
        # Store Metrics for regression
        df_result_metric.loc[item.__str__().split("(")[0]]['mse'] = round(mse,4)
        df_result_metric.loc[item.__str__().split("(")[0]]['rmse'] = round(rmse,4)
        df_result_metric.loc[item.__str__().split("(")[0]]['r2'] = round(r2,4)
        df_result_metric.loc[item.__str__().split("(")[0]]['correlation'] = round(correlation[0,1],4)
    return df_result_metric

result_metric = test_all_classfiers(data_model,feature_to_predict)


X_train, X_test, y_train, y_test = create_dataset(data_model,feature_to_predict)
model =  svm.SVR().fit(X_train,y_train)
from sklearn.externals import joblib

part_to_predict = data_model['2018-05'].copy()

predicted_df = apply_my_model_for_validation(model, part_to_predict ,feature_to_predict )

df_result_metric = pd.DataFrame(index=["SVR"],columns=['mse','rmse','r2','correlation'])
mse = mean_squared_error(predicted_df.observed, predicted_df.predicted)
rmse = sqrt(mean_squared_error(predicted_df.observed, predicted_df.predicted))
r2 = r2_score(predicted_df.observed, predicted_df.predicted)
correlation = pd.np.corrcoef(pd.np.array(predicted_df.observed),predicted_df.predicted)


# Store Metrics for regression
df_result_metric.loc["SVR"]['mse'] = mse
df_result_metric.loc["SVR"]['rmse'] = rmse
df_result_metric.loc["SVR"]['r2'] = r2
df_result_metric.loc["SVR"]['correlation'] = correlation[0,1]
pr.plot_data(predicted_df['2018-05'], predicted_df.columns, 1)
