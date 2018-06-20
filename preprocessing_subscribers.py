import pandas as pd
import models_used as mu
import preprocessing as pr
import preprocessing_weather as prw
from sklearn import svm
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

data_weather_darksky = prw.get_all_weather_darksky()
pr.check_timeindex(data_weather_darksky)

subs_BC = subs.drop_duplicates(['NB BC'], keep='last')
subs_MC = subs.drop_duplicates(['NB MC'], keep='last')
# pr.plot_data(subs_BC,subs_BC.columns,4,0)


data_site_hourly = pr.get_all_data_site()
# FOR BC
features_hourly_BC = ["Date", pr.PowerServiceBC]
data_prepared_hourly_BC = pr.prepare_date(data_site_hourly, features_hourly_BC)
data_cleaned_hourly_BC = pr.clean_data(data_prepared_hourly_BC, pr.PowerServiceBC,"zero")
pr.check_timeindex(data_cleaned_hourly_BC)
data_merged = data_cleaned_hourly_BC.join(subs['NB BC'], how='outer')
merged_backfill_BC = data_merged.fillna(method='ffill')
merged_backfill_BC.dropna(axis=0, how='any',inplace=True)
pr.check_timeindex(merged_backfill_BC)
#FOR MC
features_hourly_MC = ["Date", pr.PowerServiceMC]
data_prepared_hourly_MC = pr.prepare_date(data_site_hourly, features_hourly_MC)
data_cleaned_hourly_MC = pr.clean_data(data_prepared_hourly_MC, pr.PowerServiceMC,"zero")
pr.check_timeindex(data_cleaned_hourly_MC)
data_merged = data_cleaned_hourly_MC.join(subs['NB MC'], how='outer')
data_merged = data_cleaned_hourly_MC.join(subs['NB MC'], how='outer')
merged_backfill_MC = data_merged.fillna(method='ffill')
merged_backfill_MC.dropna(axis=0, how='any',inplace=True)
pr.check_timeindex(merged_backfill_MC)



# df_model_power = pr.pd.DataFrame(index=merged_backfill_BC.index,data={
#                         pr.PowerServiceBC:merged_backfill_BC[pr.PowerServiceBC]})
#
# df_model_nb = pr.pd.DataFrame(index=merged_backfill_BC.index,data={
#                         "NB BC":merged_backfill_MC['NB BC']})


#regression

feature_to_predict = pr.PowerServiceBC
data_model = pd.DataFrame(index=data_cleaned_hourly_BC.index)
data_model[feature_to_predict] = data_cleaned_hourly_BC[feature_to_predict].copy()
data_model['temperature'] = data_weather_darksky.loc[data_cleaned_hourly_BC.index]['apparentTemperature']
data_model['API_darksky_dewPoint'] = data_weather_darksky.loc[data_cleaned_hourly_BC.index]['dewPoint']
data_model['API_darksky_humidity'] = data_weather_darksky.loc[data_cleaned_hourly_BC.index]['humidity']
data_model['hours'] = data_model.index.hour

feature_to_predict = pr.PowerServiceMC
data_model = pd.DataFrame(index=data_cleaned_hourly_MC.index)
data_model[feature_to_predict] = data_cleaned_hourly_MC[feature_to_predict].copy()
data_model['temperature'] = data_weather_darksky.loc[data_cleaned_hourly_MC.index]['apparentTemperature']
data_model['API_darksky_dewPoint'] = data_weather_darksky.loc[data_cleaned_hourly_MC.index]['dewPoint']
data_model['API_darksky_humidity'] = data_weather_darksky.loc[data_cleaned_hourly_MC.index]['humidity']
data_model['hours'] = data_model.index.hour



mask_observed = (data_model.index > '2017') & (data_model.index < '2018-06-01 00:00:00')
data_model_observed = data_model.loc[mask_observed]
# result_metric = test_all_classfiers(data_model,feature_to_predict)
X_train, X_test, y_train, y_test = mu.create_dataset(data_model_observed,feature_to_predict)
model =  svm.SVR().fit(X_train,y_train)
mask_to_predict = (data_model.index > '2018-06-01 00:00:00')
part_to_predict = data_model.loc[mask_to_predict]
predicted_df_MC = mu.apply_my_model_for_validation(model, part_to_predict ,feature_to_predict )
pr.plot_data(predicted_df_MC, predicted_df_MC.columns,"")



PV_vs_BC_observed = pd.DataFrame(index=predicted_df_BC.index,
                                 data={
                        "PV":predicted_df.observed,
                        "BC":predicted_df_BC.observed
                                 }
                                 )

PV_vs_BC_predicted = pd.DataFrame(index=predicted_df_BC.index,
                                 data={
                        "PV":predicted_df.predicted,
                        "BC":predicted_df_BC.predicted
                                 }
                                 )


pr.plot_data(PV_vs_BC_observed, PV_vs_BC_observed.columns,"PV_vs_BC_observed")
pr.plot_data(PV_vs_BC_predicted, PV_vs_BC_predicted.columns,"PV_vs_BC_predicted")
