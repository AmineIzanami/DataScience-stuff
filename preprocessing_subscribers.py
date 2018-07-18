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
data_cleaned_hourly_BC = pr.clean_data(data_prepared_hourly_BC, pr.PowerServiceBC,"mean")
pr.check_timeindex(data_cleaned_hourly_BC)

data_merged = data_cleaned_hourly_BC.join(subs['NB BC'], how='outer')
merged_backfill_BC = data_merged.fillna(method='ffill')
merged_backfill_BC.dropna(axis=0, how='any',inplace=True)
pr.plot_data(merged_backfill_BC,merged_backfill_BC.columns,"Consomaton vs consumers",mode=0)
pr.check_timeindex(merged_backfill_BC)
#FOR MC
features_hourly_MC = ["Date", pr.PowerServiceMC]
data_prepared_hourly_MC = pr.prepare_date(data_site_hourly, features_hourly_MC)
data_cleaned_hourly_MC = pr.clean_data(data_prepared_hourly_MC, pr.PowerServiceMC,"mean")
pr.check_timeindex(data_cleaned_hourly_MC)
feature_to_predict = pr.PowerServiceBC
# mu.fbprophet_Daily(data_cleaned_hourly_BC,feature_to_predict)


data_merged = data_cleaned_hourly_MC.join(subs['NB MC'], how='outer')
data_merged = data_cleaned_hourly_MC.join(subs['NB MC'], how='outer')
merged_backfill_MC = data_merged.fillna(method='ffill')
merged_backfill_MC.dropna(axis=0, how='any',inplace=True)
pr.plot_data(merged_backfill_MC,merged_backfill_MC.columns,"Consomaton vs consumers",mode=0)
pr.check_timeindex(merged_backfill_MC)



# df_model_power = pr.pd.DataFrame(index=merged_backfill_BC.index,data={
#                         pr.PowerServiceBC:merged_backfill_BC[pr.PowerServiceBC]})
#
# df_model_nb = pr.pd.DataFrame(index=merged_backfill_BC.index,data={
#                         "NB BC":merged_backfill_MC['NB BC']})


#regression
# FOR BC
feature_to_predict = pr.PowerServiceBC
data_model_BC = pd.DataFrame(index=data_cleaned_hourly_BC.index)
data_model_BC[feature_to_predict] = data_cleaned_hourly_BC[feature_to_predict].copy()
data_model_BC['temperature'] = data_weather_darksky.loc[data_cleaned_hourly_BC.index]['apparentTemperature']
data_model_BC['API_darksky_dewPoint'] = data_weather_darksky.loc[data_cleaned_hourly_BC.index]['dewPoint']
data_model_BC['API_darksky_humidity'] = data_weather_darksky.loc[data_cleaned_hourly_BC.index]['humidity']
data_model_BC['hours'] = data_model_BC.index.hour
mask_observed = (data_model_BC.index > '2017') & (data_model_BC.index < '2018-06-01 00:00:00')
data_model_observed = data_model_BC.loc[mask_observed]

result_test_BC = mu.test_all_classfiers(data_model_observed,feature_to_predict)

# result_metric = test_all_classfiers(data_model,feature_to_predict)
X_train, X_test, y_train, y_test = mu.create_dataset(data_model_observed,feature_to_predict)
model =  svm.SVR().fit(X_train,y_train)
mask_to_predict = (data_model_BC.index > '2018-06-01 00:00:00')
part_to_predict = data_model_BC.loc[mask_to_predict]
predicted_df_BC = mu.apply_my_model_for_validation(model, part_to_predict ,feature_to_predict )
# pr.plot_data(predicted_df_BC, predicted_df_BC.columns,"")



#FOR MC
feature_to_predict = pr.PowerServiceMC
data_model_MC = pd.DataFrame(index=data_cleaned_hourly_MC.index)
data_model_MC[feature_to_predict] = data_cleaned_hourly_MC[feature_to_predict].copy()
data_model_MC['temperature'] = data_weather_darksky.loc[data_cleaned_hourly_MC.index]['apparentTemperature']
data_model_MC['API_darksky_dewPoint'] = data_weather_darksky.loc[data_cleaned_hourly_MC.index]['dewPoint']
data_model_MC['API_darksky_humidity'] = data_weather_darksky.loc[data_cleaned_hourly_MC.index]['humidity']
data_model_MC['hours'] = data_model_MC.index.hour
mask_observed = (data_model_MC.index > '2017') & (data_model_MC.index < '2018-06-01 00:00:00')
data_model_observed = data_model_MC.loc[mask_observed]
# result_metric = test_all_classfiers(data_model,feature_to_predict)
X_train, X_test, y_train, y_test = mu.create_dataset(data_model_observed,feature_to_predict)
model =  svm.SVR().fit(X_train,y_train)
mask_to_predict = (data_model_MC.index > '2018-06-01 00:00:00')
part_to_predict = data_model_MC.loc[mask_to_predict]
predicted_df_MC = mu.apply_my_model_for_validation(model, part_to_predict ,feature_to_predict )
pr.plot_data(predicted_df_MC, predicted_df_MC.columns,"")








PV_vs_BCMC_observed = pd.DataFrame(index=predicted_df_PV.index,
                                 data={
                        "PV":predicted_df_PV.observed,
                        "BCMC":predicted_df_MC.observed  + predicted_df_BC.observed
                                 }
                                 )

PV_vs_BCMC_predicted = pd.DataFrame(index=predicted_df_PV.index,
                                 data={
                        "PV":predicted_df_PV.predicted,
                        "BCMC":predicted_df_MC.predicted + predicted_df_BC.predicted
                                 }
                                 )


pr.plot_data(PV_vs_BCMC_observed, PV_vs_BCMC_observed.columns,"PV_vs_BCMC_observed")
pr.plot_data(PV_vs_BCMC_predicted, PV_vs_BCMC_predicted.columns,"PV_vs_BCMC_predicted")


ax = PV_vs_BCMC_observed[['PV']].plot(kind="bar",color="brown",label=["PV"]);
ax2 = ax.twinx()
ax2.plot(ax.get_xticks(), PV_vs_BCMC_observed[['BCMC']], marker='o',label=["BC+MC"])
pr.plt.tight_layout();
pr.plt.legend(loc='best')
pr.plt.show()



data_site_hourly_plot = pr.get_all_data_site()
import matplotlib.pyplot as plt
# FOR BC
features_hourly_BC = ["Date", pr.PowerPV,pr.PowerServiceBC,pr.PowerServiceMC]
data_prepared_hourly_BC_plot = pr.prepare_date(data_site_hourly_plot, features_hourly_BC)
data_cleaned_hourly_BC_plot = pr.clean_data(data_prepared_hourly_BC_plot, pr.PowerServiceBC,"zero")
mask_to_predict_plot = (data_cleaned_hourly_BC_plot.index > '2018-06-01 00:00:00')
data_to_plot = data_cleaned_hourly_BC_plot.loc[mask_to_predict_plot]
data_to_plot['BCMC'] = data_to_plot['Load_1.BC - P']+data_to_plot['Load_1.MC - P']

# data_to_plot.drop(['PV_1 - P','Load_1.BC - P','Load_1.MC - P', 'DC Sys.Batt - SoC',],axis=1,inplace=True)

data_cleaned_hourly_BC_plot['hours'] = data_cleaned_hourly_BC_plot.index.hour
# data_cleaned_hourly_BC_plot[['hours']].hist();plt.show()

grouped_hours = data_cleaned_hourly_BC_plot.groupby(['hours']).sum()
pr.plot_data(grouped_hours,grouped_hours.columns,'attributes',0,len(grouped_hours.columns))

ax = data_to_plot[['PV_1 - P']].plot(kind="bar",color="brown",label=["PV"]);
ax2 = ax.twinx()
ax2.plot(ax.get_xticks(), data_to_plot[['BCMC']], marker='o',label=["BC+MC"])
plt.tight_layout();
plt.legend(loc='best')
plt.show()

ax = data_cleaned_hourly_BC_plot[['DC Sys.Batt - SoC']].plot()
ax.axhline(y=40, xmin=0, xmax=1, color='r', linestyle='--', lw=2);pr.plt.show()