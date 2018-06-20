import datetime
from math import sqrt

import pandas as pd
import pytz
from sklearn import svm
from sklearn.metrics import mean_squared_error, r2_score

import models_used as mu
import preprocessing as pr
import preprocessing_weather as prw

pd.options.mode.chained_assignment = None  # to avoid the false postive warning of SettingWithCopyWarning:



def create_puissance_formule(df):
    '''

    :param df: dataframe with weather data
    :return: calculated dataframe with irradiance and temperature cellule using the formula ...
    '''
    import math
    import preprocessing_weather as prw
    alpha = 0 #elev_angle
    azimut = 0
    tilt = 20
    orientation = 0
    nb_panels = 34
    puissance_noct = 220
    temperature_noct = 45
    loss_by_degree = 0.402
    puissance_jonathan  = pd.DataFrame(index=df.index)
    puissance_jonathan["Irradiance_module"] = pd.Series(puissance_jonathan.index).apply(lambda x : max(1000*((math.cos(prw.get_sun_alt(x))*(math.sin(math.radians(tilt)))*math.cos(orientation - prw.get_sun_az(x)))+(math.sin(prw.get_sun_alt(x)))*(math.cos(math.radians(tilt)))),0)).values
    puissance_jonathan["Irradiance_module_34"] = puissance_jonathan["Irradiance_module"] * nb_panels
    puissance_jonathan["PV_irradiance"] =(puissance_jonathan["Irradiance_module"]/800) * puissance_noct
    puissance_jonathan["temperature_cellule"] = df["apparentTemperature"] + (puissance_jonathan["Irradiance_module"]/800)*(temperature_noct - 20)
    puissance_jonathan["Puissance lié au coef temp"] = puissance_jonathan["PV_irradiance"] * (loss_by_degree/100) * puissance_jonathan["temperature_cellule"] - temperature_noct
    puissance_jonathan["Puissance total théorique total"] = (nb_panels * (puissance_jonathan["PV_irradiance"] - puissance_jonathan["Puissance lié au coef temp"]))/1000
    puissance_jonathan["Puissance total théorique par panneau"] =  (puissance_jonathan["PV_irradiance"] - puissance_jonathan["Puissance lié au coef temp"])
    return puissance_jonathan


# #Check validation between forecast data and observed data for weather Darksky_forecast
def validate_presente_forecast_darksky(data_weather_darksky):
    import glob
    path = r'Darksky_forecast/'  # use your path
    allFiles = glob.glob(path + "/*.csv")
    frame = pr.pd.DataFrame()
    list_ = []
    for file_ in allFiles:
        df = pd.read_csv(file_, sep=",", index_col=0, header=0)
        list_.append(df)
    data_forecast_darksky = pd.concat(list_)
    data_forecast_darksky.set_index(pd.to_datetime(data_forecast_darksky["time.1"], unit='s',utc=True), inplace=True)
    data_forecast_darksky.index = data_forecast_darksky.index.tz_convert('Indian/Antananarivo').tz_localize(None)
    data_forecast_darksky.__len__()
    data_forecast_darksky.drop_duplicates(['time.1'], keep='last', inplace=True)
    data_forecast_darksky.__len__()
    mu.check_timeindex(data_forecast_darksky)
    data_merged = pd.merge(data_forecast_darksky, data_weather_darksky, left_index=True, right_index=True)
    diff_df = data_merged[['apparentTemperature_x','apparentTemperature_y']].copy()
    rmse = sqrt(mean_squared_error(diff_df.apparentTemperature_x[:100],diff_df.apparentTemperature_y[:100]))
    print(str(rmse)+' pas pour '+str(len(diff_df)))


# Check validation between forecast data and observed data for weather weatherbit_forecast
def validate_presente_forecast_weatherbit(data_weather_weatherbit):
    import glob
    path = r'weatherbit_forecast/'  # use your path
    allFiles = glob.glob(path + "/*.csv")
    frame = pr.pd.DataFrame()
    list_ = []
    for file_ in allFiles:
        df = pd.read_csv(file_, sep=",", index_col=0, header=0)
        list_.append(df)
    data_forecast_weatherbit = pd.concat(list_)
    data_forecast_weatherbit.set_index(pd.to_datetime(data_forecast_weatherbit["ts"], unit='s',utc=True), inplace=True)
    data_forecast_weatherbit.index = data_forecast_weatherbit.index.tz_convert('Indian/Antananarivo').tz_localize(None)
    data_forecast_weatherbit.__len__()
    data_forecast_weatherbit.drop_duplicates(['ts'], keep='last', inplace=True)
    data_forecast_weatherbit.__len__()
    mu.check_timeindex(data_forecast_weatherbit)
    data_merged = pd.merge(data_forecast_weatherbit, data_weather_weatherbit, left_index=True, right_index=True)
    diff_df = data_merged[['app_temp','temp_x']].copy()
    rmse = sqrt(mean_squared_error(diff_df.app_temp[:100],diff_df.temp_x[:100]))
    print(str(rmse)+' degrée pour '+str(len(diff_df)))


# Hourly Data
feature_to_predict = pr.PowerPV
data_site_hourly = pr.get_all_data_site()
features_hourly = ["Date", feature_to_predict, pr.Temperature, pr.Irradiation]
data_prepared_hourly = pr.prepare_date(data_site_hourly, features_hourly)
data_cleaned_hourly = pr.clean_data(data_prepared_hourly,feature_to_predict,"mean")
data_cleaned_hourly['hour'] = data_cleaned_hourly.index.hour
data_cleaned_hourly['day'] = data_cleaned_hourly.index.day
data_cleaned_hourly['month'] = data_cleaned_hourly.index.month

#extract timeserie
# timeserie = data_cleaned_hourly[feature_to_predict].copy()
# pr.test_stationarity(timeserie['2018-02'])
##############


data_weather_darksky = prw.get_all_weather_darksky()
pr.check_timeindex(data_weather_darksky)

# missing_darksky = data_weather_darksky.isnull().sum().append(pd.Series([len(data_weather_darksky)],index=['total_rows']))
# missing_darksky[:-1] = missing_darksky[:-1].apply(lambda x: x * 100 /len(data_weather_darksky))
# y=missing_darksky.values[:-1]
# x=missing_darksky.index[:-1]
# pr.plt.bar(x, y, color="green")
# pr.plt.xticks(rotation=90)
# pr.plt.title("Presence of variable weather in darksky")
# pr.plt.tight_layout()
# pr.plt.show()
#
# data_weather_darksky['visibility'][data_weather_darksky['visibility'].isnull()] = data_weather_darksky['visibility'].mean()
#
# missing_weatherbit = data_weather_weatherbit['2018-04-04':].isnull().sum().append(pd.Series([len(data_weather_weatherbit['2018-04-04':])],index=['total_rows']))
# missing_weatherbit[:-1] = missing_weatherbit[:-1].apply(lambda x: x * 100 /len(data_weather_weatherbit['2018']))


########################################################################################################################
#######################-------------------Create test dataset and train dataset---------------------####################
########################################################################################################################

data_cleaned_hourly_on_hours = data_cleaned_hourly[data_cleaned_hourly[feature_to_predict] != 0]
data_cleaned_hourly = data_cleaned_hourly_on_hours
data_weather_darksky_on_hours = data_weather_darksky.loc[data_cleaned_hourly_on_hours.index]
data_weather_darksky =  data_weather_darksky_on_hours
#Creation of the data to give to the model ( aggregation of different values )
def create_date_model_with_weather(feature_to_predict):
    puissance_thérorique= create_puissance_formule(data_weather_darksky)
    data_model = pd.DataFrame(index=data_cleaned_hourly.index)
    data_model[feature_to_predict] = data_cleaned_hourly[feature_to_predict].copy()
    data_model['temperature_cellule_théorique'] = (puissance_thérorique.loc[data_cleaned_hourly.index]['temperature_cellule']).copy()
    data_model['API_darksky_dewPoint'] = data_weather_darksky.loc[data_cleaned_hourly.index]['dewPoint']
    data_model['API_darksky_humidity'] = data_weather_darksky.loc[data_cleaned_hourly.index]['humidity']
    data_model['hours'] = data_model.index.hour
    return data_model


data_model_featured = create_date_model_with_weather(feature_to_predict).copy()

mask_observed = (data_model_featured.index > '2017') & (data_model_featured.index < '2018-06-01 00:00:00')
data_model_observed = data_model_featured.loc[mask_observed]
X_train, X_test, y_train, y_test = mu.create_dataset(data_model_observed,feature_to_predict)
model = svm.SVR(C=2, epsilon=0).fit(X_train,y_train)
mask_to_predict = (data_model_featured.index > '2018-06-01 00:00:00')
part_to_predict = data_model_featured.loc[mask_to_predict]
predicted_df = mu.apply_my_model_for_validation(model, part_to_predict,feature_to_predict)

pr.plot_data(predicted_df,predicted_df.columns,"Prediction de PV")

predicted_df.predicted.mean()/ predicted_df.observed.mean()



data_aggregated = data_cleaned_hourly['2017-02'].copy()

times = pd.DatetimeIndex(data_aggregated.index)
grouped = data_aggregated.groupby([times.month,times.day]).sum()
a = grouped.index
range = pd.date_range(data_aggregated.index[0], data_aggregated.index[-1], freq='D')
data_aggregated_2 = data_cleaned_hourly['2017-02'].copy()
data_aggregated_2 = data_aggregated_2.groupby([data_aggregated_2.index.dt.floor('H')]).agg(['sum'])
#for columns from MultiIndex
data_aggregated_2.columns = data_aggregated_2.columns.map('_'.join)




########################################################################################################################
#######################-------------------#Test my model on new predictions#---------------------#######################
########################################################################################################################

#Test my model on new predictions
import preprocessing as pr
from sklearn.externals import joblib
import pandas as pd
import models_used as mu
model_loaded =joblib.load('models/model_svr_2018-04-05.pkl')
predict_darsky_data = pr.pd.read_csv("Darksky_forecast/data_2018-06-19 12_23_32.csv")
predict_darsky_data.drop_duplicates(['time.1'], keep='first', inplace=True)
predict_darsky_data.set_index(pd.to_datetime(predict_darsky_data["time.1"], unit='s', utc=True), inplace=True)
predict_darsky_data.index = predict_darsky_data.index.tz_convert('Indian/Antananarivo').tz_localize(None)
predict_darsky_df = predict_darsky_data.drop(['time.1'], axis=1)

def predict_for_48h(model,dataframe_prediction_weather):
    predict_darsky_df = dataframe_prediction_weather.copy()
    puissance_thérorique= create_puissance_formule(predict_darsky_df)
    data_model_prediction = pd.DataFrame(index=predict_darsky_df.index)
    data_model_prediction['temperature_cellule_théorique'] = (puissance_thérorique.loc[predict_darsky_df.index]['temperature_cellule']).copy()
    data_model_prediction['API_darksky_dewPoint'] = predict_darsky_df.loc[predict_darsky_df.index]['dewPoint']
    data_model_prediction['API_darksky_humidity'] = predict_darsky_df.loc[predict_darsky_df.index]['humidity']
    data_model_prediction['hours'] = data_model_prediction.index.hour
    predicted_df = model.predict(data_model_prediction)
    return predicted_df

predicted_df = predict_for_48h(model_loaded,predict_darsky_data)

pr.plt.plot(predicted_df);pr.plt.show()

prediction_to_plot = pd.DataFrame(index=predict_darsky_data.index,
                                  data={
                                      'observed' : data_cleaned_hourly[feature_to_predict].loc[predict_darsky_data.index],
                                      'predicted': predicted_df}
                                  )
pr.plot_data(prediction_to_plot,prediction_to_plot.columns)
madatz = pytz.timezone('Indian/Antananarivo')  ## Set your timezone
madatz_now = datetime.datetime.now(madatz).date().__str__()
prediction_to_plot.to_csv("data_predicted_with_model/predictionfor"+madatz_now+".csv")

prediction_to_plot=pd.read_csv(r"C:\Users\G603344\PycharmProjects\POCv1.0\data_predicted_with_model\predictionfor2018-05-31.csv")
prediction_to_plot.set_index(prediction_to_plot["time.1"], inplace=True)
prediction_to_plot.index = prediction_to_plot.index.tz_convert('Indian/Antananarivo').tz_localize(None)




