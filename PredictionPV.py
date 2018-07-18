import datetime
from math import sqrt

import pandas as pd
import pytz
from sklearn import svm
from sklearn.grid_search import GridSearchCV
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

def create_date_model_with_weather(feature_to_predict,data_cleaned_hourly,data_weather):
    puissance_thérorique= create_puissance_formule(data_weather)
    data_model = pd.DataFrame(index=data_cleaned_hourly.index)
    data_model[feature_to_predict] = data_cleaned_hourly[feature_to_predict].copy()
    data_model['temperature_cellule_théorique'] = (puissance_thérorique.loc[data_cleaned_hourly.index]['temperature_cellule']).copy()
    data_model['irradiance'] = data_cleaned_hourly[pr.Irradiation].copy()
    data_model['API_darksky_dewPoint'] = data_weather.loc[data_cleaned_hourly.index]['dewPoint']
    data_model['API_darksky_humidity'] = data_weather.loc[data_cleaned_hourly.index]['humidity']
    data_model['hours'] = data_model.index.hour
    return data_model

if __name__=="__main__":
    # Hourly Data
    import numpy as np
    feature_to_predict = pr.PowerPV
    data_site_hourly = pr.get_all_data_site()
    features_hourly = ["Date", feature_to_predict]
    data_prepared_hourly = pr.prepare_date(data_site_hourly, features_hourly)
    data_cleaned_hourly = pr.clean_data(data_prepared_hourly,feature_to_predict,"zero")
    data_cleaned_hourly['hour'] = data_cleaned_hourly.index.hour
    data_cleaned_hourly['day'] = data_cleaned_hourly.index.day
    data_cleaned_hourly['month'] = data_cleaned_hourly.index.month
    data_weather_darksky = prw.get_all_weather_darksky()
    #extract timeserie
    # timeserie = data_cleaned_hourly[feature_to_predict].copy()
    # pr.test_stationarity(timeserie['2018-02'])
    ##############



    # missing_darksky = data_weather_darksky.isnull().sum().append(pd.Series([len(data_weather_darksky)],index=['total_rows']))
    # missing_darksky[:-1] = missing_darksky[:-1].apply(lambda x: 100-(x * 100) /len(data_weather_darksky))
    # y=missing_darksky.values[:-1]
    # x=missing_darksky.index[:-1]
    # pr.plt.bar(x, y, color="brown")
    # pr.plt.xticks(rotation=90)
    # pr.plt.xlabel("attribut API")
    # pr.plt.ylabel("Pourcentage")
    # pr.plt.title("Presence of variable weather in darksky")
    # pr.plt.tight_layout()
    # pr.plt.show()
    #
    #
    #
    # data_weather_weatherbit = prw.get_all_weather_weatherbit()
    # missing_weatherbit = data_weather_weatherbit.isnull().sum().append(pd.Series([len(data_weather_weatherbit)],index=['total_rows']))
    # missing_weatherbit[:-1] = missing_weatherbit[:-1].apply(lambda x:100-(x * 100) /len(data_weather_weatherbit))
    # y=missing_weatherbit.values[:-1]
    # x=missing_weatherbit.index[:-1]
    # pr.plt.bar(x, y, color="green")
    # pr.plt.xticks(rotation=90)
    # pr.plt.xlabel("attribut API")
    # pr.plt.ylabel("Pourcentage")
    # pr.plt.title("Presence of variable weather in weatherbit")
    # pr.plt.tight_layout()
    # pr.plt.show()



    ########################################################################################################################
    #######################-------------------Create test dataset and train dataset---------------------####################
    ########################################################################################################################

    #Creation of the data to give to the model ( aggregation of different values )

# #############Temperature cellule###################
#     feature_to_predict ="temperature_cellule"
#     data_model_temp_cellule = pd.DataFrame(index=data_cleaned_hourly.index,data={"temperature_cellule":data_cleaned_hourly[pr.Temperature].copy()})
#     data_model_temp_cellule['API_darksky_dewPoint'] = data_weather_darksky.loc[data_model_temp_cellule.index]['dewPoint']
#     data_model_temp_cellule['API_darksky_humidity'] = data_weather_darksky.loc[data_model_temp_cellule.index]['humidity']
#     data_model_temp_cellule['API_darksky_temparture'] = data_weather_darksky.loc[data_model_temp_cellule.index]['apparentTemperature']
#     mask_observed = (data_model_temp_cellule.index > '2017') & (data_model_temp_cellule.index < '2018-06-01 00:00:00')
#     data_model_observed = data_model_temp_cellule.loc[mask_observed]
#     X_train, X_test, y_train, y_test = mu.create_dataset(data_model_observed, feature_to_predict)
#     model = svm.SVR(C=2, epsilon=0).fit(X_train, y_train)
#     mask_to_predict = (data_model_temp_cellule.index > '2018-06-01 00:00:00')
#     part_to_predict = data_model_temp_cellule.loc[mask_to_predict]
#     predicted_df_temp_cellule = mu.apply_my_model_for_validation(model, part_to_predict, feature_to_predict)
#     predicted_df_temp_cellule["temp_cellul_calcul"] = data_model_featured.loc[data_model_temp_cellule.index]['temperature_cellule_théorique']
#     mse = mean_squared_error(predicted_df_temp_cellule.observed, predicted_df_temp_cellule.predicted)
#     pr.plot_data(predicted_df_temp_cellule,predicted_df_temp_cellule.columns,"Prediction de temperature cellule")
#     print("le score est de " + str(mse))
#############Temperature cellule###################

    # Label encoding for Summary
    from sklearn import preprocessing

    le = preprocessing.LabelEncoder()
    le.fit(data_weather_darksky['summary'])
    list(le.classes_).__len__()
    data_weather_darksky['summary'] = le.transform(data_weather_darksky['summary'])

    feature_to_predict = pr.PowerPV
    from time import time
    t1 = time()
    data_model = pd.DataFrame(index=data_cleaned_hourly.index)
    data_model[feature_to_predict] = data_cleaned_hourly[feature_to_predict].copy()
    # data_model["temp_cellule"] = data_cleaned_hourly[pr.Temperature].copy()

    # data_model['API_darksky_dewPoint'] = data_weather_darksky.loc[data_cleaned_hourly.index]['dewPoint']
    data_model['API_darksky_humidity'] = data_weather_darksky.loc[data_cleaned_hourly.index]['humidity']
    data_model['API_darksky_apparentTemperature'] = data_weather_darksky.loc[data_cleaned_hourly.index]['apparentTemperature']
    # data_model['API_darksky_summary'] = data_weather_darksky.loc[data_cleaned_hourly.index]['summary']
    data_model['hours'] = data_model.index.hour

    # res = mu.test_all_classfiers(data_model,feature_to_predict)
##########Time calculation for feature engineering and scaling###############
    # data_model_featured = data_model[:int(len(data_model)/2)].copy()
    # X_train, X_test, y_train, y_test = mu.create_dataset(data_model_featured,feature_to_predict,portion=0)
    # model = svm.SVR(C=2, epsilon=0).fit(X_train,y_train)
    # print("half::"+str(time() - t1))
    #
    # data_model_featured = data_model.copy()
    # X_train, X_test, y_train, y_test = mu.create_dataset(data_model_featured, feature_to_predict, portion=0)
    # model = svm.SVR(C=2, epsilon=0).fit(X_train, y_train)
    # print("full::"+str(time() - t1))

    data_model_featured = data_model.copy()
    mask_observed = (data_model_featured.index > '2018') & (data_model_featured.index < '2018-06-01 00:00:00')
    data_model_observed = data_model_featured.loc[mask_observed]
    X_train, X_test, y_train, y_test = mu.create_dataset(data_model_observed, feature_to_predict, portion=0.33)


    feature_to_predict = pr.PowerPV

    parameters = {'kernel': ['linear', 'rbf', 'poly', 'sigmoid'], 'C': [0.001, 0.1, 1, 10],
                  'epsilon': [0.1, 0.2, 0.3]}

    a = [parameters['kernel'], parameters['C'], parameters['epsilon']]


    def boxed_product(somelists):
        from itertools import product
        return [list(combo) for combo in product(*somelists)]

    combi = boxed_product(a)

    df_result_metric_forward = pd.DataFrame(index=[str(i) for i in combi],
                                            columns=['mse', 'rmse', 'r2', 'correlation'])

    df_result_metric_forward.sort_index(inplace=True)

    t1 = time()
    for i in range(len(combi)):
        print("Combi number "+str(i)+"for the combinaison "+str(combi[i]))
        mask_observed = (data_model_featured.index > '2017') & (data_model_featured.index < '2018-06-01 00:00:00')
        data_model_observed = data_model_featured.loc[mask_observed]
        X_train, X_test, y_train, y_test = mu.create_dataset(data_model_observed['2017'], feature_to_predict, portion=0.1)
        model = svm.SVR(kernel=combi[i][0],C=combi[i][1], epsilon=combi[i][2]).fit(X_train, y_train)
        mask_to_predict = (data_model_featured.index > '2018-06-01 00:00:00')
        part_to_predict = data_model_featured.loc[mask_to_predict]
        predicted_df = mu.apply_my_model_for_validation(model, part_to_predict, feature_to_predict)
        mse = mean_squared_error(predicted_df.observed, predicted_df.predicted)
        r2 = r2_score(predicted_df.observed, predicted_df.predicted)
        correlation = pd.np.corrcoef(pd.np.array(predicted_df.observed), predicted_df.predicted)
        rmse = sqrt(mean_squared_error(predicted_df.observed, predicted_df.predicted))
        df_result_metric_forward.loc[str(combi[i])]['mse'] = mse
        df_result_metric_forward.loc[str(combi[i])]['rmse'] = rmse
        df_result_metric_forward.loc[str(combi[i])]['r2'] = r2
        df_result_metric_forward.loc[str(combi[i])]['correlation'] = correlation[0, 1]
    print("end after "+ str(t1-time()))
    df_result_metric_forward_float = df_result_metric_forward.astype(float)
    writer = pd.ExcelWriter('outputForGridSearch.xlsx')
    df_result_metric_forward_float.to_excel(writer,'Sheet1')
    writer.save()
    # np.std(predicted_df.observed)
    # pr.plot_data(predicted_df_PV,predicted_df_PV.columns,"Prediction de PV")

    # testt = predicted_df.observed.copy()
    # testt = testt[testt]
    # predicted_df.observed.hist()
    # pr.plt.show()
#
# data_aggregated = data_cleaned_hourly['2017-02'].copy()
#
# times = pd.DatetimeIndex(data_aggregated.index)
# grouped = data_aggregated.groupby([times.month,times.day]).sum()
# a = grouped.index
# range = pd.date_range(data_aggregated.index[0], data_aggregated.index[-1], freq='D')
# data_aggregated_2 = data_cleaned_hourly['2017-02'].copy()
# data_aggregated_2 = data_aggregated_2.groupby([data_aggregated_2.index.dt.floor('H')]).agg(['sum'])
# #for columns from MultiIndex
# data_aggregated_2.columns = data_aggregated_2.columns.map('_'.join)




########################################################################################################################
#######################-------------------#Test my model on new predictions#---------------------#######################
########################################################################################################################

# #Test my model on new predictions
# import preprocessing as pr
# from sklearn.externals import joblib
# import pandas as pd
# import models_used as mu
# model_loaded =joblib.load('models/model_svr_2018-04-05.pkl')
# predict_darsky_data = pr.pd.read_csv("Darksky_forecast/data_2018-06-19 12_23_32.csv")
# predict_darsky_data.drop_duplicates(['time.1'], keep='first', inplace=True)
# predict_darsky_data.set_index(pd.to_datetime(predict_darsky_data["time.1"], unit='s', utc=True), inplace=True)
# predict_darsky_data.index = predict_darsky_data.index.tz_convert('Indian/Antananarivo').tz_localize(None)
# predict_darsky_df = predict_darsky_data.drop(['time.1'], axis=1)
# def predict_for_48h(model,dataframe_prediction_weather):
#     predict_darsky_df = dataframe_prediction_weather.copy()
#     puissance_thérorique= create_puissance_formule(predict_darsky_df)
#     data_model_prediction = pd.DataFrame(index=predict_darsky_df.index)
#     data_model_prediction['temperature_cellule_théorique'] = (puissance_thérorique.loc[predict_darsky_df.index]['temperature_cellule']).copy()
#     data_model_prediction['API_darksky_dewPoint'] = predict_darsky_df.loc[predict_darsky_df.index]['dewPoint']
#     data_model_prediction['API_darksky_humidity'] = predict_darsky_df.loc[predict_darsky_df.index]['humidity']
#     data_model_prediction['hours'] = data_model_prediction.index.hour
#     predicted_df = model.predict(data_model_prediction)
#     return predicted_df
#
# predicted_df = predict_for_48h(model_loaded,predict_darsky_data)
# pr.plt.plot(predicted_df);pr.plt.show()
# prediction_to_plot = pd.DataFrame(index=predict_darsky_data.index,
#                                   data={
#                                       'observed' : data_cleaned_hourly[feature_to_predict].loc[predict_darsky_data.index],
#                                       'predicted': predicted_df}
#                                   )
# pr.plot_data(prediction_to_plot,prediction_to_plot.columns)
# madatz = pytz.timezone('Indian/Antananarivo')  ## Set your timezone
# madatz_now = datetime.datetime.now(madatz).date().__str__()
# prediction_to_plot.to_csv("data_predicted_with_model/predictionfor"+madatz_now+".csv")
# prediction_to_plot=pd.read_csv(r"C:\Users\G603344\PycharmProjects\POCv1.0\data_predicted_with_model\predictionfor2018-05-31.csv")
# prediction_to_plot.set_index(prediction_to_plot["time.1"], inplace=True)
# prediction_to_plot.index = prediction_to_plot.index.tz_convert('Indian/Antananarivo').tz_localize(None)




