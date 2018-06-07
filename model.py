
import itertools
from math import sqrt

import pandas as pd
import pytz
import datetime
from sklearn import linear_model, svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import preprocessing as pr
import preprocessing_weather as prw

pd.options.mode.chained_assignment = None  # to avoid the false postive warning of SettingWithCopyWarning:

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

def apply_my_model_for_validation(model, dataframe,feature_to_predict):
    '''
    apply model on the dataframe and return a dataframe with two columns the observed and the predicted
    :param model,dataframe:
    :param hours_to_forecast:
    :return: prediction_to_plot
    '''
    df_test = dataframe.copy()
    index_predict = df_test.index
    df_test.reset_index(inplace=True)
    df_test.drop([df_test.columns[0]], axis=1, inplace=True)
    # fix random seed for reproducibility
    pd.np.random.seed(7)
    X = df_test.drop([feature_to_predict], axis=1)
    # X = scaler.transform(X)
    y = df_test.drop([x for x in df_test.columns if x not in [feature_to_predict]], axis=1)
    pred = model.predict(X)
    prediction_to_plot = pd.DataFrame(index=index_predict,
                                      data={
                                          'observed': pd.np.array(y[feature_to_predict]),
                                          'predicted': pred}
                                      )
    return prediction_to_plot

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
        predicted_df = pd.DataFrame({'observed':pd.np.array(y_test[pr.PowerPV]), 'predicted': pred})
        #Metrics for regression
        mse = mean_squared_error(predicted_df.observed, predicted_df.predicted)
        rmse = sqrt(mean_squared_error(predicted_df.observed, predicted_df.predicted))
        r2 = r2_score(predicted_df.observed, predicted_df.predicted)
        correlation = pd.np.corrcoef(pd.np.array(y_test[feature_to_predict]),pred)
        # Store Metrics for regression
        df_result_metric.loc[item.__str__().split("(")[0]]['mse'] = mse
        df_result_metric.loc[item.__str__().split("(")[0]]['rmse'] = rmse
        df_result_metric.loc[item.__str__().split("(")[0]]['r2'] = r2
        df_result_metric.loc[item.__str__().split("(")[0]]['correlation'] = correlation[0,1]
    return df_result_metric

def check_timeindex(df):
    diff = (df.index[-1] - df.index[0])
    days, seconds = diff.days, diff.seconds
    hours = (days * 24 + seconds // 3600)+1
    if(hours == len(df)):

        return "All date are ok"
    else:
        range = pd.date_range(df.index[0], df.index[-1], freq='H')
        diff_range = [x for x in range if x not in df.index]
        return pr.pd.Series(diff_range)

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
def validate_presente_forecast_darksky():
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
    check_timeindex(data_forecast_darksky)
    data_merged = pd.merge(data_forecast_darksky, data_weather_darksky, left_index=True, right_index=True)
    diff_df = data_merged[['apparentTemperature_x','apparentTemperature_y']].copy()
    rmse = sqrt(mean_squared_error(diff_df.apparentTemperature_x[:100],diff_df.apparentTemperature_y[:100]))
    print(str(rmse)+' pas pour '+str(len(diff_df)))


#Check validation between forecast data and observed data for weather weatherbit_forecast
def validate_presente_forecast_weatherbit():
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
    check_timeindex(data_forecast_weatherbit)
    data_merged = pd.merge(data_forecast_weatherbit, data_weather_weatherbit, left_index=True, right_index=True)
    diff_df = data_merged[['app_temp','temp_x']].copy()
    rmse = sqrt(mean_squared_error(diff_df.app_temp[:100],diff_df.temp_x[:100]))
    print(str(rmse)+' degrée pour '+str(len(diff_df)))

# benchmarking of the temperature between different API weather
def bench_temp_apis():
    slice_bit = data_weather_weatherbit['2018-03-17 01:00:00':'2018-04-05 07:00:00'].copy()
    slice_darksky = data_weather_darksky['2018-03-17 01:00:00':'2018-04-05 07:00:00'].copy()
    slice_panel = data_cleaned_hourly['2018-03-17 01:00:00':'2018-04-05 07:00:00'].copy()
    df_bench = pd.DataFrame(index=slice_bit.index)
    df_bench["Temperature_panel"] = slice_panel[pr.Temperature]
    df_bench["API bit"] = slice_bit['temp']
    df_bench["API darksky temperature"] = (slice_darksky['temperature'])
    df_bench["API darksky apparentTemperature"] = (slice_darksky['apparentTemperature'])
    a = df_bench.corr()
    pr.plot_data(df_bench,df_bench.columns,1)

def forward_selection_features(data_model):
    combi = []
    features_to_use = [x for x in data_model.columns if x not in [pr.PowerPV]]
    for i in range(1,len(features_to_use)):
        combi += itertools.combinations(features_to_use,i)
    all_combi = [list(t) for t in combi]
    df_result_metric_mae =[]
    df_result_metric_mse =[]
    df_result_metric_r2 =[]
    df_result_metric_correlation =[]
    df_result_metric_rmse =[]
    for i in range(len(all_combi)):
        features_validation = data_model[[pr.PowerPV]+all_combi[i]].copy()
        print("Validation for :" + all_combi[i].__str__())
        #test the best model SVR with default hyperparameters
        X_train, X_test, y_train, y_test = create_dataset(features_validation['2017'])
        model = svm.SVR(C=2, epsilon=0).fit(X_train, y_train)
        part_to_predict = features_validation['2018'].copy()
        predicted_df = apply_my_model_for_validation(model, part_to_predict)
        mse = mean_squared_error(predicted_df.observed, predicted_df.predicted)
        mae = mean_absolute_error(predicted_df.observed, predicted_df.predicted)
        r2 = r2_score(predicted_df.observed, predicted_df.predicted)
        correlation = pd.np.corrcoef(pd.np.array(predicted_df.observed), predicted_df.predicted)
        rmse = sqrt(mean_squared_error(predicted_df.observed, predicted_df.predicted))
        df_result_metric_mse += [mse]
        df_result_metric_mae += [mae]
        df_result_metric_r2 += [r2]
        df_result_metric_correlation += [correlation]
        df_result_metric_rmse += [rmse]
    for i in range(len(df_result_metric_mae)):
        if(df_result_metric_mse[i] == min(df_result_metric_mse)):
            print("min mse : "+str(all_combi[i]))
        if (df_result_metric_mae[i] == min(df_result_metric_mae)):
            print("min ma : " + str(all_combi[i]))
        if (df_result_metric_rmse[i] == min(df_result_metric_rmse)):
            print("min rmse : " + str(all_combi[i]))
        if (df_result_metric_r2[i] == max(df_result_metric_r2)):
            print("max r2 : " + str(all_combi[i]))
        # if (df_result_metric_correlation[i] == max(df_result_metric_correlation)):
        #     print("max correlation : " + str(all_combi[i]))

def stats_models_summary():
    import statsmodels.api as sm
    model = sm.OLS(y_train, X_train).fit()
    predictions = model.predict(X_test)
    model.summary()

def model_to_pmml(model,X_train,y_train):
    from sklearn2pmml.pipeline import PMMLPipeline
    power_pipeline = PMMLPipeline([
        ("classifier",model)
    ])

    power_pipeline.fit(X_train, y_train)
    from sklearn2pmml import sklearn2pmml
    sklearn2pmml(power_pipeline, "LogisticRegressionPowerPV.pmml", with_repr = True)

# Hourly Data
data_site_hourly = pr.get_all_data_site()
features_hourly = ["Date", pr.PowerPV, pr.Temperature, pr.Irradiation]
data_prepared_hourly = pr.prepare_date(data_site_hourly, features_hourly)
data_cleaned_hourly = pr.clean_data(data_prepared_hourly)
data_cleaned_hourly['hour'] = data_cleaned_hourly.index.hour
data_cleaned_hourly['day'] = data_cleaned_hourly.index.day
data_cleaned_hourly['month'] = data_cleaned_hourly.index.month

data_cleaned_hourly[pr.PowerPV].plot()
pr.plt.show()


data_weather_weatherbit = prw.get_all_weather_weatherbit()
pr.check_timeindex(data_weather_weatherbit)
data_weather_darksky = prw.get_all_weather_darksky()
pr.check_timeindex(data_weather_darksky)

missing_darksky = data_weather_darksky.isnull().sum().append(pd.Series([len(data_weather_darksky)],index=['total_rows']))
missing_darksky[:-1] = missing_darksky[:-1].apply(lambda x: x * 100 /len(data_weather_darksky))

data_weather_darksky['visibility'][data_weather_darksky['visibility'].isnull()] = data_weather_darksky['visibility'].mean()

missing_weatherbit = data_weather_weatherbit['2018-04-04':].isnull().sum().append(pd.Series([len(data_weather_weatherbit['2018-04-04':])],index=['total_rows']))
missing_weatherbit[:-1] = missing_weatherbit[:-1].apply(lambda x: x * 100 /len(data_weather_weatherbit['2018']))


########################################################################################################################
#######################-------------------Create test dataset and train dataset---------------------####################
########################################################################################################################




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


data_model_featured = create_date_model_with_weather(pr.PowerPV).copy()
X_train, X_test, y_train, y_test = create_dataset(data_model)
model = svm.SVR(C=2, epsilon=0).fit(X_train,y_train)
from sklearn.externals import joblib
madatz = pytz.timezone('Indian/Antananarivo')  ## Set your timezone
madatz_now = datetime.datetime.now(madatz).date().__str__()
joblib.dump(model, 'models/model_svr_'+madatz_now+'.pkl')

model_to_pmml(model,X_train,y_train)
# part_to_predict.to_csv("Dataset_for_pmml.csv")

part_to_predict = data_model['2018-05'].copy()

predicted_df = apply_my_model_for_validation(model, part_to_predict)
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


# #############################################################################
# Fit regression model
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
n_neighbors = 5

for i, weights in enumerate(['uniform', 'distance']):
    knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
    y_ = knn.fit(X_train, y_train).predict(X_test)

    plt.subplot(2, 1, i + 1)
    plt.scatter(X_train, y_train, c='k', label='data')
    plt.plot(y_test, y_, c='g', label='prediction')
    plt.axis('tight')
    plt.legend()
    plt.title("KNeighborsRegressor (k = %i, weights = '%s')" % (n_neighbors,
                                                                weights))

plt.show()


#
# from operator import itemgetter
#
# import numpy as np
# from sklearn import grid_search
# from sklearn.model_selection import train_test_split
# from neupy import algorithms, estimators, environment
# environment.reproducible()
#
#
# def scorer(network, X, y):
#     result = network.predict(X)
#     return estimators.rmsle(result, y)
#
#
# def report(grid_scores, n_top=3):
#     scores = sorted(grid_scores, key=itemgetter(1), reverse=False)
#     for i, score in enumerate(scores[:n_top]):
#         print("Model with rank: {0}".format(i + 1))
#         print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
#               score.mean_validation_score,
#               np.std(score.cv_validation_scores)))
#         print("Parameters: {0}".format(score.parameters))
#         print("")
#
# def GRNN():
#     x_train, x_test, y_train, y_test = create_dataset(data_model['2017'])
#
#     grnnet = algorithms.GRNN(std=0.5, verbose=True)
#     grnnet.train(x_train, y_train)
#     error = scorer(grnnet, x_test, y_test)
#     print("GRNN RMSLE = {:.3f}\n".format(error))
#     part_to_predict = data_model['2018'].copy()
#     df_test = part_to_predict.copy()
#     index_predict = df_test.index
#     df_test.reset_index(inplace=True)
#     df_test.drop(["Date"], axis=1, inplace=True)
#     # fix random seed for reproducibility
#     pd.np.random.seed(7)
#     X = df_test.drop([pr.PowerPV], axis=1)
#     y = df_test.drop([x for x in df_test.columns if x not in [pr.PowerPV]], axis=1)
#     pred = grnnet.predict(X)
#     prediction_to_plot = pd.DataFrame(index=index_predict,
#                                       data={
#                                           'observed': pd.np.array(y[pr.PowerPV]),
#                                           'predicted': pred.reshape(pred.shape[0],)}
#                                       )
#     pr.plot_data(prediction_to_plot['2018-04-01':'2018-04-05'], prediction_to_plot.columns, 1)
#
#     print("Run Random Search CV")
#     grnnet.verbose = False
#     random_search = grid_search.RandomizedSearchCV(
#         grnnet,
#         param_distributions={'std': np.arange(1e-2, 1, 1e-4)},
#         n_iter=400,
#         scoring=scorer,
#     )
#     random_search.fit(data_model[[x for x in df_test.columns if x not in [pr.PowerPV]]], data_model[pr.PowerPV])
#     report(random_search.grid_scores_)



# # define base model
# def baseline_model():
#     from keras.models import Sequential
#     from keras.layers import Dense
#     model = Sequential()
#     model.add(Dense(6, input_dim=6, kernel_initializer='normal', activation='relu'))
#     model.add(Dense(1, kernel_initializer='normal'))
#     # Compile model
#     model.compile(loss='mean_squared_error', optimizer='adam')
#     return model
#
# def keras_try(data_model):
#     import numpy
#     from keras.wrappers.scikit_learn import KerasRegressor
#     from sklearn.model_selection import cross_val_score
#     from sklearn.model_selection import KFold
#     dataset = data_model['2017'].values
#     # fix random seed for reproducibility
#     X = dataset[:, 1:7]
#     y = dataset[:, 0]
#     seed = 7
#     numpy.random.seed(seed)
#     # evaluate model with standardized dataset
#     estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)
#     kfold = KFold(n_splits=10, random_state=seed)
#     results = cross_val_score(estimator, X, y, cv=kfold)
#     print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
#
#     part_to_predict = data_model['2018'].copy()
#     df_test = part_to_predict.copy()
#     index_predict = df_test.index
#     df_test.reset_index(inplace=True)
#     df_test.drop(["Date"], axis=1, inplace=True)
#     # fix random seed for reproducibility
#     pd.np.random.seed(7)
#     X = df_test.drop([pr.PowerPV], axis=1)
#     y = df_test.drop([x for x in df_test.columns if x not in [pr.PowerPV]], axis=1)
#     estimator.fit(X, y)
#     pred = estimator.predict(X)
#     prediction_to_plot = pd.DataFrame(index=index_predict,
#                                       data={
#                                           'observed': pd.np.array(y[pr.PowerPV]),
#                                           'predicted': pred}
#                                       )
#     pr.plot_data(prediction_to_plot['2018-04-01':'2018-04-05'], prediction_to_plot.columns, 1)




#Test my model on new predictions
import preprocessing as pr
from sklearn.externals import joblib
model_loaded =joblib.load('models/model_svr_2018-04-05.pkl')
predict_darsky_data = pr.pd.read_csv("Darksky_forecast/data_2018-05-31 16_47_27.csv")
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
prediction_to_plot = pd.DataFrame(index=predict_darsky_data.index,
                                  data={
                                      'observed' : data_cleaned_hourly[pr.PowerPV].loc[predict_darsky_data.index],
                                      'predicted': predicted_df}
                                  )
pr.plot_data(prediction_to_plot,prediction_to_plot.columns)
madatz = pytz.timezone('Indian/Antananarivo')  ## Set your timezone
madatz_now = datetime.datetime.now(madatz).date().__str__()
prediction_to_plot.to_csv("data_predicted_with_model/predictionfor"+madatz_now+".csv")

import pandas as pd
prediction_to_plot=pd.read_csv(r"C:\Users\G603344\PycharmProjects\POCv1.0\data_predicted_with_model\predictionfor2018-05-31.csv")
prediction_to_plot.set_index(prediction_to_plot["time.1"], inplace=True)
prediction_to_plot.index = prediction_to_plot.index.tz_convert('Indian/Antananarivo').tz_localize(None)












