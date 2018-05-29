######---------------test all
# def run_all_models(dataframe):
    # X_train, X_test, y_train, y_test = create_dataset(dataframe)
    # classifiers = [
    #     svm.SVR(),
    #     linear_model.BayesianRidge(),
    #     linear_model.LassoLars(),
    #     linear_model.ARDRegression(),
    #     linear_model.PassiveAggressiveRegressor(),
    #     linear_model.TheilSenRegressor()]
    # for item in classifiers:
    #     print("##################")
    #     print(item.__str__().split("(")[0])
    #     clf = item
    #     clf.fit(pd.np.array(X_train), pd.np.array(y_train))
    #     pred = clf.predict(pd.np.array(X_test))
    #     rms = sqrt(mean_squared_error(pd.np.array(y_test), pred))
    #     prediction_to_plot = pd.DataFrame({'observed':pd.np.array(y_test[pr.PowerPV]), 'predicted': pred})
    #     x = prediction_to_plot[:48].index
    #     fig = plt.figure()
    #     for i_feature in prediction_to_plot.columns:
    #         plt.plot(x, prediction_to_plot[i_feature][:48], label=str(i_feature))
    #     plt.title(item.__str__().split("(")[0])
    #     plt.legend(loc='best')
    #     file_name = 'results/'+item.__str__().split("(")[0]
    #     plt.savefig(file_name)
    #     plt.close(fig)
    #     print("the rmse : "+ str(rms))
    #     print("##################")
    # print("END")
############################################################


# for fbproph

#################################################################################################################
################################----------------------Daily----------------------################################
#################################################################################################################
# import fbprophet as ph
# # Prophete Facebok
# data_train = data_cleaned.copy()
# data_train.reset_index(level=0, inplace=True)
# # Prophet requires columns ds (Date) and y (value)
# data_train = data_train.rename(columns={'Date': 'ds', pr.PowerPV: 'y'})
# data_train.drop([x for x in data_train.columns.tolist() if x not in ['ds','y']],axis=1,inplace=True)
# # Make the prophet model and fit on the data
# periode = 30
# model = ph.Prophet(changepoint_prior_scale=0.01)
# mask = (data_train['ds'] > '2017') & (data_train['ds'] < '2018-02')
# model.fit(data_train.loc[mask])
# # Make a future dataframe for 1 month = 30 Day
# future_powerpv = model.make_future_dataframe(periods=periode, freq='D')
# # Make predictions
# forecast_powerpv = model.predict(future_powerpv)
#
# model.plot_components(forecast_powerpv)
# # model.plot_components(forecast_powerpv) # plot the trend
#
# forecast = forecast_powerpv['yhat'].iloc[-periode:]
# observed = data_cleaned[forecast_powerpv.iloc[forecast.index[0],0].__str__():forecast_powerpv.iloc[forecast.index[-1],0].__str__()][pr.PowerPV]
#
# #Plot observed vs predicted
# # date_start_prediction  = data_train.loc[len(data_train.loc[mask])-1,'ds']
# # days = pd.date_range(date_start_prediction + timedelta(days=1), date_start_prediction + timedelta(days=periode), freq='D')
# prediction_to_plot = pd.DataFrame({'observed':observed.values, 'predicted': forecast.values}, index=observed.index)
# pr.plot_data(prediction_to_plot,prediction_to_plot.columns,1)

#################################################################################################################
################################----------------------Hourly----------------------###############################
#################################################################################################################

# #Prophete Facebok
# import fbprophet as ph
# data_train_hourly = data_cleaned_hourly.copy()
# data_train_hourly.reset_index(level=0, inplace=True)
# # Prophet requires columns ds (Date) and y (value)
# data_train_hourly = data_train_hourly.rename(columns={'Date': 'ds', pr.PowerPV: 'y'})
# data_train_hourly.drop([x for x in data_train_hourly.columns.tolist() if x not in ['ds','y']],axis=1,inplace=True)
#
# periode_hourly = 3*24
# model = ph.Prophet(changepoint_prior_scale=0.01,yearly_seasonality=False,weekly_seasonality=False,daily_seasonality=True)
# model.add_seasonality(name='daily', period=12, fourier_order=5)
# mask = (data_train_hourly['ds'] > '2017') & (data_train_hourly['ds'] < '2018')
# model.fit(data_train_hourly.loc[mask])
# # Make a future dataframe for 24hour
# future_powerpv = model.make_future_dataframe(periods=periode_hourly, freq='H')
# # Make predictions
# forecast_powerpv = model.predict(future_powerpv)
# # model.plot_components(forecast_powerpv).savefig(str(i)+'.png');
#
# # model.plot_components(forecast_powerpv) # plot the trend
# forecast = forecast_powerpv['yhat'].iloc[-periode_hourly:]
# observed = data_cleaned_hourly[forecast_powerpv.iloc[forecast.index[0],0].__str__():forecast_powerpv.iloc[forecast.index[-1],0].__str__()][pr.PowerPV]
# #Plot observed vs predicted
# # date_start_prediction  = data_train_hourly.loc[len(data_train_hourly.loc[mask])-1,'ds']
# # days = pd.date_range(date_start_prediction + timedelta(hours=1), date_start_prediction + timedelta(hours=periode_hourly), freq='H')
# prediction_to_plot = pd.DataFrame({'observed':observed.values, 'predicted': forecast.values}, index=observed.index)
# pr.plot_data(prediction_to_plot,prediction_to_plot.columns,1)


########################################################################################################################
#########################----------------------- Arima Model ---------------------######################################
########################################################################################################################

# X = data_to_plot.values
# size = int(len(X) * 0.66)
# train, test = X[0:size], X[size:len(X)]
# history = [x for x in train]
# predictions = list()
# for t in range(len(test)):
#     model = ARIMA(history, order=(5, 1, 0))
#     model_fit = model.fit(disp=0)
#     output = model_fit.forecast(steps=7)
#     yhat = output[0]
#     predictions.append(yhat)
#     obs = test[t]
#     history.append(obs)
#
# error = mean_squared_error(test, predictions)
# print('Test MSE: %.3f' % error)
# # plot
# plt.plot(test)
# plt.plot(predictions, color='red')
# plt.show()


##########


########################################################################################################################
#####################Comparaison entre Irradiation API et Irradiation panel#############################################
########################################################################################################################
# import preprocessing_weather as prw
# data_weather = prw.get_all_weather()
# irradiance_api = pd.DataFrame(data_weather['2018-04-01 00:00:00':'2018-04-05 07:00:00']['temp'])
# irradiance_api["Temperature_panel"] = data_cleaned_hourly['2018-04-01 00:00:00':'2018-04-05 07:00:00'][pr.Temperature]
# irradiance_api["Irradiation_panel"] = data_cleaned_hourly['2018-04-01 00:00:00':'2018-04-05 07:00:00'][pr.Irradiation]
# irradiance_api["dhi"] = data_weather['2018-04-01 00:00:00':'2018-04-05 07:00:00']['dhi']
# irradiance_api["dni"] = data_weather['2018-04-01 00:00:00':'2018-04-05 07:00:00']['dni'] # the good one
# irradiance_api["ghi"] = data_weather['2018-04-01 00:00:00':'2018-04-05 07:00:00']['ghi']
# a = irradiance_api.corr()
# pr.plot_data(irradiance_api,irradiance_api.columns,1)
# pr.plot_data_correlation(irradiance_api,irradiance_api.columns)


########################################################################################################################
#####################---------------------Arima----------------------------#############################################
########################################################################################################################

# from pandas import Series
# from matplotlib import pyplot
# from statsmodels.tsa.arima_model import ARIMA
# from sklearn.metrics import mean_squared_error
# from math import sqrt
#
# def predict(coef, history):
# 	yhat = 0.0
# 	for i in range(1, len(coef)+1):
# 		yhat += coef[i-1] * history[-i]
# 	return yhat
#
# data_train_arima = data_cleaned_hourly.copy()
# X = data_train_arima[pr.PowerPV].values
# size = int(len(X) * 0.66)
# train, test = X[0:size], X[size:len(X)]
# history = [x for x in train]
# predictions = list()
# model = ARIMA(history, order=(1, 0, 0))
# model_fit = model.fit(trend='nc', disp=False)
# ar_coef = model_fit.arparams
# for t in range(len(test)):
# 	yhat = predict(ar_coef, history)
# 	predictions.append(yhat)
# 	obs = test[t]
# 	print('>predicted=%.3f, expected=%.3f' % (yhat, obs))
# rmse = sqrt(mean_squared_error(test, predictions))
# print('Test RMSE: %.3f' % rmse)
#
# # model.plot_components(forecast_powerpv) # plot the trend
# prediction_to_plot = pd.DataFrame({'observed':test[1], 'predicted': forecast}, index=data_cleaned_hourly[pr.PowerPV].iloc[-len(forecast):].index)
# pr.plot_data(prediction_to_plot['2018-04'],prediction_to_plot.columns,1)
# forecast = model_fit.forecast(steps=7)[0]

#########################################################################################################
###########------triple_exponential_smoothing(Holt-Winters Forecasting)( add seasonality effect)------###
#########################################################################################################
# def initial_trend(series, slen):
#     sum = 0.0
#     for i in range(slen):
#         sum += float(series[i+slen] - series[i]) / slen
#     return sum / slen
# def initial_seasonal_components(series, slen):
#     seasonals = {}
#     season_averages = []
#     n_seasons = int(len(series)/slen)
#     # compute season averages
#     for j in range(n_seasons):
#         season_averages.append(sum(series[slen*j:slen*j+slen])/float(slen))
#     # compute initial values
#     for i in range(slen):
#         sum_of_vals_over_avg = 0.0
#         for j in range(n_seasons):
#             sum_of_vals_over_avg += series[slen*j+i]-season_averages[j]
#         seasonals[i] = sum_of_vals_over_avg/n_seasons
#     return seasonals
# def triple_exponential_smoothing(series, slen, alpha, beta, gamma, n_preds):
#     result = []
#     seasonals = initial_seasonal_components(series, slen)
#     for i in range(len(series)+n_preds):
#         if i == 0: # initial values
#             smooth = series[0]
#             trend = initial_trend(series, slen)
#             result.append(series[0])
#             continue
#         if i >= len(series): # we are forecasting
#             m = i - len(series) + 1
#             result.append((smooth + m*trend) + seasonals[i%slen])
#         else:
#             val = series[i]
#             last_smooth, smooth = smooth, alpha*(val-seasonals[i%slen]) + (1-alpha)*(smooth+trend)
#             trend = beta * (smooth-last_smooth) + (1-beta)*trend
#             seasonals[i%slen] = gamma*(val-smooth) + (1-gamma)*seasonals[i%slen]
#             result.append(smooth+trend+seasonals[i%slen])
#     return result
# data_train_arima = data_cleaned_hourly.copy()
# data_train_arima.drop([pr.Irradiation,pr.Temperature],axis=1,inplace=True)
# slice_date = '2017'
# hour_to_forcast = 3*24
# alpha = 0.06
# beta = 0.09
# gamma =  0.06
# size = int(len(data_train_arima)-hour_to_forcast)
# train, test = data_train_arima[0:size], data_train_arima[size:len(data_train_arima)]
# forecast = triple_exponential_smoothing(data_train_arima[pr.PowerPV][slice_date], 24, alpha, beta,gamma, hour_to_forcast)
# from datetime import timedelta
# forecast = forecast[-hour_to_forcast:]
# start_date = data_train_arima[pr.PowerPV][slice_date].index[-1] + timedelta(hours=1)
# end_date = start_date + timedelta(hours=hour_to_forcast-1)
# observed = data_train_arima[start_date.__str__():end_date.__str__()][pr.PowerPV]
# prediction_to_plot = pd.DataFrame({'observed':observed.values, 'predicted': forecast}, index=observed.index)
# pr.plot_data(prediction_to_plot,prediction_to_plot.columns,1)

#########################################################################################################
############################------Average of each hour------#############################################
#########################################################################################################
# data_train_arima = data_cleaned_hourly.copy()
# data_train_arima.drop([pr.Irradiation,pr.Temperature],axis=1,inplace=True)
# slice_date = '2018-01-03'
# X = data_train_arima[pr.PowerPV][:slice_date]
# hours_forecast = 3*24
# size = int(len(X)- hours_forecast)
# train, test = data_train_arima[0:size], data_train_arima[size:len(X)]
# dict_mean_hours = {}
# for i_hours in range(0,24):
#     dict_mean_hours[i_hours] = train[(train.index.hour == i_hours)].mean().values[0].round(3)
# forecast = [dict_mean_hours[x] for x in test.index.hour]
# prediction_to_plot = pd.DataFrame({'observed':test[pr.PowerPV].values, 'predicted': forecast}, index=test.index)
# pr.plot_data(prediction_to_plot,prediction_to_plot.columns,1)
