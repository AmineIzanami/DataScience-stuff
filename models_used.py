import matplotlib.pyplot as plt
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
######Create data return x_test, x_train,y_test,y_train
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
#####---------------test all
def run_all_models(dataframe):
    X_train, X_test, y_train, y_test = create_dataset(dataframe)
    classifiers = [
        svm.SVR(),
        linear_model.BayesianRidge(),
        linear_model.LassoLars(),
        linear_model.ARDRegression(),
        linear_model.PassiveAggressiveRegressor(),
        linear_model.TheilSenRegressor()]
    for item in classifiers:
        print("##################")
        print(item.__str__().split("(")[0])
        clf = item
        clf.fit(pd.np.array(X_train), pd.np.array(y_train))
        pred = clf.predict(pd.np.array(X_test))
        rms = sqrt(mean_squared_error(pd.np.array(y_test), pred))
        prediction_to_plot = pd.DataFrame({'observed':pd.np.array(y_test[pr.PowerPV]), 'predicted': pred})
        x = prediction_to_plot[:48].index
        fig = plt.figure()
        for i_feature in prediction_to_plot.columns:
            plt.plot(x, prediction_to_plot[i_feature][:48], label=str(i_feature))
        plt.title(item.__str__().split("(")[0])
        plt.legend(loc='best')
        file_name = 'results/'+item.__str__().split("(")[0]
        plt.savefig(file_name)
        plt.close(fig)
        print("the rmse : "+ str(rms))
        print("##################")
    print("END")
###########################################################
#########Validate the model with test dataframe
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

##########test all classifiers on the dataframe, return a dataframe with metrics
def test_all_classfiers(dataframe,feature_to_predict):
    X_train, X_test, y_train, y_test = create_dataset(dataframe,feature_to_predict)
    classifiers = [
        svm.SVR(),
        RandomForestRegressor(max_depth=2, random_state=0),
        linear_model.BayesianRidge(),
        linear_model.LassoLars(),
        linear_model.TheilSenRegressor()]
    df_result_metric = pd.DataFrame(index= [ item.__str__().split("(")[0] for item in classifiers]+["NeuralNetwork"],columns=['mse','rmse','r2','correlation'])
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


###### forward selection iterate throught all possible combinasion of features and return metric
def forward_selection_features(data_model):
    data_model = data_model.copy()
    feature_to_predict = pr.PowerPV
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
    df_result_metric_forward = pd.DataFrame(index=[','.join(i) for i in all_combi],
                                    columns=['mse', 'rmse', 'r2', 'correlation'])
    df_result_metric_forward.sort_index(inplace=True)
    for i in range(len(all_combi)):
        features_validation = data_model[[pr.PowerPV]+all_combi[i]].copy()
        print("Validation for :" + all_combi[i].__str__())
        #test the best model SVR with default hyperparameters
        mask_observed = (features_validation.index > '2017') & (features_validation.index < '2018-06-01 00:00:00')
        data_model_observed = features_validation.loc[mask_observed]
        X_train, X_test, y_train, y_test = create_dataset(data_model_observed['2017'],feature_to_predict,portion=0.1)
        model = svm.SVR(C=2, epsilon=0).fit(X_train, y_train)
        mask_to_predict = (features_validation.index > '2018-06-01 00:00:00')
        part_to_predict = features_validation.loc[mask_to_predict]
        predicted_df = apply_my_model_for_validation(model, part_to_predict,feature_to_predict)
        mse = mean_squared_error(predicted_df.observed, predicted_df.predicted)
        mae = mean_absolute_error(predicted_df.observed, predicted_df.predicted)
        r2 = r2_score(predicted_df.observed, predicted_df.predicted)
        correlation = pd.np.corrcoef(pd.np.array(predicted_df.observed), predicted_df.predicted)
        rmse = sqrt(mean_squared_error(predicted_df.observed, predicted_df.predicted))
        # df_result_metric_mse += [mse]
        # df_result_metric_mae += [mae]
        # df_result_metric_r2 += [r2]
        # df_result_metric_correlation += [correlation]
        # df_result_metric_rmse += [rmse]

        df_result_metric_forward.loc[','.join(all_combi[i])]['mse'] = mse
        df_result_metric_forward.loc[','.join(all_combi[i])]['rmse'] = rmse
        df_result_metric_forward.loc[','.join(all_combi[i])]['r2'] = r2
        df_result_metric_forward.loc[','.join(all_combi[i])]['correlation'] = correlation[0, 1]
        df_result_metric_forward_float = df_result_metric_forward.astype(float)
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
#
# writer = pd.ExcelWriter('output.xlsx')
# df_result_metric_forward_float.to_excel(writer,'Sheet1')
# writer.save()
###### return p_value and other stats regression
def stats_models_summary(X_train,X_test,y_train):
    import statsmodels.api as sm
    model = sm.OLS(y_train, X_train).fit()
    predictions = model.predict(X_test)
    model.summary()


#####dump model to PMML structure
def model_to_pmml(model,X_train,y_train):
    from sklearn2pmml.pipeline import PMMLPipeline
    power_pipeline = PMMLPipeline([
        ("classifier",model)
    ])

    power_pipeline.fit(X_train, y_train)
    from sklearn2pmml import sklearn2pmml
    sklearn2pmml(power_pipeline, "LogisticRegressionPowerPV.pmml", with_repr = True)

####keras architecture plot observed vs predicted
# define base model
def baseline_model():
    from keras.models import Sequential
    from keras.layers import Dense
    model = Sequential()
    model.add(Dense(4, input_dim=4, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
def keras_try(data_model_featured, feature_to_predict):
    import numpy
    from keras.wrappers.scikit_learn import KerasRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import KFold

    data_model_featured = data_model_observed
    mask_observed = (data_model_featured.index > '2017') & (data_model_featured.index < '2018-06-03 00:00:00')

    data_model_observed = data_model_featured.loc[mask_observed]

    dataset = data_model_observed.values
    # fix random seed for reproducibility
    X = dataset[:, 1:7]
    y = dataset[:, 0]
    seed = 7
    numpy.random.seed(seed)
    # evaluate model with standardized dataset
    estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)
    kfold = KFold(n_splits=10, random_state=seed)
    results = cross_val_score(estimator, X, y, cv=kfold)
    print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))



    mask_to_predict = (data_model_featured.index > '2018-06-03 00:00:00')

    part_to_predict = data_model_featured.loc[mask_to_predict]

    df_test = part_to_predict.copy()
    index_predict = df_test.index
    df_test.reset_index(inplace=True)
    df_test.drop(["Date"], axis=1, inplace=True)
    # fix random seed for reproducibility
    pd.np.random.seed(7)
    X = df_test.drop([feature_to_predict], axis=1)
    y = df_test.drop([x for x in df_test.columns if x not in [feature_to_predict]], axis=1)
    estimator.fit(X, y)
    pred = estimator.predict(X)
    prediction_to_plot = pd.DataFrame(index=index_predict,
                                      data={
                                          'observed': pd.np.array(y[feature_to_predict]),
                                          'predicted': pred}
                                      )
    predicted_df = prediction_to_plot.copy()
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(predicted_df.observed, predicted_df.predicted)
    rmse = sqrt(mean_squared_error(predicted_df.observed, predicted_df.predicted))
    r2 = r2_score(predicted_df.observed, predicted_df.predicted)
    correlation = pd.np.corrcoef(predicted_df.observed, predicted_df.predicted)
    # Store Metrics for regression

    result_test_BC.loc["RandomForestRegressor"]['mse'] = 0.006874955
    result_test_BC.loc["RandomForestRegressor"]['rmse'] = 0.089859954
    result_test_BC.loc["RandomForestRegressor"]['r2'] = 0.326595
    result_test_BC.loc["RandomForestRegressor"]['correlation'] = 0.65248778

    result_test_BC_float = result_test_BC.astype(float)

    pr.plot_data(prediction_to_plot, prediction_to_plot.columns, 1)
############################################################
#####GRNN model with grid seach param
from operator import itemgetter
import numpy as np
from sklearn import grid_search
from sklearn.model_selection import train_test_split
from neupy import algorithms, estimators, environment
environment.reproducible()
def scorer(network, X, y):
    result = network.predict(X)
    return estimators.rmsle(result, y)
def report(grid_scores, n_top=3):
    scores = sorted(grid_scores, key=itemgetter(1), reverse=False)
    for i, score in enumerate(scores[:n_top]):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")
def GRNN(data_model):
    x_train, x_test, y_train, y_test = create_dataset(data_model['2017'])

    grnnet = algorithms.GRNN(std=0.5, verbose=True)
    grnnet.train(x_train, y_train)
    error = scorer(grnnet, x_test, y_test)
    print("GRNN RMSLE = {:.3f}\n".format(error))
    part_to_predict = data_model['2018'].copy()
    df_test = part_to_predict.copy()
    index_predict = df_test.index
    df_test.reset_index(inplace=True)
    df_test.drop(["Date"], axis=1, inplace=True)
    # fix random seed for reproducibility
    pd.np.random.seed(7)
    X = df_test.drop([pr.PowerPV], axis=1)
    y = df_test.drop([x for x in df_test.columns if x not in [pr.PowerPV]], axis=1)
    pred = grnnet.predict(X)
    prediction_to_plot = pd.DataFrame(index=index_predict,
                                      data={
                                          'observed': pd.np.array(y[pr.PowerPV]),
                                          'predicted': pred.reshape(pred.shape[0],)}
                                      )
    pr.plot_data(prediction_to_plot['2018-04-01':'2018-04-05'], prediction_to_plot.columns, 1)

    print("Run Random Search CV")
    grnnet.verbose = False
    random_search = grid_search.RandomizedSearchCV(
        grnnet,
        param_distributions={'std': np.arange(1e-2, 1, 1e-4)},
        n_iter=400,
        scoring=scorer,
    )
    random_search.fit(data_model[[x for x in df_test.columns if x not in [pr.PowerPV]]], data_model[pr.PowerPV])
    report(random_search.grid_scores_)

####################PLOT residuals
def residual_predicted_df(predicted_df):
    import seaborn as sns
    sns.set(style="whitegrid")
    # Plot the residuals after fitting a linear model
    sns.residplot(predicted_df.observed.values, predicted_df.predicted.values, lowess=True, color="g");pr.plt.show()

################################################################################################################
###############################----------------------Daily----------------------################################
################################################################################################################
def fbprophet_Daily(data_cleaned,feature_to_predict):

    import fbprophet as ph
    import pandas as pd
    import preprocessing as pr
    # Prophete Facebok
    data_train = data_cleaned.copy()

    data_train.reset_index(level=0, inplace=True)
    # Prophet requires columns ds (Date) and y (value)
    data_train = data_train.rename(columns={'Date': 'ds',feature_to_predict: 'y'})
    data_train.drop([x for x in data_train.columns.tolist() if x not in ['ds','y']],axis=1,inplace=True)
    # Make the prophet model and fit on the data
    periode = 30
    model = ph.Prophet(changepoint_prior_scale=0.01)
    mask = (data_train['ds'] > '2017') & (data_train['ds'] < '2018-06-01 00:00:00')
    model.fit(data_train.loc[mask])
    # Make a future dataframe for 1 month = 30 Day
    future_powerpv = model.make_future_dataframe(periods=periode, freq='D')
    # Make predictions
    forecast_powerpv = model.predict(future_powerpv)

    # model.plot_components(forecast_powerpv)
    # model.plot_components(forecast_powerpv) # plot the trend

    forecast = forecast_powerpv['yhat'].iloc[-periode:]
    observed = data_cleaned[forecast_powerpv.iloc[forecast.index[0],0].__str__():forecast_powerpv.iloc[forecast.index[-1],0].__str__()][feature_to_predict]

    #Plot observed vs predicted
    # date_start_prediction  = data_train.loc[len(data_train.loc[mask])-1,'ds']
    # days = pd.date_range(date_start_prediction + timedelta(days=1), date_start_prediction + timedelta(days=periode), freq='D')
    prediction_to_plot = pd.DataFrame({'observed':observed.values, 'predicted': forecast.values}, index=observed.index)
    pr.plot_data(prediction_to_plot,prediction_to_plot.columns,1)

################################################################################################################
###############################----------------------Hourly----------------------###############################
################################################################################################################
def fbprophet_hourly(data_cleaned_hourly,feature_to_predict):
    #Prophete Facebok
    import fbprophet as ph
    import pandas as pd
    import preprocessing as pr

    data_train_hourly = data_cleaned_hourly.copy()

    data_train_hourly.reset_index(level=0, inplace=True)
    # Prophet requires columns ds (Date) and y (value)
    data_train_hourly = data_train_hourly.rename(columns={'Date': 'ds',feature_to_predict: 'y'})
    data_train_hourly.drop([x for x in data_train_hourly.columns.tolist() if x not in ['ds','y']],axis=1,inplace=True)

    periode_hourly = 3*24
    model = ph.Prophet(changepoint_prior_scale=0.01,yearly_seasonality=False,weekly_seasonality=False,daily_seasonality=True)
    model.add_seasonality(name='daily', period=12, fourier_order=5)
    mask = (data_train_hourly['ds'] > '2017') & (data_train_hourly['ds'] < '2018-06-01 00:00:00')
    model.fit(data_train_hourly.loc[mask])
    # Make a future dataframe for 24hour
    future_powerpv = model.make_future_dataframe(periods=periode_hourly, freq='H')
    # Make predictions
    forecast_powerpv = model.predict(future_powerpv)
    # model.plot_components(forecast_powerpv).savefig(str(i)+'.png');

    # model.plot_components(forecast_powerpv) # plot the trend
    forecast = forecast_powerpv['yhat'].iloc[-periode_hourly:]
    observed = data_cleaned_hourly[forecast_powerpv.iloc[forecast.index[0],0].__str__():forecast_powerpv.iloc[forecast.index[-1],0].__str__()][feature_to_predict]
    #Plot observed vs predicted
    # date_start_prediction  = data_train_hourly.loc[len(data_train_hourly.loc[mask])-1,'ds']
    # days = pd.date_range(date_start_prediction + timedelta(hours=1), date_start_prediction + timedelta(hours=periode_hourly), freq='H')
    prediction_to_plot = pd.DataFrame({'observed':observed.values, 'predicted': forecast.values}, index=observed.index)
    pr.plot_data(prediction_to_plot,prediction_to_plot.columns,1)




#######################################################################################################################
####################---------------------Arima----------------------------#############################################
#######################################################################################################################
def arima(data_cleaned_hourly,feature_to_predict):
    from statsmodels.tsa.arima_model import ARIMA
    from sklearn.metrics import mean_squared_error
    from math import sqrt

    def predict(coef, history):
        yhat = 0.0
        for i in range(1, len(coef)+1):
            yhat += coef[i-1] * history[-i]
        return yhat

    data_train_arima = data_cleaned_hourly.copy()
    X = data_train_arima[feature_to_predict].values
    size = int(len(X) * 0.66)
    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train]
    predictions = list()
    model = ARIMA(history, order=(1, 0, 0))
    model_fit = model.fit(trend='nc', disp=False)
    ar_coef = model_fit.arparams
    for t in range(len(test)):
        yhat = predict(ar_coef, history)
        predictions.append(yhat)
        obs = test[t]
        print('>predicted=%.3f, expected=%.3f' % (yhat, obs))
    rmse = sqrt(mean_squared_error(test, predictions))
    print('Test RMSE: %.3f' % rmse)

    # model.plot_components(forecast_powerpv) # plot the trend
    # prediction_to_plot = pd.DataFrame({'observed':test[1], 'predicted': forecast}, index=data_cleaned_hourly[pr.PowerPV].iloc[-len(forecast):].index)
    # pr.plot_data(prediction_to_plot['2018-04'],prediction_to_plot.columns,1)
    # forecast = model_fit.forecast(steps=7)[0]

########################################################################################################
##########------triple_exponential_smoothing(Holt-Winters Forecasting)( add seasonality effect)------###
########################################################################################################
def initial_trend(series, slen):
    sum = 0.0
    for i in range(slen):
        sum += float(series[i+slen] - series[i]) / slen
    return sum / slen
def initial_seasonal_components(series, slen):
    seasonals = {}
    season_averages = []
    n_seasons = int(len(series)/slen)
    # compute season averages
    for j in range(n_seasons):
        season_averages.append(sum(series[slen*j:slen*j+slen])/float(slen))
    # compute initial values
    for i in range(slen):
        sum_of_vals_over_avg = 0.0
        for j in range(n_seasons):
            sum_of_vals_over_avg += series[slen*j+i]-season_averages[j]
        seasonals[i] = sum_of_vals_over_avg/n_seasons
    return seasonals
def triple_exponential_smoothing(series, slen, alpha, beta, gamma, n_preds):
    result = []
    seasonals = initial_seasonal_components(series, slen)
    for i in range(len(series)+n_preds):
        if i == 0: # initial values
            smooth = series[0]
            trend = initial_trend(series, slen)
            result.append(series[0])
            continue
        if i >= len(series): # we are forecasting
            m = i - len(series) + 1
            result.append((smooth + m*trend) + seasonals[i%slen])
        else:
            val = series[i]
            last_smooth, smooth = smooth, alpha*(val-seasonals[i%slen]) + (1-alpha)*(smooth+trend)
            trend = beta * (smooth-last_smooth) + (1-beta)*trend
            seasonals[i%slen] = gamma*(val-smooth) + (1-gamma)*seasonals[i%slen]
            result.append(smooth+trend+seasonals[i%slen])
    return result
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


