import statistics

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import statsmodels.tsa.api as smt
from statsmodels.tsa.stattools import adfuller
import glob
import api_def_smart_village as ap

# UsefullParams
DryLoopA1 = "Delestage_Actif_0.MC - State"
DryLoopA2 = "Delestage_Actif_0.BC - State"
K01 = "Delestage_Actif_0.MC - Cmd"
K02 = "Delestage_Actif_0.BC - Cmd"
PowerPV = "PV_1 - P"
PowerBattery = "DC Sys.Batt.Out - P"
Temperature = "PV_1 - Temp"
Soc = "DC Sys.Batt - SoC"
Irradiation = "Solar_Rad - W/m2"
PowerEol1 = "Eol_1 - P"
PowerEol2 = "Eol_2 - P"

data_site_usefull_columns = ["Date",
                             DryLoopA1,
                             DryLoopA2,
                             # K01,
                             # K02,
                             PowerPV,
                             PowerBattery,
                             Temperature,
                             Soc,
                             Irradiation,
                             # PowerEol1,
                             # PowerEol2
                             ]

def get_all_data_site():
    path =r'data_site_madagascar/' # use your path
    allFiles = glob.glob(path + "/*.csv")
    frame = pd.DataFrame()
    list_ = []
    for file_ in allFiles:
        df = pd.read_csv(file_,sep=",",index_col=0, header=0)
        list_.append(df)
    frame = pd.concat(list_)
    frame.drop_duplicates(['Date'], keep='first',inplace=True)
    return frame


import pytz
def prepare_date(data_site, features):
    '''
    Prepare the dataframe by converting the date the "datatime64" structure and return the specific field to study eg : PowerPV, irradiation
    Return dataframe with "Date", PowerPV,Temperature,Irradiation
    '''
    data_site_usefull = data_site[data_site_usefull_columns].copy()
    # take the value of "Date" convert to a date format with millis_to_date, apply strptime to use the datetime64 type benefits (like getting only the value of a specific month eg :data_to_plot['2018-04']);
    # data_site_usefull["Date"] = data_site_usefull["Date"].apply(
    #     lambda x: pd.datetime.strptime(str(ap.millis_to_date(x)), '%Y-%m-%d %H:%M:%S'))
    data_site_usefull["Date"] = data_site_usefull["Date"]/1000
    # data_site_usefull[[item for item in data_site_usefull.columns if item not in ["Date"]]] = data_site_usefull[[item for item in data_site_usefull.columns if item not in ["Date"]]].astype(float)
    data_to_plot = data_site_usefull[features].copy()
    data_to_plot.set_index(pd.to_datetime(data_to_plot["Date"], unit='s', utc=True), inplace=True)
    # data_to_plot = data_to_plot.set_index('Date')
    data_to_plot.drop(['Date'], 1, inplace=True)
    data_to_plot.index = data_to_plot.index.tz_convert('Indian/Antananarivo').tz_localize(None)
    return data_to_plot


def summary_missing_value(data_not_cleaned):
    '''
    Calculate the missing values related to the dataframe and plot the percentage / total values
    '''
    # get the hours where missing is showing
    df_tmp = data_not_cleaned.copy()
    missing = df_tmp[df_tmp[PowerPV].astype(str).str.contains("Missing", na=False)]
    missing["hour"] = missing.index.hour
    # show the frequency of the missing values
    frequency_hours_missing = missing.hour.value_counts().to_dict()
    # here to get the frequency of hours
    df_tmp["hour"] = df_tmp.index.hour
    frequency_hours_total = df_tmp.hour.value_counts().to_dict()
    # Summary of missing  / total values
    summary_missing_dict = {x: (frequency_hours_missing[x] / frequency_hours_total[x]) * 100 for x in
                            frequency_hours_missing}
    summary_missing_dict_sorted = {}
    for key in sorted(summary_missing_dict):
        summary_missing_dict_sorted[key] = summary_missing_dict[key]
    return summary_missing_dict_sorted


def clean_data(data_prepared):
    '''
    * Clean the dataframe by replacing the missing value in the off_hour(no sun) by '0'
    * Deleting the rows with missing values for the on_hour(sun) and the the abberant data within a threshold eg : 50
    '''
    print("Début du nettoyage de donnée \r \r")
    df_tmp = data_prepared.copy()
    summary_value = summary_missing_value(df_tmp)
    df_tmp["hour"] = df_tmp.index.hour
    list_off_hour = [x for x in summary_value.keys() if summary_value[
        x] == 100]  # off_hours are hours without sun, so there is no data in that range of hours
    df_tmp[PowerPV][df_tmp["hour"].isin(list_off_hour)] = 0
    print("Number of off-hours with missing value for PowerPV (Replaced by 0) :",
          len(df_tmp[PowerPV][df_tmp["hour"].isin(list_off_hour)]), "|| percentage/total :",
          (len(df_tmp[PowerPV][df_tmp["hour"].isin(list_off_hour)]) / len(df_tmp)) * 100, "%")
    df_tmp.drop(["hour"], axis=1, inplace=True)

    # replace the on_hours that have missing values with
    # the mean of the according hour, base on a dictionnary
    for feature in data_prepared.columns:
        print("\r\r")
        print("##########################################")
        print("Préprocessing pour la variable " + feature)
        print("##########################################")
        on_hour_missing = df_tmp[df_tmp[feature].astype(str) == "Missing"].index
        print("Number of on-hours with missing value(replaced by the mean) for " + feature + " :", len(on_hour_missing),
              "|| percentage/total :", (len(on_hour_missing) / len(df_tmp)) * 100, "%")
        # Debug :         feature = pr.Irradiation
        with_no_missing = df_tmp[df_tmp[feature].astype(str) != "Missing"]
        with_no_missing[feature] = with_no_missing[feature].astype(float)
        hours_healthy = with_no_missing[with_no_missing[feature] > 0].index.hour.unique()
        dict_value_for_missing = {}
        for i_hours in hours_healthy:
            dict_value_for_missing[i_hours] = \
                with_no_missing[(with_no_missing.index.hour == i_hours) & (with_no_missing[feature] > 0)].mean().values[
                    0].round(3)

        for off_hour in range(0, 24):
            if off_hour not in dict_value_for_missing:
                dict_value_for_missing[off_hour] = 0

        with_missing = df_tmp[df_tmp[feature].astype(str).str.contains("Missing", na=False)]

        for i_missing in with_missing.index:
            df_tmp.loc[i_missing, feature] = dict_value_for_missing[i_missing.hour]

        df_tmp[feature] = df_tmp[feature].astype(float)

        threshold = 50
        aberrant_data = df_tmp[df_tmp[feature] < 0]
        numbers = [dict_value_for_missing[key] for key in dict_value_for_missing]
        mean_ = statistics.mean(numbers)
        df_tmp.loc[df_tmp[feature]<0, feature] = mean_
        # print("mean " + str(mean_))
        print("Number of aberrant data(replaced by the mean) for " + feature + " :", len(aberrant_data),
              "|| percentage /total :", (len(aberrant_data) / len(df_tmp)) * 100, "%")
        # df_tmp.drop(aberrant_data.index,inplace=True)

    return df_tmp


def plot_data(data_to_plot, features, mode=1):
    '''
      Plot the dataframe eg : PowerPV/Date possible withing a specific month eg: 2018-01 data_to_plot['2018-01']
      mode : 0 = plot each column in separate plots
             1 = plot the columns on the same plot
    '''
    if (mode == 0):
        fig, axes = plt.subplots(nrows=len(features), ncols=1, sharex=True)
        x = data_to_plot.index
        # for i_feature in features[1:]:
        for ax, feature in zip(axes.flat, features):
            ax.plot(x, data_to_plot[feature])  # ,label=str(i_feature))
            ax.tick_params(axis='x', rotation=70)
            ax.set_title(str(feature))
            ax.grid(True)
            plt.legend(loc='best')
            plt.tight_layout();
        plt.show()
    else:
        fig, ax = plt.subplots()
        x = data_to_plot.index
        for i_feature in features:
            plt.plot(x, data_to_plot[i_feature], label=str(i_feature))
        title = "_".join(features[1:])
        plt.title(title)
        plt.xticks(rotation=70);
        plt.tight_layout();
        plt.legend(loc='best')
        fig.canvas.set_window_title(title)
        plt.show()


def plot_data_correlation(data_to_plot, features):
    '''
      Plot the correlation dataframe
    '''
    x = data_to_plot.index

    for i_feature in features:
        plt.plot(x, (data_to_plot[i_feature] - min(data_to_plot[i_feature])) / (
                max(data_to_plot[i_feature]) - min(data_to_plot[i_feature])), label=str(i_feature))

    title = " ".join(features[1:])
    plt.title("correlation of : " + title)
    plt.xticks(rotation=70);
    plt.tight_layout();
    plt.legend(loc='best')
    plt.show()


def test_stationarity(timeseries):
    '''
     test stationarity of the time serie with the Dickey-Fuller Test and plotting the Rolling Meand and standard deviation
    '''
    # Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=24)
    rolstd = pd.rolling_std(timeseries, window=24)

    # Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('PowerPV/Date with Rolling Mean & Standard Deviation')
    plt.xticks(rotation=70)
    plt.ylabel('PV_1 - P');
    plt.xlabel("Time step : 1 hour");
    plt.tight_layout()
    plt.show(block=False)

    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)


# for more information test Dickey-Fuller, qq plot, acf, pacf
def ts_diagnostics(y, lags=None, title='', filename=''):
    '''
    Calculate acf, pacf, qq plot and Augmented Dickey Fuller test for a given time series
    '''
    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    # weekly moving averages (5 day window because of workdays)
    rolling_mean = pd.rolling_mean(y, window=12)
    rolling_std = pd.rolling_std(y, window=12)

    fig = plt.figure(figsize=(14, 12))
    layout = (3, 2)
    ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
    acf_ax = plt.subplot2grid(layout, (1, 0))
    pacf_ax = plt.subplot2grid(layout, (1, 1))
    qq_ax = plt.subplot2grid(layout, (2, 0))
    hist_ax = plt.subplot2grid(layout, (2, 1))

    # time series plot
    y.plot(ax=ts_ax)
    rolling_mean.plot(ax=ts_ax, color='crimson');
    rolling_std.plot(ax=ts_ax, color='darkslateblue');
    plt.legend(loc='best')
    ts_ax.set_title(title, fontsize=24);

    # acf and pacf
    smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)
    smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5)

    # qq plot
    sm.qqplot(y, line='s', ax=qq_ax)
    qq_ax.set_title('QQ Plot')

    # hist plot
    y.plot(ax=hist_ax, kind='hist', bins=25);
    hist_ax.set_title('Histogram');
    plt.tight_layout();
    # plt.savefig('./img/{}.png'.format(filename))
    plt.show()

    # perform Augmented Dickey Fuller test
    print('Results of Dickey-Fuller test:')
    dftest = adfuller(y, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['test statistic', 'p-value', '# of lags', '# of observations'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)
    return


# ACF and PACF plots:
def plot_acf_pacf(data_to_plot):
    from statsmodels.tsa.stattools import acf, pacf
    lag_acf = acf(data_to_plot, nlags=20)
    lag_pacf = pacf(data_to_plot, nlags=20, method='ols')
    # Plot ACF:
    plt.subplot(121)
    plt.plot(lag_acf)
    plt.axhline(y=0, linestyle='--', color='gray')
    plt.axhline(y=-1.95 / pd.np.sqrt(len(data_to_plot)), linestyle='--', color='gray')
    plt.axhline(y=1.95 / pd.np.sqrt(len(data_to_plot)), linestyle='--', color='gray')
    plt.title('Autocorrelation Function')

    # Plot PACF:
    plt.subplot(122)
    plt.plot(lag_pacf)
    plt.axhline(y=0, linestyle='--', color='gray')
    plt.axhline(y=-1.95 / pd.np.sqrt(len(data_to_plot)), linestyle='--', color='gray')
    plt.axhline(y=1.95 / pd.np.sqrt(len(data_to_plot)), linestyle='--', color='gray')
    plt.title('Partial Autocorrelation Function')
    plt.tight_layout()
