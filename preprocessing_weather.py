import datetime
import ephem
import json
from pandas.io.json import json_normalize
import pandas as pd
import glob
import pandas as pd
import matplotlib.pyplot as plt

latitude = -18.9588056
longitude = 49.1077778

def get_sun_alt(date):
    obs = ephem.Observer()
    obs.date = ephem.Date(date)  # ajoute le nombre d'heures à la date
    obs.elevation = 10
    obs.lat = str(latitude)
    obs.long = str(longitude)
    sun = ephem.Sun(obs)
    return sun.alt

def get_sun_az(date):
    obs = ephem.Observer()
    obs.date = ephem.Date(date)  # ajoute le nombre d'heures à la date
    obs.elevation = 10
    obs.lat = str(latitude)
    obs.long = str(longitude)
    sun = ephem.Sun(obs)
    return sun.az

def get_all_weather_weatherbit():
    path =r'data_weather_weatherbit/' # use your path
    allFiles = glob.glob(path + "/*.csv")
    frame = pd.DataFrame()
    list_ = []
    for file_ in allFiles:
        df = pd.read_csv(file_,sep=";",index_col=0, header=0)
        list_.append(df)
    frame = pd.concat(list_)
    frame.drop_duplicates(['datetime'], keep='first',inplace=True)
    frame.set_index(pd.to_datetime(frame["ts"], unit='s', utc=True), inplace=True)
    frame.index = frame.index.tz_convert('Indian/Antananarivo').tz_localize(None)
    frame = frame.drop(['datetime', 'ts'], axis=1)
    return frame

def get_all_weather_darksky():
    path = r'Darksky/'  # use your path
    allFiles = glob.glob(path + "/*.csv")
    frame = pd.DataFrame()
    list_ = []
    for file_ in allFiles:
        df = pd.read_csv(file_, sep=",", index_col=0, header=0)
        list_.append(df)
    frame = pd.concat(list_)
    frame.drop_duplicates(['time.1'], keep='first', inplace=True)
    frame.set_index(pd.to_datetime(frame["time.1"], unit='s',utc=True), inplace=True)
    frame.index = frame.index.tz_convert('Indian/Antananarivo').tz_localize(None)
    frame = frame.drop(['time.1'], axis=1)
    return frame


# alpha = angle d’élévation du soleil
# 𝜃 = angle d’azimut (0)
# β = inclination du module(20)
# Ψ = Un module dans l'hémisphère sud sera face au nord avec Ψ = 0°, le poc dans hémi sud

#NOTC_Power#1 220
# NOCT_temp#1 45
#loss_by_degree#1 0.402
# module_tilt#1 20
# MPPT_yield#1  95


#
#
# import math
#
# alpha = 0 #elev_angle
# azimut = 0
# tilt = 20
# orientation = 0
# nb_panels = 34
# puissance_noct = 220
# temperature_noct = 45
# loss_by_degree = 0.402
#
# puissance  = pd.DataFrame(index=data_weather['2018-03-17 01:00:00':'2018-03-24 01:00:00'].index)
#
# #irradiance of 34 module
#
# import ephem
#

#
#
# puissance_jonathan  = pd.DataFrame(index=data_weather['2018-03-17 01:00:00':'2018-03-24 01:00:00'].index)
#
# puissance_jonathan["module"] = pd.Series(puissance_jonathan.index).apply(lambda x : max(1000*((math.cos(get_sun_alt(x))*(math.sin(math.radians(tilt)))*math.cos(orientation - get_sun_az(x)))+(math.sin(get_sun_alt(x)))*(math.cos(math.radians(tilt)))),0)).values
# puissance_jonathan["Intensité_module_34"] = puissance_jonathan["module"] * nb_panels
# puissance_jonathan["PV_irradiance"] =(puissance_jonathan["module"]/800) * puissance_noct
# puissance_jonathan["temperature_cellule"] = data_cleaned['2018-03-17 01:00:00':'2018-03-24 01:00:00']["PV_1 - Temp"] + (puissance_jonathan["module"]/800)*(temperature_noct - 20)
# puissance_jonathan["Puissance lié au coef temp"] = puissance_jonathan["PV_irradiance"] * (loss_by_degree/100) * puissance_jonathan["temperature_cellule"] - temperature_noct
# puissance_jonathan["Puissance total théorique total"] = (nb_panels * (puissance_jonathan["PV_irradiance"] - puissance_jonathan["Puissance lié au coef temp"]))/1000
# puissance_jonathan["Puissance total théorique par panneau"] =  (puissance_jonathan["PV_irradiance"] - puissance_jonathan["Puissance lié au coef temp"])
#
# puissance_api  = pd.DataFrame(index=data_prepared['2018-03'].index)
# puissance_api["irradiance_module"] = data_prepared['2018-03']['Solar_Rad - W/m2']
# puissance_api["PV_irradiance"] =(puissance_api["irradiance_module"]/800) * puissance_noct
# puissance_api["temperature_cellule"] = data_prepared['2018-03']["PV_1 - Temp"] + (puissance_api["irradiance_module"]/800)*(temperature_noct - 20)
# puissance_api["Puissance lié au coef temp"] = puissance_api["PV_irradiance"] * (loss_by_degree/100) * puissance_api["temperature_cellule"] - temperature_noct
# puissance_api["Puissance total théorique total"] = (nb_panels * (puissance_api["PV_irradiance"] - puissance_api["Puissance lié au coef temp"]))/1000
# puissance_api["Puissance total théorique par panneau"] =  (puissance_api["PV_irradiance"] - puissance_api["Puissance lié au coef temp"])
#
#
# puissance_jonathan[puissance_jonathan["Puissance total théorique total"] == 1.53] = 0
# puissance_api[puissance_api["Puissance total théorique total"] == 1.53] =0
#
#
# x = data_prepared['2018-03']["PV_1 - P"].index
# y1 = data_prepared['2018-03']["PV_1 - P"].values
# # y2 = puissance_jonathan['2018-03-17 01:00:00':'2018-03-24 01:00:00']["Puissance total théorique total"].values
# y3 = puissance_api['2018-03']["Puissance total théorique total"].values
# # y4 = data_weather['2018-03-17 01:00:00':'2018-03-24 01:00:00']['ghi'].values
# # y5 = puissance_jonathan["module"].values
#
# plt.figure();
# plt.xticks(rotation=70);
# plt.plot(x, y1, label='PV_1 - P',color='blue')
# # plt.plot(x, y2, label='Puissance  théorique total avec calcul de paramétre capteur',color='red')
# plt.plot(x, y3, label='Puissance  théorique total avec donnée de capteur sur place ',color='green')
# # plt.plot(x, y4, label='Global horizontal solar irradiance',color='pink')
# # plt.plot(x, y5, label='Jonathan irradiance',color='blue')
#
# plt.legend(loc='best')
# plt.tight_layout();
# plt.title("Comparaison entre les 3 différents modéles")
# plt.show()

