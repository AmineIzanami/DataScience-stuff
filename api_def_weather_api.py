from pandas.io.json import json_normalize
import requests
import pandas as pd
import datetime
import time
from pathlib import Path


proxies = {
    "http": "http://10.66.243.130:8080/"
}

base_url = "http://api.weatherbit.io/v2.0/history/hourly"
key_acces = {"key":"e8ed2d8e42994b90837ef3f3f22db4ef"}

def get_data_hourly(start_date,end_date):
    data_params = {
        "start_date":start_date,
        "end_date":end_date,
        "lat":-18.95443,
        "lon":49.1094,
        "key": "e8ed2d8e42994b90837ef3f3f22db4ef"
    }
    response = requests.get(base_url,params=data_params, proxies=proxies)
    reader_json = response.json()
    data_weather = json_normalize(reader_json["data"])
    data_weather.drop(['weather.icon', 'weather.code','precip6h','snow'], axis=1,inplace=True)
    data_weather['clouds'].replace(pd.np.nan,0,inplace=True)
    data_weather.to_csv('data_weather/'+start_date.split(":")[0]+'.csv',sep=';')
    # data_weather.to_csv('data_weather/'+str(timestamp)+'.csv',sep=';')

def timestamp_date(timestamp):
    return datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d:%H')

def add_day(timestamp_start):
    return timestamp_start + 86400

# get_data_hourly(timestamp_date(1521936000),timestamp_date(1521936000+86400-3600)) # Faire manuelle le jour d'ajout d'heure d'été

if __name__ == "__main__":
    # timestamp_start = 1521936000 #'2018-03-17:01' a faire en deux temps pour l'heure d'ete ajouté en fin de mars
    timestamp_start = 1522018800 #'2018-03-26:01'
    timestamp_final = 1523833200 #'2018-04-16:01'
    nb_day = int((timestamp_final - timestamp_start) / 86400)
    iter_timestamp = timestamp_start
    for i in range(0,nb_day):
        my_file = Path('data_weather/' + timestamp_date(iter_timestamp).split(":")[0]  + '.csv')
        if not my_file.exists():
            get_data_hourly(timestamp_date(iter_timestamp), timestamp_date(iter_timestamp + 86400))
            time.sleep(5)
            print("Done for :"+timestamp_date(iter_timestamp))
        else:
            print("Already exist :" + timestamp_date(iter_timestamp))
        iter_timestamp = iter_timestamp + 86400
    print("Done Data weather up-to-date")