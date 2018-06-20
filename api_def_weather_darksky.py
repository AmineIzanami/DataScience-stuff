import calendar
import datetime
import time
from pathlib import Path
import pandas as pd
import pytz
import requests
from pandas.io.json import json_normalize

proxies = {
    "http": "http://10.66.243.130:8080/"
}
base_url = "https://api.darksky.net/forecast/629758f8a31408bc57c787adb007f9de/-18.95443,49.1094"
madatz = pytz.timezone('Indian/Antananarivo')



def get_data_hourly_darksy(start_date):
    data_params = {
        "units":"si", # take apropriate units celsius
    }
    response = requests.get(base_url+','+str(start_date),params=data_params)
    reader_json = response.json()
    df_json = json_normalize(reader_json['hourly']['data'])
    df_json.drop(['icon'], 1, inplace=True)
    df_json.set_index(pd.to_datetime(df_json["time"], unit='s',utc=True), inplace=True)
    df_json.index = df_json.index.tz_convert('Indian/Antananarivo').tz_localize(None)
    df_json.drop_duplicates(['time'], keep='first', inplace=True)
    df_json.to_csv("Darksky/data_" + timestamp_date(start_date,madatz).split(":")[0]+"_"+str(start_date)+ ".csv")

def get_data_forecast_darksy():

    madatz = pytz.timezone('Indian/Antananarivo')  ## Set your timezone
    madatz_now = datetime.datetime.now(madatz)
    data_params = {
        "units":"si", # take apropriate units celsius
    }
    response = requests.get(base_url,params=data_params)
    reader_json = response.json()
    df_json = json_normalize(reader_json['hourly']['data'])
    df_json.drop(['icon'], 1, inplace=True)
    df_json.set_index(pd.to_datetime(df_json["time"], unit='s',utc=True), inplace=True)
    df_json.drop_duplicates(['time'], keep='first', inplace=True)
    df_json.to_csv("Darksky_forecast/data_" +str(madatz_now.__str__().split('.')[0].replace(":","_"))+".csv")


def timestamp_date(timestamp,madatz=madatz):
    return datetime.datetime.fromtimestamp(timestamp,tz=madatz).strftime('%Y-%m-%d:%H')

if __name__ == "__main__":
    madatz = pytz.timezone('Indian/Antananarivo')  ## Set your timezone
    madatz_now = datetime.datetime.now(madatz)
    # timestamp_start = 1521936000 #'2018-03-17:01' a faire en deux temps pour l'heure d'ete ajouté en fin de mars
    timestamp_start = 1525640400 #'2017-02-02:00' timezone de madagascar
    timestamp_final = calendar.timegm(madatz_now.timetuple())
    nb_day = int((timestamp_final - timestamp_start) / 86400)
    iter_timestamp = timestamp_start
    for i in range(0,nb_day):
        my_file = Path('Darksky/data_'+timestamp_date(iter_timestamp,madatz).split(":")[0]+"_"+str(iter_timestamp)+'.csv')
        if not my_file.exists():
            get_data_hourly_darksy(iter_timestamp)
            time.sleep(5)
            print("Done for :"+timestamp_date(iter_timestamp,madatz).split(":")[0])
        else:
            print("Already exist :" + timestamp_date(iter_timestamp,madatz).split(":")[0])
        iter_timestamp = iter_timestamp + 86400
    print("Done Data weather darksky up-to-date")