from pandas.io.json import json_normalize
import requests
import pandas as pd
import datetime
import time
from pathlib import Path
import calendar
import json
from requests.adapters import HTTPAdapter
from urllib3 import Retry
import pytz  ## pip install pytz
import datetime

proxies = {
    "http": "http://10.66.243.130:8080/"
}

base_url = "http://api.openweathermap.org/data/2.5/"


madatz = pytz.timezone('Indian/Antananarivo')


def get_data_hourly_owp():
    madatz = pytz.timezone('Indian/Antananarivo')  ## Set your timezone
    madatz_now = datetime.datetime.now(madatz)
    data_params = {
        "lat":-18.95443,
        "lon":49.1094,
        "APPID": "52deccdaba3d22888ae45aa0bb0d82fd"
    }
    response = requests.get(base_url+"weather",params=data_params,proxies=proxies)
    reader_json = response.json()
    df_json = json_normalize(reader_json)
    df_json.set_index(pd.to_datetime(df_json["dt"], unit='s',utc=True), inplace=True)
    df_json.index = df_json.index.tz_convert('Indian/Antananarivo').tz_localize(None)
    df_json.drop_duplicates(['dt'], keep='first', inplace=True)
    df_json.to_csv("OWP/data_" + str(madatz_now.__str__().split('.')[0].replace(":","_"))+ ".csv")

from apscheduler.schedulers.blocking import BlockingScheduler
scheduler = BlockingScheduler()
scheduler.add_job(get_data_hourly_owp, 'interval', hours=1)
scheduler.start()

def get_data_forecast_owp():
    madatz = pytz.timezone('Indian/Antananarivo')  ## Set your timezone
    madatz_now = datetime.datetime.now(madatz)
    data_params = {
        "lat":-18.95443,
        "lon":49.1094,
        "APPID": "52deccdaba3d22888ae45aa0bb0d82fd"
    }
    response = requests.get(base_url+"forecast",params=data_params)
    reader_json = response.json()
    df_json = json_normalize(reader_json['list'])
    df_json.to_csv("OWP_forecast/data_" +str(madatz_now.__str__().split('.')[0].replace(":","_"))+".csv")


def timestamp_date(timestamp,madatz=madatz):
    return datetime.datetime.fromtimestamp(timestamp,tz=madatz).strftime('%Y-%m-%d:%H')

if __name__ == "__main__":
    madatz = pytz.timezone('Indian/Antananarivo')  ## Set your timezone
    madatz_now = datetime.datetime.now(madatz)
    # timestamp_start = 1521936000 #'2018-03-17:01' a faire en deux temps pour l'heure d'ete ajouté en fin de mars
    timestamp_start = 1485982800 #'2017-02-02:00' timezone de madagascar
    timestamp_final = calendar.timegm(madatz_now.timetuple())
    nb_day = int((timestamp_final - timestamp_start) / 86400)
    iter_timestamp = timestamp_start
    for i in range(0,nb_day):
        my_file = Path('OWP_forecast/data_'+timestamp_date(iter_timestamp,madatz).split(":")[0]+"_"+str(iter_timestamp)+'.csv')
        if not my_file.exists():
            get_data_forecast_owp(iter_timestamp)
            time.sleep(5)
            print("Done for :"+timestamp_date(iter_timestamp,madatz).split(":")[0])
        else:
            print("Already exist :" + timestamp_date(iter_timestamp,madatz).split(":")[0])
        iter_timestamp = iter_timestamp + 86400
    print("Done Data weather up-to-date")