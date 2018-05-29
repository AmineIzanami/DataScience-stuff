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

base_url = "https://api.darksky.net/forecast/629758f8a31408bc57c787adb007f9de/-18.95443,49.1094"

madatz = pytz.timezone('Indian/Antananarivo')

def get_data_forecast_darksy():
    madatz = pytz.timezone('Indian/Antananarivo')  ## Set your timezone
    madatz_now = datetime.datetime.now(madatz)
    data_params = {
        "units":"si", # take apropriate units celsius
    }
    response = requests.get(base_url,params=data_params,proxies=proxies)
    reader_json = response.json()
    df_json = json_normalize(reader_json['hourly']['data'])
    df_json.drop(['icon'], 1, inplace=True)
    df_json.set_index(pd.to_datetime(df_json["time"], unit='s',utc=True), inplace=True)
    df_json.index = df_json.index.tz_convert('Indian/Antananarivo').tz_localize(None)
    df_json.drop_duplicates(['time'], keep='first', inplace=True)
    df_json.to_csv("C:/Users/G603344/PycharmProjects/POCv1.0/Darksky_forecast/data_" +str(madatz_now.__str__().split('.')[0].replace(":","_"))+".csv")
    print("done for "+str(madatz_now.__str__().split('.')[0].replace(":","_")) )

if __name__ == "__main__":
    print("Start Scrapping the API forcast Darksky data each 48 hours since: "+str(datetime.datetime.now(madatz)))
    from apscheduler.schedulers.blocking import BlockingScheduler
    scheduler = BlockingScheduler()
    scheduler.add_job(get_data_forecast_darksy, 'interval', hours=48)
    scheduler.start()