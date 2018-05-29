from pandas.io.json import json_normalize
import requests
import pandas as pd
import datetime
import time
from pathlib import Path
import calendar
import pytz

proxies = {
    "http": "http://10.66.243.130:8080/"
}

base_url = "http://api.weatherbit.io/v2.0/forecast/hourly"
key_acces = {"key":"e8ed2d8e42994b90837ef3f3f22db4ef"}

madatz = pytz.timezone('Indian/Antananarivo')

def get_data_forecast_48h_weatherbit():
    madatz = pytz.timezone('Indian/Antananarivo')  ## Set your timezone
    madatz_now = datetime.datetime.now(madatz)
    data_params = {
        "lat":-18.95443,
        "lon":49.1094,
        "key": "e8ed2d8e42994b90837ef3f3f22db4ef"
    }
    response = requests.get(base_url,params=data_params)#, proxies=proxies)
    reader_json = response.json()
    data_weather = json_normalize(reader_json["data"])
    data_weather.drop(['weather.icon', 'weather.code','snow'], axis=1,inplace=True)
    data_weather['clouds'].replace(pd.np.nan,0,inplace=True)
    data_weather.to_csv('C:/Users/G603344/PycharmProjects/POCv1.0/weatherbit_forecast/'+str(madatz_now.__str__().split('.')[0].replace(":","_"))+".csv")
    print("done for "+str(madatz_now.__str__().split('.')[0].replace(":","_")) )
    # data_weather.to_csv('data_weather/'+str(timestamp)+'.csv',sep=';')


if __name__ == "__main__":
    print("Start Scrapping the API forcast weatherbit data each 48 hours since: "+str(datetime.datetime.now(madatz)))
    from apscheduler.schedulers.blocking import BlockingScheduler
    scheduler = BlockingScheduler()
    scheduler.add_job(get_data_forecast_48h_weatherbit, 'interval', hours=48)
    scheduler.start()