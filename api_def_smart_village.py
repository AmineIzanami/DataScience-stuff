import calendar
import datetime
import time
from pathlib import Path

import pytz
import requests
import pandas as pd
import io
import datetime

server = "41.188.38.121"
port = ":8000"

proxies = {
    "http": "http://10.66.243.130:8080/"
}
base_url = "http://" + server + port + "/esm"
key_acces = {"USER":"adminsys", "password":"secret"}
login_param = {"Request":"LOGIN","USER":key_acces['USER'],"PASSWORD":key_acces['password']}

madatz = pytz.timezone('Indian/Antananarivo')

def date_to_millis(dt):
    # datetime.datetime(2018, 4, 5, 8, 57, 14, 345000)
    epoch = datetime.datetime.utcfromtimestamp(0)
    return (dt - epoch).total_seconds() * 1000.0

def millis_to_date(ml):
    return datetime.datetime.fromtimestamp(ml/1000.0)

def list_sites(session):
    response = session.get(base_url + "/services/crud/Site?rootNodeId=1", proxies=proxies)
    reader = response.content
    all_data = pd.read_csv(io.StringIO(reader.decode('utf-8')), sep=";")
    return all_data

def get_data_site(session,site_id,granularity,start_date,end_date):
    data_params = {
        #"Request":"GetData",
        "site":site_id,
        "start":start_date,
        "end":end_date,
        "granularity":granularity,
        "type":"SITE",
        "domain": "Overview.All",
        "Listener": "Charts",
        "sensorsData": "true"
    }
    response = session.get(base_url + "/services/rrd/GetData",params=data_params,proxies=proxies)
    reader = response.content
    all_data = pd.read_csv(io.StringIO(reader.decode('utf-8')), sep=";")
    all_data.to_csv("data_site_madagascar/data_" + timestamp_date(start_date/1000, madatz).split(":")[0] + "_" + str(start_date) + ".csv")

def timestamp_date(timestamp,madatz):
    return datetime.datetime.fromtimestamp(timestamp,tz=madatz).strftime('%Y-%m-%d:%H')

if __name__ == "__main__":
    madatz = pytz.timezone('Indian/Antananarivo')  ## Set your timezone
    madatz_now = datetime.datetime.now(madatz)
    # timestamp_start = 1521936000 #'2018-03-17:01' a faire en deux temps pour l'heure d'ete ajouté en fin de mars
    timestamp_start = 1485982800*1000  # '2017-02-02:00' timezone de madagascar
    timestamp_final = calendar.timegm(madatz_now.timetuple())*1000
    nb_day = int((timestamp_final - timestamp_start) / 86400000)
    iter_timestamp = timestamp_start
    for i in range(0, nb_day):
        with requests.Session() as s:
            my_file = Path(
                'data_site_madagascar/data_' + timestamp_date(iter_timestamp/1000, madatz).split(":")[0] + "_" + str(iter_timestamp) + '.csv')
            if not my_file.exists():
                login_url = base_url + "/HTTPALARM"
                s.get(login_url, params=login_param, proxies=proxies)
                get_data_site(s, "TATS060R", "HOUR", iter_timestamp, iter_timestamp + 86400000)
                # time.sleep(2)
                print("Done for :" + timestamp_date(iter_timestamp/1000, madatz).split(":")[0])
            else:
                print("Already exist :" + timestamp_date(iter_timestamp/1000, madatz).split(":")[0])
            iter_timestamp = iter_timestamp + 86400000 #millisecond
    print("Done Data site up-to-date")


#TATS060R - > Atsinananan
# all_sites[all_sites['Nom'] == "TATS060R"]['C01_AdminRegion'].values[0]
