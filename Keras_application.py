import fbprophet as ph
import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA
import preprocessing as pr
import preprocessing_weather as prw
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy
import os
from sklearn.externals import joblib
pd.options.mode.chained_assignment = None  # to avoid the false postive warning of SettingWithCopyWarning:


# later...

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")
print("Loaded model from disk")

scaler = joblib.load('scaler_keras_model.save')
print("Loaded scaler from disk")

#Daily Data
data_site_check = pd.read_csv(r"data_site/data_1522997834000.csv", low_memory=False)
features_check = ["Date", pr.PowerPV]
data_prepared_check = pr.prepare_date(data_site_check,features_check)
data_cleaned_check = pr.clean_data(data_prepared_check)



data = data_cleaned_check.reset_index()

# Drop date variable
data = data.drop(['Date'], 1)

# Make data a np.array
data = data.values

data = scaler.transform(data)
dataX = []
value_start = data[0:1]
obs = value_start
for i in range(0,12):
    obs_reshaped = pd.np.reshape(obs, (obs.shape[0], 1, obs.shape[1]))
    predict_scaled = model.predict(obs_reshaped)
    predict = scaler.inverse_transform(predict_scaled)
    dataX.append(predict[0])
    print(obs)
    obs = predict

dataX = pd.np.array(dataX)