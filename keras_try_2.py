import numpy as np
import matplotlib.pyplot as plt
# fix random seed for reproducibility
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

np.random.seed(7)

# Import
import pandas as pd
import preprocessing as pr

data_site_hourly = pd.read_csv(r"data_site/data_1485990000000.csv", low_memory=False)

features_hourly = ["Date", pr.PowerPV,pr.Temperature,pr.Irradiation]
data_prepared_hourly = pr.prepare_date(data_site_hourly,features_hourly)
data_cleaned_hourly = pr.clean_data(data_prepared_hourly)

# data = data.reset_index()

# Drop date variable

data = data_model.copy()

data = data.reset_index()
data = data.drop(['Date'], 1)

# Dimensions of dataset
n = data.shape[0]
p = data.shape[1]

# Make data a np.array
data = data.values

# Training and test data
train_start = 0
train_end = int(np.floor(0.8*n))
test_start = train_end + 1
test_end = n
data_train = data[np.arange(train_start, train_end), :]
data_test = data[np.arange(test_start, test_end), :]

# Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(data_train)
data_train = scaler.transform(data_train)
data_test = scaler.transform(data_test)


# convert an array of values into a dataset matrix
def create_dataset(dataset):
	return data_test[:,1:],data_test[:,0]


trainX, trainY = create_dataset(data_train)
testX, testY = create_dataset(data_test)

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, 6)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=10, batch_size=4, verbose=2)

# # serialize model to JSON
# model_json = model.to_json()
# with open("model_look_back_3.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("model_look_back_3.h5")
# print("Saved model to disk")


import math
# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY, testPredict))
print('Test Score: %.2f RMSE' % (testScore))

# shift train predictions for plotting
trainPredictPlot = np.empty_like(data)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[len(trainPredict), :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(data)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict), :] = testPredict
# plot baseline and predictions
plt.plot(data)
plt.plot(trainPredict[-50:], label="predicted")
plt.plot(trainY[-50:],label="observed")
plt.show()
plt.legend(loc='best')
plt.tight_layout();

tit = trainX[1:2]
model.predict(tit)







