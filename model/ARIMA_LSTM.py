#!/usr/bin/env python
# coding: utf-8
!pip install statsmodels
!pip install scipy==1.3.0
!pip install numpy==1.21.0

!pip install numpy==1.19.5

import itertools

import statsmodels as sm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import concatenate
from pandas import concat, DataFrame

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler

from tensorflow_core.python.keras import Sequential
from tensorflow_core.python.keras.layers import LSTM, Dropout, Dense


# arima-lstm---arima--lstm--->predict
# LOAD DATA
data_raw = pd.read_csv('https://raw.githubusercontent.com/herecomesmax/herecomesmax/data/IXIC_w_EV.csv')
df = pd.read_csv('https://raw.githubusercontent.com/herecomesmax/herecomesmax/data/IXIC_w_EV.csv')
# GET FEATURES
features=['Close']
data_raw=data_raw[features]

CPI = df['CPI']
print(CPI)

#FINDING THE BEST p, q, d FOR ARIMA
#OR TEST THE HYPER PARAMETERS USING PMDARImA

q_arima = range(0, 6)
d_arima = 0
p_arima = range(0, 10)
AIC_arima = []
ARIMAX_model = []
pdqs = [(x[0], d_arima, x[1]) for x in list(itertools.product(p_arima, q_arima))]

for pdq in pdqs:
    try:
        mod = ARIMA(data_raw, order=pdq)
        results = mod.fit()
        print('ARIMAX{} - AIC:{}'.format(pdq, results.aic))
        AIC_arima.append(results.aic)
        ARIMAX_model.append(pdq)
    except:
        continue

#GET pqd

#TRAIN ARIMA
index=AIC_arima.index(min(AIC_arima))
order = ARIMAX_model[index]
print('order num',order)

#data_raw= data_raw.drop([0])


#TRAIN ARIMA
order=(0,4,5)
#mod = sm.tsa.arima.model.ARIMA(data_raw, order=(1, 0, 0))
mod = ARIMA(data_raw,CPI, order=(6,0,4))


model = ARIMA(data_raw, order=(0,4,5))
fit = model.fit()

#GET ARIMA PREDICTION
forr = fit.forecast
print(forr)


preds = fit.predict(1,len(data_raw)+10, typ='levels')
preds_pd = preds.to_frame()
preds_pd.index -= 1




# USING THE ARIMA OUTPUT TO CONSTRUCT A NEW DATASET
arima_result = pd.DataFrame(columns=['Close'])
arima_result['Close'] = data_raw['Close']
arima_result['predicted'] = preds
arima_result['residuals'] = arima_result['Close'] - arima_result['predicted']



# USE THE OUTPUT OF ARIMA AS THE INPUT OF LSTM LAYER
new_data = arima_result
lstm_data = new_data['residuals'][:].values.astype(float)


# DATA SPLIT
def split_data(value, timestep, test_percentage):
    data_in = value
    data = []

    # create all possible sequences of length seq_len
    for index in range(len(data_in) - timestep):
        data.append(data_in[index: index + timestep])

    data = np.array(data);
    test_set_size = int(np.round(test_percentage * data.shape[0]))
    train_set_size = data.shape[0] - test_set_size

    x_train = data[1:train_set_size, :-1, :]
    y_train = data[1:train_set_size, -1, :]

    x_test = data[train_set_size:, :-1]
    y_test = data[train_set_size:, -1, :]

    return [x_train, y_train, x_test, y_test, train_set_size]

def series_to_supervised(data, n_in, n_out, dropnan=True):

    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1),t-2,t-1-->t
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# GENERATE THE 3D LSTM DATASET (SAMPLE, TIME STEP, FEATURES)
def dataprepare(values,timestep):
    reframed = series_to_supervised(values,timestep, 1)#X,y

    values = reframed.values
    # TRAINING/TESTING SET SPLIT
    train = values[1:train_len, :]
    test = values[train_len:, :]
    # GET THE CORRESPONDNIG X AND Y LABEL
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]

    # RESHAPE DATA
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

    print("train_X.shape:%s train_y.shape:%s test_X.shape:%s test_y.shape:%s" % (
    train_X.shape, train_y.shape, test_X.shape, test_y.shape))
    return train_X,train_y,test_X,test_y




# STANDARDIZATION
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler_data = scaler.fit_transform(lstm_data.reshape(-1, 1))
# DROP NAN
scaler_data=scaler_data[~np.isnan(scaler_data).any(axis=1), :]
# GET THE TRAINING DATA FOR LSTM

# TRAINING/TESTING SET SPLIT
train_len = int(len(data_raw) * 0.7)
test_len=len(data_raw)-train_len
print(train_len)
#
timestep = 24  #SLIDING WINDOW
x_train, y_train, x_test, y_test = dataprepare(scaler_data,timestep)
# PRINT DATA
print('x_train.shape = ', x_train.shape)
print('y_train.shape = ', y_train.shape)
print('x_test.shape = ', x_test.shape)
print('y_test.shape = ', y_test.shape)
# LSTM
model = Sequential()

model.add(LSTM(units=128, input_shape=(x_train.shape[1], x_train.shape[2]),activation='tanh',return_sequences=True))
model.add(LSTM(units=128,activation='tanh',return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
# TRAIN MODEL
history = model.fit(x_train, y_train, epochs=50, batch_size=32, callbacks=None, validation_split=None,validation_data=None, shuffle=False, verbose=2)


# GET PREDICTIONS
model.save('xx.h5')
y_test_pred = model(x_test)

# RESTORATION X TEST

test_X = x_test.reshape((x_test.shape[0], x_test.shape[2]))
y_test_pred = concatenate((y_test_pred, test_X[:, 1:]), axis=1)
y_test_pred = scaler.inverse_transform(y_test_pred)
y_test_pred = y_test_pred[:, 0]

# RESTORATION Y TEST
y_testy = y_test.reshape((len(y_test), 1))
y_testy = concatenate((y_testy, test_X[:, 1:]), axis=1)
y_testy = scaler.inverse_transform(y_testy)
y_testy = y_testy[:, 0]

# METRICS
test_mse = mean_squared_error(y_testy, y_test_pred)
test_mae = mean_absolute_error(y_testy, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_rsq = r2_score(y_testy, y_test_pred)



print(f'X_test MSE: {test_mse}')
print(f'X_test MAE: {test_mae}')
print(f'X_test RMSE: {test_rmse}')
print(f'X_test R-sq: {test_rsq}')


# STORE THE PREDICTION OF ARIMA-LSTM AS draw_test
draw_test = new_data['predicted'][-len(y_test_pred):].values.astype(float)+y_test_pred
draw_test=draw_test.reshape(-1,1)
# PLOT THE RESULT
plt.plot(new_data['Close'], label="Reference", color='tab:blue')
print(len(new_data),len(x_train)+26)
# SHITF DATA TO BETTER CORRESPOND AND COMPARE
plt.plot(range(len(x_train)+26,len(new_data)),draw_test, color='orange',label='test_Prediction')
plt.title('Prediction', size=12)
plt.legend()
plt.savefig('test.jpg')
plt.show()




########################################################

## LSTM
url = 'https://raw.githubusercontent.com/herecomesmax/herecomesmax/data/IXIC_w_EV.csv'
df2 = pd.read_csv(url, usecols=['Close','Date'])
# Dataset is now stored in a Pandas Dataframe

df2 = df2.dropna()
df2.info()
df2.describe()
df2.isnull().sum()

df2 = df2.sort_values('Date')
df2.head()

plt.plot(df2[['Close']])


################################################
################################################
#TESTING SINGLE LSTM SHOWN BELOW

price = df2[['Close']]
price.info()

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1, 1))
price['Close'] = scaler.fit_transform(price['Close'].values.reshape(-1,1))

def split_data(stock, lookback):
    data_raw = stock.to_numpy() # convert to numpy array
    data = []
    
    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - lookback): 
        data.append(data_raw[index: index + lookback])
    
    data = np.array(data);
    test_set_size = int(np.round(0.2*data.shape[0]));
    train_set_size = data.shape[0] - (test_set_size);
    
    x_train = data[:train_set_size,:-1,:]
    y_train = data[:train_set_size,-1,:]
    
    x_test = data[train_set_size:,:-1]
    y_test = data[train_set_size:,-1,:]
    
    return [x_train, y_train, x_test, y_test]

lookback = 1 # choose sequence length
x_train, y_train, x_test, y_test = split_data(price, lookback)
print('x_train.shape = ',x_train.shape)
print('y_train.shape = ',y_train.shape)
print('x_test.shape = ',x_test.shape)
print('y_test.shape = ',y_test.shape)

import torch
import torch.nn as nn

x_train = torch.from_numpy(x_train).type(torch.Tensor)
x_test = torch.from_numpy(x_test).type(torch.Tensor)
y_train_lstm = torch.from_numpy(y_train).type(torch.Tensor)
y_test_lstm = torch.from_numpy(y_test).type(torch.Tensor)
y_train_gru = torch.from_numpy(y_train).type(torch.Tensor)
y_test_gru = torch.from_numpy(y_test).type(torch.Tensor)

input_dim = 1
hidden_dim = 128
num_layers = 2
output_dim = 1
num_epochs = 50
