#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 01:20:23 2022

@author: archquin
"""

# CODE https://www.datacamp.com/tutorial/tutorial-for-recurrent-neural-network, only plot and return mse function. Idea also .



def plot_predictions(test, predicted):
    plt.plot(test, '*',color="gray", label="Real")
    plt.plot(predicted, color="red", label="Predicted")
    plt.title("Price Prediction")
    plt.legend()
    plt.show()

def return_rmse(test, predicted):
    rmse = np.sqrt(mean_squared_error(test, predicted))
    print("The root mean squared error is {:.2f}.".format(rmse))


# CODE https://www.datacamp.com/tutorial/tutorial-for-recurrent-neural-network, only plot and return mse function. Idea also .


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from tensorflow.keras.optimizers import SGD
from keras.models import Sequential
#from keras.layers import SimpleRNN
#Data Source
import yfinance as yf

# hours forecast
h = 6

# dated trend

pas = 4
# days interval , that trend was premeditated was 4 day moving average
mps = 1 
# minutes interval for 4dmavg

# to be fit into near feauture interrvals
# paradoxically the smaller the X-Y fit data for trend to predicted fit that is compared to test set, the more elaborate the predction 
# has been. I.e. to fit long term range prospect of a market seems trivial noise for the LSTM, but to fit weekly trend to daily test set
# is an actual thing. Wow, tomorrow is just enough, two days, and good staff lays. Three, and you won t ever be free :P. Four, and your
# prediction is floor. Five, just make it live, and six, what a fix; it would be but its aint....

prd = 2 # days to fit
mrd = 1 # minute intervals for y

data = yf.download(tickers = 'EUR=X' ,period =str(pas)+'d', interval = str(mps)+'m')
data2 = yf.download(tickers = 'EUR=X' ,period =str(prd)+'d', interval = str(mrd)+ 'm')

dataX = np.array(data)
dataX2 = np.array(data2)
X= dataX[:,1] # X is high
X2 = dataX2[-60*h:,1] # short term data fit was compilation of next 6 hours
# so all data boils up to 6 last hours to mimic trend fancynes of the 4 day moving avg in order to predict based on the test set,
#i.e. current data of last 2 days next 2000 minutes. No idea why it seems to be working.


# Array reshapes and manipulation
X = np.array(X).reshape(-1,1)
X2 = np.array(X2).reshape(-1,1)

# Horizontal recurence of GRU
step = 128

training_set, test_set = train_test_split(X,test_size=1/3,shuffle=False) # I used test set equal training set to predict next 6 hours

# The idea is to match an entire weeks data trend X to the corresponding Y i.e. X2 that is lef out as a test sample.
# the model test then is just as the last day fit for tomorrrow which had been deemed to fit as 4 days of to fit to a test/last day set.
# Hence test, gives tomorrow's rest, but not for sure....



def vanilly(pvc,setp):
    # the idea of villy is to make horizontal fractioned vertically split arrays that can be used reccurently from this NN type.
    povc = []
    pov = []
    for i,m in enumerate(pvc):
        poc = []
        cbc = []
        if i+setp < len(pvc):
            for j in range(setp):
                poc.append(pvc[i+j])
                if j == setp-1:
                    cbc.append(pvc[i+j])
                    povc.append(poc)
                    pov.append(cbc)
        
    return np.array(povc), np.array(pov) # returns X as X _train, and Y as _, cause _ wont be used




sc = MinMaxScaler(feature_range=(0, 1))
training_set = training_set.reshape(-1, 1)
# transforms test to Nx1 matrix 
TsC = sc.fit_transform(training_set)

# forming training tensor for GRU NN
X_train,_ = vanilly(TsC,step)
# transforms train to NxS(step)x1 matrix for LSTM regression
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],1)

# forming tensors of eq.size
if len(X2)!=X_train.shape[0]:
    if len(X2)>X_train.shape[0]:
        X2=X2[:X_train.shape[0]]
    else:
        X_train = X_train[:len(X2)]

X2 = X2.reshape(X2.shape[0],1)




""" XEROX_INFINITY """
############################################################################### #                                                                     
XeroX_infinity = Sequential()                                                   #    
XeroX_infinity.add(GRU(units = 256,return_sequences=False,input_shape=(step,1)))#
#XeroX_infinity.add(LSTM(500,return_sequences=True))
#XeroX_infinity.add(LSTM(500))
XeroX_infinity.add(Dense(1))                                                    #            
XeroX_infinity.compile(optimizer="RMSprop", loss="mse")                         #    
XeroX_infinity.summary()                                                        #    
XeroX_infinity.fit(X_train, X2, epochs=50, batch_size=30)                       #    
############################################################################### #
""" #Essenece of evil; Auth: J.Beaudrillard #post modernism #prediction dream """


dst = X[len(training_set):]
#inputs = dst[len(dst) - len(test_set) - step :]
inputs = dst.reshape(-1, 1)

inputs = sc.transform(inputs)

X_test, _ = vanilly(inputs, step)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)


x3Rx = XeroX_infinity.predict(X_test)
# makes prediction
x3Rx = sc.inverse_transform(x3Rx)
# plots test set predicted and pattern on test + prediction
plot_predictions(test_set, np.concatenate((test_set,x3Rx)))
