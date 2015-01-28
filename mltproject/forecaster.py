#import csv
import numpy as np
import pandas as pd
import math
import QSTK.qstkutil.qsdateutil as du
import datetime as dt
import QSTK.qstkutil.DataAccess as da
import QSTK.qstkutil.tsutil as tsu
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
import scipy.fftpack as fftpack

lookback_window=100

def getFrequency(price_values):
    #Piazza Code
    yhat = fftpack.rfft(price_values)
    idx = (yhat[1:]**2).argmax() + 1
    freqs = fftpack.rfftfreq(len(price_values), d = (1.0)/(2*np.pi))
    frequency = freqs[idx]
    return frequency

def train(data):
    inpLen = data.shape[1]
    predict_window=5
    Xtrain = np.zeros([inpLen*(len(data)-(predict_window+lookback_window)),11])
    Ytrain = np.zeros([inpLen*(len(data)-(predict_window+lookback_window)),1])
    Y = np.zeros([inpLen*(len(data)-(predict_window+lookback_window)),1])
    
    index = 0
    for i in range(0,inpLen):
        for j in range (0, len(data)-lookback_window-predict_window):
            price_values = data[j:j+lookback_window+1, i]
            #Stack Over Flow Code
            n = len(price_values)
            l = np.linspace(0, 4*np.pi, n)

            guess_mean = np.mean(price_values)
            guess_std = 3*np.std(price_values)/(2**0.5)
            guess_phase = 0

            data_first_guess = guess_std*np.sin(l+guess_phase) + guess_mean
            optimize_func = lambda x: x[0]*np.sin(l+x[1]) + x[2] - price_values
            est_std, est_phase, est_offset = leastsq(optimize_func, [guess_std, guess_phase, guess_mean])[0]
            #Stack Over Flow Code
            x1 = getFrequency(price_values)
            x2 = est_phase
            x3 = max(price_values)-min(price_values)
            x4 = price_values[lookback_window]
            x5 = price_values[lookback_window-1]-price_values[lookback_window-2]
            x6 = price_values[lookback_window]**2
            x7 = price_values[lookback_window]**3
            x8 = max(price_values)
            x9 = min(price_values)
            x10= np.mean(price_values)
            x11= price_values[lookback_window]**.5 
            Xtrain[index, 0] = x1
            Xtrain[index, 1] = x2            
            Xtrain[index, 2] = x3 
            Xtrain[index, 3] = x4 
            Xtrain[index, 4] = x5 
            Xtrain[index, 5] = x6 
            Xtrain[index, 6] = x7 
            Xtrain[index, 7] = x8 
            Xtrain[index, 8] = x9 
            Xtrain[index, 9] = x10 
            Xtrain[index, 10] = x11
            Ytrain[index, 0] = (data[j+lookback_window+predict_window,i]-data[j+lookback_window,i])/data[j+lookback_window,i]
            Y[index, 0] = data[j+lookback_window,i]
            index = index + 1

    return Xtrain, Ytrain, Y

def getTestData():
    files = []    
    files.append('ML4T-309')
    #files.append('ML4T-309')
    start = dt.datetime(2006, 1, 1)
    end = dt.datetime(2007, 12, 31)

    timeofday = dt.timedelta(hours=16)
    timestamps = du.getNYSEdays(start, end, timeofday)
    dataobj = da.DataAccess('Yahoo')

    keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']

    data = dataobj.get_data(timestamps, files, keys)
    dataDic = dict(zip(keys, data))

    for key in keys:
        dataDic[key] = dataDic[key].fillna(method='ffill')
        dataDic[key] = dataDic[key].fillna(method='bfill')
        dataDic[key] = dataDic[key].fillna(1.0)
    
    priceList = dataDic['actual_close'].values
    
    Xtest, Y, Ytest = train(priceList)

    return Xtest, Ytest

def getTrainData():
    files = []
    for i in range (0, 200):
        a="ML4T-%03d" % (i)
        files.append(a)
    print files
    start = dt.datetime(2001, 1, 1)
    end = dt.datetime(2005, 12, 31)

    timeofday = dt.timedelta(hours=16)
    timestamps = du.getNYSEdays(start, end, timeofday)
    dataobj = da.DataAccess('Yahoo')

    keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']

    data = dataobj.get_data(timestamps, files, keys)
    print "data from yahoo"
    dataDic = dict(zip(keys, data))

    for key in keys:
        dataDic[key] = dataDic[key].fillna(method='ffill')
        dataDic[key] = dataDic[key].fillna(method='bfill')
        dataDic[key] = dataDic[key].fillna(1.0)
    priceList = dataDic['actual_close'].values
    print "Getting Trainding features"
    Xtrain, Ytrain, Y = train(priceList)
    return Xtrain, Ytrain
    
    
def main():
    Xtrain, Ytrain = getTrainData()
    Xtest, Ytest, = getTestData()
    print "Got Data"
    first200X1 = Xtest[0:200, 0]
    first200X2 = Xtest[0:200, 1]
    first200X3 = Xtest[0:200, 2]
    first200X4 = Xtest[0:200, 3]
    first200X5 = Xtest[0:200, 10]

    regr = linear_model.LinearRegression(normalize=False)
    
    regr.fit(Xtrain,Ytrain)
    print "Model Fitted"
    Y = Ytest[:,0]
    Yzeros = np.zeros([100,1])
    first200Y = Y[0:200]
    last200Y = Y[len(Y)-200: len(Y)]

    daysForFeatureCompare = [x for x in range(0,200)]
    queryRes=regr.predict(Xtest)
           
    resY = queryRes[:,-1]

    for i in range (0, resY.shape[0]):
        resY[i] = (resY[i] + 1) * Y[i]
                
    first200ResY = np.zeros([200])
    first200ResY[0:100] = None
    first200ResY[100:200] = resY[100:200]
    last200ResY = resY[len(resY)-200: len(resY)]

    rmsVal = math.sqrt(np.mean((resY - Y) ** 2))
    corrCoefVal = np.corrcoef(resY,Y)[0,1]
    print rmsVal,corrCoefVal
    xAxisRange = range(1,201)
    
    plotLegend = ['Y actual', 'Y predict']
    plt.clf()
    fig = plt.figure()
    plt.plot(xAxisRange, first200Y, color = 'blue')
    plt.plot(xAxisRange, first200ResY, color = 'red')
    plt.legend(plotLegend,loc=4)
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.savefig("first200Comparison.png", format='png')
    plotLegend = ['Y actual', 'Y predict']
    plt.clf()
    fig = plt.figure()
    plt.plot(xAxisRange, last200Y, color = 'blue')
    plt.plot(xAxisRange, last200ResY, color = 'red')
    plt.legend(plotLegend,loc=4)
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.savefig("last200Comparison.png", format='png')    
    plt.clf()
    fig = plt.figure()
    plt.plot(resY, Y, 'o')
    plt.xlabel('Predicted Y')
    plt.ylabel('Actual Y')
    plt.savefig("scatter_plot.png", format='png')
    plotLegend = ['Frequency','Phase', 'Max-Min','Price','Price Square Root']
    plt.clf()
    fig = plt.figure()
    plt.plot(daysForFeatureCompare, first200X1)
    plt.plot(daysForFeatureCompare, first200X2)
    plt.plot(daysForFeatureCompare, first200X3)
    plt.plot(daysForFeatureCompare, first200X4**0.33)
    plt.plot(daysForFeatureCompare, first200X5)
    plt.legend(plotLegend,loc=4)
    plt.xlabel("Days")
    plt.ylabel("Values")
    plt.savefig("first200DaysFeatures.png", format='png')

if __name__ == '__main__':
    main()