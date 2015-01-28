import csv
import numpy as np
import pandas as pd
import math
import copy
import QSTK.qstkutil.qsdateutil as du
import datetime as dt
import QSTK.qstkutil.DataAccess as da
import QSTK.qstkutil.tsutil as tsu
import QSTK.qstkstudy.EventProfiler as ep
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import neighbors
from scipy.optimize import leastsq
from sklearn.preprocessing import StandardScaler

lookback_window=100

def getAmplitude(pastValues):
    return max(pastValues)-min(pastValues)

def getFrequency(pastValues):
    mean = sum(pastValues) / len(pastValues)
    frequency = 0
    for i in range(0, lookback_window):
        if pastValues[i]<=mean and pastValues[i+1]>mean:
            frequency = frequency + 1
        elif pastValues[i]<mean and pastValues[i+1]>=mean:
            frequency = frequency + 1
        elif pastValues[i]>=mean and pastValues[i+1]<mean:
            frequency = frequency + 1
        elif pastValues[i]>mean and pastValues[i+1]<=mean:
            frequency = frequency + 1
    
    return frequency

def train(data):
    inpLen = data.shape[1]
    lookback_window=100
    predict_lookback_window=5
    Xtrain = np.zeros([inpLen*(len(data)-(predict_lookback_window+lookback_window)),3])
    Ytrain = np.zeros([inpLen*(len(data)-(predict_lookback_window+lookback_window)),1])
    Y = np.zeros([inpLen*(len(data)-(predict_lookback_window+lookback_window)),1])
    

    count = 0
    for i in range(0,inpLen):
        for j in range (0, len(data)-lookback_window-predict_lookback_window):
            pastValues = data[j:j+lookback_window+1, i]
            
            n = len(pastValues)
            l = np.linspace(0, 4*np.pi, n)

            guess_mean = np.mean(pastValues)
            guess_std = 3*np.std(pastValues)/(2**0.5)
            guess_phase = 0

            data_first_guess = guess_std*np.sin(l+guess_phase) + guess_mean
            optimize_func = lambda x: x[0]*np.sin(l+x[1]) + x[2] - pastValues
            est_std, phase, offset = leastsq(optimize_func, [guess_std, guess_phase, guess_mean])[0]
            
            x1 = getFrequency(pastValues)
            x2 = getAmplitude(pastValues)
            x3 = phase
                    
            Xtrain[count, 0] = x1
            Xtrain[count, 1] = x2            
            Xtrain[count, 2] = x3 
            
            Ytrain[count, 0] = (data[j+lookback_window+predict_lookback_window,i]-data[j+lookback_window,i])/data[j+lookback_window,i]
            Y[count, 0] = data[j+lookback_window,i]
            count = count + 1

    return Xtrain, Ytrain, Y


def getTrainData():
    filenames = []
    
    for i in range (0, 200):
        if i < 10:
            filename = 'ML4T-00'+str(i)
        elif i < 100:
            filename = 'ML4T-0'+str(i)
        else:
            filename = 'ML4T-'+str(i)


        filenames.append(filename)
    
    start = dt.datetime(2001, 1, 1)
    end = dt.datetime(2005, 12, 31)

    timeofday = dt.timedelta(hours=16)
    timestamps = du.getNYSEdays(start, end, timeofday)
    dataobj = da.DataAccess('Yahoo')

    keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']

    data = dataobj.get_data(timestamps, filenames, keys)
    dataDic = dict(zip(keys, data))

    for key in keys:
        dataDic[key] = dataDic[key].fillna(method='ffill')
        dataDic[key] = dataDic[key].fillna(method='bfill')
        dataDic[key] = dataDic[key].fillna(1.0)
    
    priceList = dataDic['actual_close'].values
    
    Xtrain, Ytrain, Y = train(priceList)

    return Xtrain, Ytrain


def getTestData320():
    filenames = []    
    filenames.append('ML4T-320')
    start = dt.datetime(2006, 1, 1)
    end = dt.datetime(2007, 12, 31)

    timeofday = dt.timedelta(hours=16)
    timestamps = du.getNYSEdays(start, end, timeofday)
    dataobj = da.DataAccess('Yahoo')

    keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']

    data = dataobj.get_data(timestamps, filenames, keys)
    dataDic = dict(zip(keys, data))

    for key in keys:
        dataDic[key] = dataDic[key].fillna(method='ffill')
        dataDic[key] = dataDic[key].fillna(method='bfill')
        dataDic[key] = dataDic[key].fillna(1.0)
    
    priceList = dataDic['actual_close'].values

    
    Xtest, Y, Ytest = train(priceList)

    return Xtest, Ytest

def getTestData292():
    filenames = []    
    filenames.append('ML4T-292')
    start = dt.datetime(2006, 1, 1)
    end = dt.datetime(2007, 12, 31)

    timeofday = dt.timedelta(hours=16)
    timestamps = du.getNYSEdays(start, end, timeofday)
    dataobj = da.DataAccess('Yahoo')

    keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']

    data = dataobj.get_data(timestamps, filenames, keys)
    dataDic = dict(zip(keys, data))

    for key in keys:
        dataDic[key] = dataDic[key].fillna(method='ffill')
        dataDic[key] = dataDic[key].fillna(method='bfill')
        dataDic[key] = dataDic[key].fillna(1.0)
    
    priceList = dataDic['actual_close'].values

    
    Xtest, Y, Ytest = train(priceList)

    return Xtest, Ytest    

def findRMS(Y, Ytest):
    total = 0
    for i in range(0, len(Y)):
        total = total + (Y[i] - Ytest[i]) * (Y[i] - Ytest[i])

    rms = math.sqrt(total / len(Y))
    return rms

def findCorrCoef(Y, Ytest):
    corr = np.corrcoef(Y, Ytest)
    return corr[0,1]

def drawScatterPlot(x, y, filename):
    plt.clf()
    fig = plt.figure()
    plt.plot(x, y, 'o')
    plt.xlabel('Predicted Y')
    plt.ylabel('Actual Y')
    plt.savefig(filename, format='png')

def drawLineGraph(xData, y1, y2, filename, linename):
    plt.clf()
    fig = plt.figure()
    plt.plot(xData, y1, color = 'blue')
    plt.plot(xData, y2, color = 'red')
    plt.legend(linename,loc=4)
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.savefig(filename, format='png')

def drawFeatureComparePlot(xData, y1, y2, y3,filename,plotLegend):
    plt.clf()
    fig = plt.figure()
    plt.plot(xData, y1)
    plt.plot(xData, y2)
    plt.plot(xData, y3)
    plt.legend(plotLegend,loc=4)
    plt.xlabel("Days")
    plt.ylabel("Values")
    plt.savefig(filename, format='png')
    
    
def main():
    Xtrain, Ytrain = getTrainData()

#----------------------------
#320.csv (300 + 1(A) + 19(S))
#----------------------------
    Xtest, Ytest, = getTestData320()
    
    first100X1 = Xtest[0:100, 0]
    first100X2 = Xtest[0:100, 1]
    first100X3 = Xtest[0:100, 2]
    
    Y = Ytest[:,0]
    Yzeros = np.zeros([100,1])
    first200Y = Y[0:200]
    last100Y = Y[len(Y)-100: len(Y)]

    firstDate = np.zeros([100])
    lastDate = np.zeros([100])
    
    for i in range (0, 100):
        firstDate[i] = i-1
        lastDate[i] = i-1 
    
    learner=neighbors.KNeighborsRegressor(21)
    queryRes=learner.fit(Xtrain, Ytrain).predict(Xtest)
     
    resY = queryRes[:,-1]

    for i in range (0, resY.shape[0]):
        resY[i] = (resY[i] + 1) * Y[i]
        
    first200ResY = np.zeros([200])
    first200ResY[100:200] = resY[100:200]
    last100ResY = resY[len(resY)-100: len(resY)]


    rfRMS = findRMS(resY, Y)

    rfCorr = findCorrCoef(resY, Y)

    print rfRMS, rfCorr

    #xAxisRange = np.zeros([200])
    #for i in range(0, 200):
    #    xAxisRange[i] = i+1
    xAxisRange = [x for x in range(1,201)]
    
    plotLegend = ['Y actual', 'Y predict']
    drawLineGraph(xAxisRange, first200Y, first200ResY, 'first200Comparison_320.png', plotLegend)
    
    #xAxisRange = np.zeros([100])
    #for i in range(0, 100):
    #    xAxisRange[i] = i+1
    xAxisRange = [x for x in range(1,101)]

    plotLegend = ['Y actual', 'Y predict']
    drawLineGraph(xAxisRange, last100Y, last100ResY, 'last100Comparison_320.png', plotLegend)
    
    drawScatterPlot(resY, Y, 'scatter_plot_320.png')

    linename = ['Frequency', 'Amplitude','Phase']
    drawFeatureComparePlot(firstDate, first100X1, first100X2, first100X3, 'first100Features_320.png', linename)

#----------------------------
#292.csv
#----------------------------

    Xtest, Ytest, = getTestData292()
    
    first100X1 = Xtest[0:100, 0]
    first100X2 = Xtest[0:100, 1]
    first100X3 = Xtest[0:100, 2]
    
    Y = Ytest[:,0]
    Yzeros = np.zeros([100,1])
    first200Y = Y[0:200]
    last100Y = Y[len(Y)-100: len(Y)]

    firstDate = np.zeros([100])
    lastDate = np.zeros([100])
    
    for i in range (0, 100):
        firstDate[i] = i-1
        lastDate[i] = i-1 
    
    learner=neighbors.KNeighborsRegressor(21)
    queryRes=learner.fit(Xtrain, Ytrain).predict(Xtest)
     
    resY = queryRes[:,-1]

    for i in range (0, resY.shape[0]):
        resY[i] = (resY[i] + 1) * Y[i]
        
    first200ResY = np.zeros([200])
    first200ResY[100:200] = resY[100:200]
    last100ResY = resY[len(resY)-100: len(resY)]


    rfRMS = findRMS(resY, Y)

    rfCorr = findCorrCoef(resY, Y)

    print rfRMS, rfCorr

    #xAxisRange = np.zeros([200])
    #for i in range(0, 200):
    #    xAxisRange[i] = i+1
    xAxisRange = [x for x in range(1,201)]
    
    plotLegend = ['Y actual', 'Y predict']
    drawLineGraph(xAxisRange, first200Y, first200ResY, 'first200Comparison_292.png', plotLegend)
    
    #xAxisRange = np.zeros([100])
    #for i in range(0, 100):
    #    xAxisRange[i] = i+1
    xAxisRange = [x for x in range(1,101)]

    plotLegend = ['Y actual', 'Y predict']
    drawLineGraph(xAxisRange, last100Y, last100ResY, 'last100Comparison_292.png', plotLegend)
    
    drawScatterPlot(resY, Y, 'scatter_plot_292.png')

    linename = ['Frequency', 'Amplitude','Phase']
    drawFeatureComparePlot(firstDate, first100X1, first100X2, first100X3, 'first100Features_292.png', linename)

if __name__ == '__main__':
    main()