"""
uses the ML4T-000 .... files from the Yahoo folder in qstk
"""
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

class LinRegLearner(object):

    def __init__(self):       
        self.a1 = None
        self.a2 = None
        self.b = None

    def addEvidence(self,xTrain,yTrain):
        x1 = np.zeros([xTrain.shape[0]])
        x1 = xTrain[:,0]
        x2 = xTrain[:,1]
        A = np.vstack([x1, x2, np.ones(len(x1))]).T
        self.a1, self.a2, self.b = np.linalg.lstsq(A, yTrain)[0]
    
    def query(self,xTest):
        result = np.zeros([xTest.shape[0],xTest.shape[1]+1])
        result[:,0:xTest.shape[1]] = xTest[:,0:xTest.shape[1]]
        for m in range(0,xTest.shape[0]):
            test = xTest[m]
            y = self.a1 * test[0] + self.a2 * test[1] +self.b
            result[m, (xTest.shape[1])] = y

        return result


lookback_window=100

def getFrequency(lookbackValues):
    import scipy.fftpack as fftpack
    yhat = fftpack.rfft(lookbackValues)
    idx = (yhat[1:]**2).argmax() + 1
    freqs = fftpack.rfftfreq(len(lookbackValues), d = (1.0)/(2*np.pi))
    frequency = freqs[idx]
    return frequency

def peakToPeakAmplitude(lookbackValues):
    return max(lookbackValues)-min(lookbackValues)

def getFirstDerivative(lookbackValues):
    return lookbackValues[lookback_window-1]-lookbackValues[lookback_window-2]

def getSecondDerivative(lookbackValues):
    return (lookbackValues[lookback_window-1]-lookbackValues[lookback_window-2])-(lookbackValues[lookback_window-2]-lookbackValues[lookback_window-3])

def train(data):
    inpLen = data.shape[1]
    predict_window=5
    Xtrain = np.zeros([inpLen*(len(data)-(predict_window+lookback_window)),7])
    Ytrain = np.zeros([inpLen*(len(data)-(predict_window+lookback_window)),1])
    Y = np.zeros([inpLen*(len(data)-(predict_window+lookback_window)),1])
    
    index = 0
    for i in range(0,inpLen):
        for j in range (0, len(data)-lookback_window-predict_window):
            lookbackValues = data[j:j+lookback_window+1, i]
            
            n = len(lookbackValues)
            l = np.linspace(0, 4*np.pi, n)

            guess_mean = np.mean(lookbackValues)
            guess_std = 3*np.std(lookbackValues)/(2**0.5)
            guess_phase = 0

            data_first_guess = guess_std*np.sin(l+guess_phase) + guess_mean
            optimize_func = lambda x: x[0]*np.sin(l+x[1]) + x[2] - lookbackValues
            est_std, est_phase, est_offset = leastsq(optimize_func, [guess_std, guess_phase, guess_mean])[0]
            
            x1 = getFrequency(lookbackValues)
            x2 = est_phase
            x3 = peakToPeakAmplitude(lookbackValues)
            x4 = lookbackValues[lookback_window]
            x5 = getFirstDerivative(lookbackValues)
            x6 = lookbackValues[lookback_window]**2
            x7 = lookbackValues[lookback_window]**3
            
                    
            Xtrain[index, 0] = x1
            Xtrain[index, 1] = x2            
            Xtrain[index, 2] = x3 
            Xtrain[index, 3] = x4 
            Xtrain[index, 4] = x5 
            Xtrain[index, 5] = x6 
            Xtrain[index, 6] = x7 

            
            Ytrain[index, 0] = (data[j+lookback_window+predict_window,i]-data[j+lookback_window,i])/data[j+lookback_window,i]
            Y[index, 0] = data[j+lookback_window,i]
            index = index + 1

    return Xtrain, Ytrain, Y

def getTestData309():
    filenames = []    
    filenames.append('ML4T-309')
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
    
    
def main():
    Xtrain, Ytrain = getTrainData()

#----------------------------
#292.csv
#----------------------------

    Xtest, Ytest, = getTestData292()
    
    init200X1 = Xtest[0:200, 0]
    init200X2 = Xtest[0:200, 1]
    init200X3 = Xtest[0:200, 2]
    init200X4 = Xtest[0:200, 3]
    init200X5 = Xtest[0:200, 4]
    

    linLearner = LinRegLearner()
    linLearner.addEvidence(Xtrain,Ytrain)

    Y = Ytest[:,0]
    Yzeros = np.zeros([100,1])
    init200Y = Y[0:200]
    end200Y = Y[len(Y)-200: len(Y)]

    daysForFeatureCompare = [x for x in range(0,200)]
    
    print "for 292.csv"
    queryRes=linLearner.query(Xtest)
             
    resY = queryRes[:,-1]

    for i in range (0, resY.shape[0]):
        resY[i] = (resY[i] + 1) * Y[i]
                
    init200ResY = np.zeros([200])
    init200ResY[0:100] = None
    init200ResY[100:200] = resY[100:200]
    end200ResY = resY[len(resY)-200: len(resY)]

    rmsVal = math.sqrt(np.mean((resY - Y) ** 2))
    corrCoefVal = np.corrcoef(resY,Y)[0,1]
    print rmsVal,corrCoefVal

    xAxisRange = [x for x in range(1,201)]
    
    plotLegend = ['Y actual', 'Y predict']
    plt.clf()
    fig = plt.figure()
    plt.plot(xAxisRange, init200Y, color = 'blue')
    plt.plot(xAxisRange, init200ResY, color = 'red')
    plt.legend(plotLegend,loc=4)
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.savefig("init200Comparison_292", format='png')
    print "created init200Comparison_292.png"
    xAxisRange = [x for x in range(1,201)]

    plotLegend = ['Y actual', 'Y predict']
    plt.clf()
    fig = plt.figure()
    plt.plot(xAxisRange, end200Y, color = 'blue')
    plt.plot(xAxisRange, end200ResY, color = 'red')
    plt.legend(plotLegend,loc=4)
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.savefig("end200Comparison_292", format='png')    

    print "created end200Comparison_292.png"
    plt.clf()
    fig = plt.figure()
    plt.plot(resY, Y, 'o')
    plt.xlabel('Predicted Y')
    plt.ylabel('Actual Y')
    plt.savefig("scatter_plot_292", format='png')
    print "created scatter_plot_292.png"
    plotLegend = ['Frequency','Phase', 'Amplitude','Price','First Derivative']
    plt.clf()
    fig = plt.figure()
    plt.plot(daysForFeatureCompare, init200X1)
    plt.plot(daysForFeatureCompare, init200X2)
    plt.plot(daysForFeatureCompare, init200X3)
    plt.plot(daysForFeatureCompare, init200X4**0.33)
    plt.plot(daysForFeatureCompare, init200X5)
    plt.legend(plotLegend,loc=4)
    plt.xlabel("Days")
    plt.ylabel("Values")
    plt.savefig("init200DaysFeatures_292", format='png')
    print "created init200DaysFeatures_292.png"
#----------------------------
#320.csv (300 + 1(A) + 19(S))
#----------------------------
    Xtest, Ytest, = getTestData320()
    
    Y = Ytest[:,0]
    Yzeros = np.zeros([100,1])
    init200Y = Y[0:200]
    end200Y = Y[len(Y)-200: len(Y)]

    daysForFeatureCompare = [x for x in range(0,200)]
    
    print "for 309.csv"
    
    queryRes=linLearner.query(Xtest)
             
    resY = queryRes[:,-1]

    for i in range (0, resY.shape[0]):
        resY[i] = (resY[i] + 1) * Y[i]
                
    init200ResY = np.zeros([200])
    init200ResY[0:100] = None
    init200ResY[100:200] = resY[100:200]
    end200ResY = resY[len(resY)-200: len(resY)]

    rmsVal = math.sqrt(np.mean((resY - Y) ** 2))
    corrCoefVal = np.corrcoef(resY,Y)[0,1]
    print rmsVal,corrCoefVal

    xAxisRange = [x for x in range(1,201)]
    
    plotLegend = ['Y actual', 'Y predict']
    plt.clf()
    fig = plt.figure()
    plt.plot(xAxisRange, init200Y, color = 'blue')
    plt.plot(xAxisRange, init200ResY, color = 'red')
    plt.legend(plotLegend,loc=4)
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.savefig("init200Comparison_320", format='png')
    print "created init200Comparison.png"

    plotLegend = ['Y actual', 'Y predict']
    plt.clf()
    fig = plt.figure()
    plt.plot(xAxisRange, end200Y, color = 'blue')
    plt.plot(xAxisRange, end200ResY, color = 'red')
    plt.legend(plotLegend,loc=4)
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.savefig("end200Comparison", format='png')    

    print "created end200Comparison.png"
    plt.clf()
    fig = plt.figure()
    plt.plot(resY, Y, 'o')
    plt.xlabel('Predicted Y')
    plt.ylabel('Actual Y')
    plt.savefig("scatter_plot", format='png')

     

if __name__ == '__main__':
    main()