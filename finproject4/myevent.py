#Akshat Harit
#GTId: 903090915
#finporject2

import QSTK.qstkutil.qsdateutil as du
import QSTK.qstkutil.tsutil as tsu
import QSTK.qstkutil.DataAccess as da

import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math as math
import sys
import copy
import QSTK.qstkstudy.EventProfiler as ep
import csv

def func1():
    dataobj = da.DataAccess('Yahoo')
    symbols8 = dataobj.get_symbols_from_list("sp5002008")
    symbols8.append('SPY')
    symbols12 = dataobj.get_symbols_from_list("sp5002012")
    symbols12.append('SPY')
    startdate=dt.datetime(2008,1,1)
    enddate=dt.datetime(2009, 12,31)


    dt_timeofday = dt.timedelta(hours=16)
    dt_start = startdate
    dt_end = enddate
    ls_symbols=symbols12
    ldt_timestamps = du.getNYSEdays(startdate, enddate, dt_timeofday)
    ls_keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']
    ldf_data = dataobj.get_data(ldt_timestamps, ls_symbols, ls_keys)
    d_data = dict(zip(ls_keys, ldf_data))
    for s_key in ls_keys:
        d_data[s_key] = d_data[s_key].fillna(method = 'ffill')
        d_data[s_key] = d_data[s_key].fillna(method = 'bfill')
        d_data[s_key] = d_data[s_key].fillna(1.0)
    #print ldt_timestamps[1].year
    df_close = d_data['actual_close']            
    df_events = copy.deepcopy(df_close)
    df_events = df_events * np.NAN
    #norm = na_price / na_price[0,:]
    f1=open("orders.csv", "w")

    for s_sym in ls_symbols: # for each symbol
        #print "Symbol : "+str(s_sym)
        for i in range(1, len(ldt_timestamps)): # for each day
            df_close[s_sym]
            f_symprice_today = df_close[s_sym].ix[ldt_timestamps[i]]
            f_symprice_yest = df_close[s_sym].ix[ldt_timestamps[i - 1]]
            f_symprice_yest_prev=df_close[s_sym].ix[ldt_timestamps[i - 2]]
            if (f_symprice_today-f_symprice_yest >= 5):  
                df_events[s_sym].ix[ldt_timestamps[i]] = 1
                  #print "even happened for "+str(s_sym)
                today=ldt_timestamps[i]
                if(i<len(ldt_timestamps)-5):
                    fiveday=ldt_timestamps[i+5]
                else:
                    fiveday=ldt_timestamps[len(ldt_timestamps)-1]
                    print "##"

                print >>f1, str(today.year)+","+str(today.month)+","+str(today.day)+","+str(s_sym)+","+"Sell"+","+str(100)
                print >> f1, str(fiveday.year)+","+str(fiveday.month)+","+str(fiveday.day)+","+str(s_sym)+","+"Buy"+","+str(100)

    print "Creating Study"
    ep.eventprofiler(df_events, d_data, i_lookback=20, i_lookforward=20,
                    s_filename='MyEventStudy.pdf', b_market_neutral=True, b_errorbars=True,
                    s_market_sym='SPY')



#################################################################################
def func2():
    print "Market Sim"
    cash_init=50000
    dataobj = da.DataAccess('Yahoo')
    orders_file_name="orders.csv"
    values_file_name="value.csv"
    reader=csv.reader(open(orders_file_name, 'rU'), delimiter=',')
    symbol_list=[]
    date_trade_list=[]
    date_list=[]
    for row in reader:
        symbol_list.append(row[3])
        date=dt.datetime(int(row[0]), int(row[1]),int(row[2]),16)   
        date_trade_list.append([date, row[3],row[4],row[5]])
        date_list.append(date)

    symbol_list=list(set(symbol_list))
    date_trade_list=sorted(date_trade_list)
    date_list=sorted(list(set(date_list)))
    dt_end_read=date_trade_list[-1][0]+dt.timedelta(days=1)

    symbols = symbol_list
    startdate=date_trade_list[0][0]
    enddate=dt_end_read


    dt_timeofday = dt.timedelta(hours=16)
    dt_start = startdate
    dt_end = enddate
    ls_symbols=symbols
    ldt_timestamps = du.getNYSEdays(startdate, enddate, dt_timeofday)
    ls_keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']
    ldf_data = dataobj.get_data(ldt_timestamps, ls_symbols, ls_keys)
    d_data = dict(zip(ls_keys, ldf_data))
    for s_key in ls_keys:
        d_data[s_key] = d_data[s_key].fillna(method = 'ffill')
        d_data[s_key] = d_data[s_key].fillna(method = 'bfill')
        d_data[s_key] = d_data[s_key].fillna(1.0)

    df_close = d_data['close']            
    df=pd.DataFrame(np.zeros((len(ldt_timestamps), len(symbols))), index=ldt_timestamps, columns=symbols)
    df_price=pd.DataFrame(np.zeros((len(ldt_timestamps), len(symbols))), index=ldt_timestamps, columns=symbols)

    for index, row in df_price.iterrows():
        date=index
        for sym in symbols:
            row[sym]=df_close.loc[date,sym]

    cash=pd.Series(np.zeros(len(ldt_timestamps)),index=ldt_timestamps)
    cash.at[ldt_timestamps[0]]=cash_init
    for i, row in enumerate(date_trade_list):
        date, sym, decision, quant=row[0],row[1],row[2],row[3]
        if(decision=="Buy"):
            df.at[date, sym]+=int(quant)
            print "Buying "+quant+ " of "+str(sym)
            cash.loc[date]-=int(quant)*df_close.loc[date, sym]
        else:
            df.at[date,sym]-=int(quant)
            print "Selling "+quant+ " of "+str(sym) 
            cash.loc[date]+=int(quant)*df_close.loc[date, sym]



    df_price['_CASH']=1
    df['_CASH']=cash
    holdings=df.cumsum(axis=0)
    ts_fund=holdings.mul(df_price,1)
    total2=holdings.dot(df_price.transpose())
    ts_fund=ts_fund.sum(axis=1)

    writer=csv.writer(open(values_file_name,'wb'),delimiter=',')
    for row_index in ts_fund.index[:-1]:
        a=row_index.to_datetime()
        row_to_enter=[str(a.year), str(a.month), str(a.day), str(ts_fund[row_index])]
        writer.writerow(row_to_enter) 


############################################################################
def func3():
    values_file_name="value.csv"
    symbol_compare="$SPX"
    csvreader=csv.reader(open(values_file_name, 'rU'), delimiter=',')
    reader=[]
    for line in csvreader:
        reader.append(line)
    price_list=[]
    for l in reader:
        price_list.append(float(l[3]))

    price_list=np.array(price_list)
    temp_date=reader[0]
    startdate=dt.datetime(int(temp_date[0]), int(temp_date[1]),int(temp_date[2]))
    temp_date=reader[-1]
    enddate=dt.datetime(int(temp_date[0]), int(temp_date[1]),int(temp_date[2]))+dt.timedelta(days=1)

    dataobj = da.DataAccess('Yahoo')
    ls_symbols = [symbol_compare]
    dt_timeofday = dt.timedelta(hours=16)

    ldt_timestamps = du.getNYSEdays(startdate, enddate, dt_timeofday)
    dataobj = da.DataAccess('Yahoo')
    ls_keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']
    ldf_data = dataobj.get_data(ldt_timestamps, ls_symbols, ls_keys)
    d_data = dict(zip(ls_keys, ldf_data))
    for s_key in ls_keys:
        d_data[s_key] = d_data[s_key].fillna(method = 'ffill')
        d_data[s_key] = d_data[s_key].fillna(method = 'bfill')
        d_data[s_key] = d_data[s_key].fillna(1.0)

    df_close = d_data['close']


    na_price = df_close.values
    norm = na_price / na_price[0,:]
    allocation=norm
    total=np.sum(allocation, axis=1)
    rf_daily=total*price_list[0]
    cum_ret=total[-1]
    daily_rets=tsu.returnize0(total)
    vol=np.std(daily_rets)
    avg=np.average(daily_rets)
    sharpe=(avg/vol)*math.sqrt(252)
    rf_vol, rf_avg, rf_sharpe, rf_cum_ret= vol, avg, sharpe,cum_ret


    na_price = price_list
    norm = na_price / na_price[0]
    allocation=norm
    total=allocation
    fl_daily=total*price_list[0]
    cum_ret=total[-1]
    daily_rets=tsu.returnize0(total)
    vol=np.std(daily_rets)
    avg=np.average(daily_rets)
    sharpe=(avg/vol)*math.sqrt(252)
    fl_vol, fl_avg, fl_sharpe, fl_cum_ret= vol, avg, sharpe,cum_ret

    #print fl_vol, fl_avg, fl_sharpe, fl_cum_ret

    print "The final value of the portfolio using the sample file is -- " +str(df_close.index[-1])+": "+str(price_list[0]*fl_cum_ret)
    print "Details of the Performance of the portfolio :"
    print ""
    print "Data Range :  "+str(df_close.index[0])+ " to " +str(df_close.index[-1])
    print ""
    print "Sharpe Ratio of Fund : "+str(fl_sharpe)
    print "Sharpe Ratio of "+str(symbol_compare)+" : "+str(rf_sharpe)
    print ""
    print "Total Return of Fund : "+str(fl_cum_ret)
    print "Total Return of "+str(symbol_compare)+" : "+str(rf_cum_ret)
    print ""
    print "Standard Deviation of Fund :  "+str(fl_vol)
    print "Standard Deviation of "+str(symbol_compare)+" : "+str(rf_vol)
    print ""
    print "Average Daily Return of Fund :  "+str(fl_avg)
    print "Average Daily Return of "+str(symbol_compare)+" : "+str(rf_avg)

    rf_line=plt.plot(ldt_timestamps, rf_daily,'g-', label="Reference "+str(symbol_compare))
    fl_line=plt.plot(ldt_timestamps, fl_daily, 'r-', label="Analyzed Fund")

    plt.ylabel('Fund Value')
    plt.xlabel('Date')
    plt.legend()
    plt.show()
    savepath="plt.jpeg"
    plt.savefig(savepath)
    plt.close()


#func1()
#func2()
func3()

