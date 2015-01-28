import sys
import csv
import datetime as dt

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math as math
import sys
import copy

import QSTK.qstkutil.qsdateutil as du
import QSTK.qstkutil.tsutil as tsu
import QSTK.qstkutil.DataAccess as da

values_file_name=sys.argv[1]
symbol_compare=sys.argv[2]
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
