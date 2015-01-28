#Akshat Harit
#GTId: 903090915
#finporject1

import QSTK.qstkutil.qsdateutil as du
import QSTK.qstkutil.tsutil as tsu
import QSTK.qstkutil.DataAccess as da

import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math as math
import sys

startyear=int(sys.argv[1]) 
startmonth=int(sys.argv[2]) 
startday=int(sys.argv[3]) 
endyear=int(sys.argv[4] )
endmonth=int(sys.argv[5]) 
endday=int(sys.argv[6] )
symbol1=sys.argv[7] 
symbol2=sys.argv[8] 
symbol3=sys.argv[9] 
symbol4=sys.argv[10]


startdate=dt.datetime(startyear,startmonth,startday)
enddate=dt.datetime(endyear, endmonth,endday)

ls_symbols = [symbol1, symbol2, symbol3, symbol4]
dt_timeofday = dt.timedelta(hours=16)

c_dataobj = da.DataAccess('Yahoo')

def simulate(startdate, enddate, symbols, portfolio):
	dt_timeofday = dt.timedelta(hours=16)
	dt_start = startdate
	dt_end = enddate
	ls_symbols=symbols
	ldt_timestamps = du.getNYSEdays(startdate, enddate, dt_timeofday)
	ls_keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']
	ldf_data = c_dataobj.get_data(ldt_timestamps, ls_symbols, ls_keys)
	d_data = dict(zip(ls_keys, ldf_data))
	na_price = d_data['close'].values
	norm = na_price / na_price[0,:]
	allocation=portfolio*norm
	total=np.sum(allocation, axis=1)
	cum_ret=total[-1]
	daily_rets=tsu.returnize0(total)
	vol=np.std(daily_rets)
	avg=np.average(daily_rets)
	sharpe=(avg/vol)*math.sqrt(252)
	return vol, avg, sharpe,cum_ret


sharpe_best=0
list=range(0,11)
list=[i/10.0 for i in list]
for p in list:
	for q in list:
		for r in list:
			for s in list:
				if(p+q+r+s==1):
					vol, daily_ret, sharpe, cum_ret = simulate(startdate, enddate, ls_symbols, [p,q,r,s])
					if(sharpe>sharpe_best):
						vol_best=vol
						daily_ret_best=daily_ret
						sharpe_best=sharpe
						cum_ret_best=cum_ret
						allocation=[p, q, r ,s]



print "Start Date: "+startdate.strftime("%b, %d %Y")
print "End Date: "+enddate.strftime("%b, %d %Y")
print "Symbols: "+str(ls_symbols)
print "Optimal Allocations: "+str(allocation)
print "Sharpe Ratio: "+str(sharpe_best)
print "Volatility (stdev of daily returns): "+str(vol_best)
print "Average Daily Return: "+str(daily_ret_best) 
print "Cumulative Return:  "+str(cum_ret_best)
