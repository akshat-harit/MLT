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


import pandas as pd
import copy
import QSTK.qstkstudy.EventProfiler as ep


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
print ldt_timestamps[i]
df_close = d_data['actual_close']            
df_events = copy.deepcopy(df_close)
df_events = df_events * np.NAN
#norm = na_price / na_price[0,:]


for s_sym in ls_symbols: # for each symbol
    print "Symbol : "+str(s_sym)
    for i in range(1, len(ldt_timestamps)): # for each day
        # Calculating the returns for this timestamp
        #print ldt_timestamps[i]
        #print s_sym
        df_close[s_sym]
        f_symprice_today = df_close[s_sym].ix[ldt_timestamps[i]]
        f_symprice_yest = df_close[s_sym].ix[ldt_timestamps[i - 1]]
        f_symprice_yest_prev=df_close[s_sym].ix[ldt_timestamps[i - 2]]
        if (f_symprice_today-f_symprice_yest >= 5): #and (f_symprice_yest - f_symprice_yest_prev> 5): 
              df_events[s_sym].ix[ldt_timestamps[i]] = 1
              #print "even happened for "+str(s_sym)

        # Event is found if the symbol is down more then 3% while the
        # market is up more then 2%
        # if f_symreturn_today <= -0.03 and f_marketreturn_today >= 0.02:
        #      df_events[s_sym].ix[ldt_timestamps[i]] = 1


print "Creating Study"
ep.eventprofiler(df_events, d_data, i_lookback=20, i_lookforward=20,
                s_filename='MyEventStudy.pdf', b_market_neutral=True, b_errorbars=True,
                s_market_sym='SPY')

#