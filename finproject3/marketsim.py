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


cash_init=sys.argv[1]
orders_file_name=sys.argv[2]
values_file_name=sys.argv[3]
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

dataobj = da.DataAccess('Yahoo')
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

