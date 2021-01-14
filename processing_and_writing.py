import time
import dask
import numpy as np
import pandas as pd
import glob
import os
import pyarrow as pa
import pyarrow.parquet as pq
from multiprocessing import Pool

number_of_trading_days = 5


def load_trade(filename,
               tz_exchange="America/New_York",
               only_non_special_trades=True,
               only_regular_trading_hours=True,
               open_time="09:30:00",
               close_time="16:00:00",
               merge_sub_trades=True):
    DF = pd.read_csv(filename)
    if DF.empty:
        return None
    if only_non_special_trades:
        DF = DF[DF["trade-stringflag"] == "uncategorized"]
    DF.drop(columns=["trade-rawflag", "trade-stringflag"], axis=1, inplace=True)
    DF.index = pd.to_datetime(DF["xltime"], unit="D", origin="1899-12-30", utc=True)
    DF.index = DF.index.tz_convert(tz_exchange)
    DF.drop(columns="xltime", inplace=True)
    if only_regular_trading_hours:
        DF = DF.between_time(open_time, close_time)
    if merge_sub_trades:
        DF = DF.groupby(DF.index).agg(trade_price=pd.NamedAgg(column='trade-price', aggfunc='mean'),
                                      trade_volume=pd.NamedAgg(column='trade-volume', aggfunc='sum'))
    return DF


def load_bbo(filename,
             tz_exchange="America/New_York",
             open_time="09:30:00",
             close_time="16:00:00",
             only_regular_trading_hours=True):
    DF = pd.read_csv(filename)
    if DF.empty:
        return None
    DF.index = pd.to_datetime(DF["xltime"], unit="D", origin="1899-12-30", utc=True)
    DF.index = DF.index.tz_convert(tz_exchange)
    DF.drop(columns="xltime", inplace=True)
    if only_regular_trading_hours:
        DF = DF.between_time(open_time, close_time)
    return DF


@dask.delayed
def load_trade_dask(filename):
    DF = load_trade(filename)
    return DF


@dask.delayed
def load_bbo_dask(filename):
    DF = load_bbo(filename)
    return DF


def type_is_not_None(obj):
    if type(obj) is type(None):
        return False
    return True


path = "/Users/bkuznets/FBD_project/data/data/extraction/TRTH/raw/equities/US"
allfiles_trade = glob.glob(os.path.join(path, "trade/AAPL.OQ/*"))
allfiles_bbo = glob.glob(os.path.join(path, "bbo/AAPL.OQ/*"))
allfiles_trade = np.sort(allfiles_trade)[:number_of_trading_days]
allfiles_bbo = np.sort(allfiles_bbo)[:number_of_trading_days]

t0 = time.time()
df_trade = pd.DataFrame()
for fn in allfiles_trade:
    alltrades = load_trade(fn)
    if type_is_not_None(alltrades):
        df_trade = df_trade.append(alltrades)
df_bbo = pd.DataFrame()
for fn in allfiles_bbo:
    allbbos = load_bbo(fn)
    if type_is_not_None(allbbos):
        df_bbo = df_bbo.append(allbbos)
allevents = df_trade.join(df_bbo, how='inner')
allevents.ffill(inplace=True)
t1 = time.time()
preprocessing_time = t1 - t0

t0 = time.time()
df_trade = pd.DataFrame()
df_bbo = pd.DataFrame()
if __name__ == '__main__':
    pool = Pool(os.cpu_count())
    alltrades_multi = pool.map(load_trade, allfiles_trade)
    allbbos_multi = pool.map(load_bbo, allfiles_bbo)
    alltrades_multi = list(filter(type_is_not_None, alltrades_multi))
    df_trade = df_trade.append(alltrades_multi)
    allbbos_multi = list(filter(type_is_not_None, allbbos_multi))
    df_bbo = df_bbo.append(allbbos_multi)
    allevents_multi = df_trade.join(df_bbo, how='inner')
    allevents_multi.ffill(inplace=True)
t1 = time.time()
multiprocessing_time = t1 - t0

t0 = time.time()
df_trade = pd.DataFrame()
df_bbo = pd.DataFrame()
if __name__ == '__main__':
    allpromises_trade = [load_trade_dask(fn) for fn in allfiles_trade]
    allpromises_bbo = [load_bbo_dask(fn) for fn in allfiles_bbo]
    alltrades_dask = dask.compute(allpromises_trade)[0]
    allbbos_dask = dask.compute(allpromises_bbo)[0]
    alltrades_dask = list(filter(type_is_not_None, alltrades_dask))
    allbbos_dask = list(filter(type_is_not_None, allbbos_dask))
    df_trade = df_trade.append(alltrades_dask)
    df_bbo = df_bbo.append(allbbos_dask)
    allevents_dask = df_trade.join(df_bbo, how='inner')
    allevents_dask.ffill(inplace=True)
t1 = time.time()
dask_time = t1 - t0

# t0 = time.time()
# allevents.to_csv('AAPL.OQ.csv.gz', compression='gzip')
# t1 = time.time()
# write_to_csvgz_time = t1 - t0
# t0 = time.time()
# allevents.to_parquet('AAPL.OQ.parquet', index=False)
# t1 = time.time()
# write_to_parquet_time = t1 - t0
# t0 = time.time()
# allevents.to_hdf('AAPL.OQ.h5', key='allevents')
# t1 = time.time()
# write_to_hdf5_time = t1 - t0
# t0 = time.time()
# table = pa.Table.from_pandas(allevents, preserve_index=False)
# pq.write_table(table, 'AAPL.OQ.parquet')
# t1 = time.time()
# write_to_parquet_via_pyarrow_time = t1 - t0
#
# print('Processing time, straightforward computation...', preprocessing_time, 's')
# print('Processing time, multiprocessing, all cores...', preprocessing_time, 's')
# print('Processing time, dask...', preprocessing_time, 's')
# print('Writing time, csv.gz...', write_to_csvgz_time, 's')
# print('Writing time, parquet...', write_to_parquet_time, 's')
# print('Writing time, hdf5...', write_to_hdf5_time, 's')
# print('Writing time, parquet via pyarrow...', write_to_parquet_via_pyarrow_time, 's')
