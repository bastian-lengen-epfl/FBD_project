import numpy as np
import pandas as pd
import glob
import os
import dask.delayed
import pyarrow as pa
import time
import tables

dask.config.set(scheduler="processes")


@dask.delayed
def load_TRTH_trade(filename,
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
    DF.index = DF.index.tz_convert(tz_exchange)  # .P stands for Arca, which is based at New York
    DF.drop(columns="xltime", inplace=True)

    if only_regular_trading_hours:
        DF = DF.between_time(open_time, close_time)  # warning: ever heard e.g. about Thanksgivings?

    if merge_sub_trades:
        DF = DF.groupby(DF.index).agg(trade_price=pd.NamedAgg(column='trade-price', aggfunc='mean'),
                                      trade_volume=pd.NamedAgg(column='trade-volume', aggfunc='sum'))

    return DF


@dask.delayed
def load_TRTH_bbo(filename,
                  tz_exchange="America/New_York",
                  open_time="09:30:00",
                  close_time="16:00:00",
                  only_regular_trading_hours=True):
    DF = pd.read_csv(filename)
    if DF.empty:
        return None

    DF.index = pd.to_datetime(DF["xltime"], unit="D", origin="1899-12-30", utc=True)
    DF.index = DF.index.tz_convert(tz_exchange)  # .P stands for Arca, which is based at New York
    DF.drop(columns="xltime", inplace=True)

    if only_regular_trading_hours:
        DF = DF.between_time(open_time, close_time)  # warning: ever heard e.g. about Thanksgivings?

    return DF


def type_is_not_None(obj):
    if type(obj) is type(None):
        return False
    return True


# Set paths and file names
path_raw = "/Users/bkuznets/FBD_project/data/data/extraction/TRTH/raw/equities/US"
path_clean = "/Users/bkuznets/FBD_project/data/data/extraction/TRTH/raw/equities/US"

# List of the files
allfiles_trade = glob.glob(os.path.join(path_raw, "trade/AAPL.OQ/*"))
allfiles_bbo = glob.glob(os.path.join(path_raw, "bbo/AAPL.OQ/*"))

# Only keeps the 5 first to begin with
allfiles_trade = np.sort(allfiles_trade)[:5]
allfiles_bbo = np.sort(allfiles_bbo)[:5]

if __name__ == '__main__':
    # Load the TRTH data
    allpromises_trade = [load_TRTH_trade(fn) for fn in allfiles_trade]
    allpromises_bbo = [load_TRTH_bbo(fn) for fn in allfiles_bbo]

    # Compute with dask
    alltrades = dask.compute(allpromises_trade)[0]
    allbbos = dask.compute(allpromises_bbo)[0]

    # Remove the None from the empty DF
    alltrades = list(filter(type_is_not_None, alltrades))
    allbbos = list(filter(type_is_not_None, allbbos))

    # Regroup the DF
    alltrades = pd.concat(alltrades)
    allbbos = pd.concat(allbbos)

    # Join them together
    allevents = alltrades.join(allbbos, how='inner')

    # Fill the DF
    allevents.ffill(inplace=True)

    # Save the cleaned DF
    if not os.path.exists(path_clean):
        os.makedirs(path_clean)

    # Save in 4 different formats and compare timings
    # CSV.GZ
    t0 = time.time()
    allevents.to_csv(os.path.join(path_clean, "AAPL.OQ.csv.gz"), compression='gzip')
    t1 = time.time()
    write_csv_time = t1 - t0

    # PARQUET.GZ
    t0 = time.time()
    # allevents.index = pd.to_datetime(allevents.index, unit='us')
    allevents.to_parquet(os.path.join(path_clean, "AAPL.OQ.parquet.gz"), index=False, compression='gzip')
    t1 = time.time()
    write_parquet_time = t1 - t0

    # HDF5
    t0 = time.time()
    allevents.to_hdf(os.path.join(path_clean, "AAPL.OQ.h5"), key='allevents')
    t1 = time.time()
    write_hdf5_time = t1 - t0

    # PYARROW
    # TODO: find out how to write pyarrow to file. For now I only convert the pandas table to pyarrow.
    t0 = time.time()
    pa.Table.from_pandas(allevents)
    t1 = time.time()
    write_arrow_time = t1 - t0
