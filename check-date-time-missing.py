import xarray as xr
import numpy as np
import os.path
from datetime import datetime, timedelta

def datespan(start_date, end_date, delta):
    current_date = start_date
    while current_date < end_date:
        yield current_date
        current_date += delta
def nextday(start_date, end_date, delta):
    current_date = start_date
    while current_date <= end_date:
        yield current_date
        current_date += delta

start_date = datetime(2016, 1, 1, 0, 0)
end_date = datetime(2026, 7, 1, 0, 0)
for timestamp in datespan(start_date, end_date, delta=timedelta(hours=1)):
    dd=timestamp.strftime("%d")
    mm=timestamp.strftime("%m")
    yy=timestamp.strftime("%Y")
    hh=timestamp.strftime("%H%M")
    if (os.path.isfile("/path/to/data/netcdf/MODIS/MODIS_IST_CARRA2_"+yy+""+mm+""+dd+"_"+hh+".nc")):
            ds = xr.open_dataset("/path/to/data/netcdf/MODIS/MODIS_IST_CARRA2_"+yy+""+mm+""+dd+"_"+hh+".nc")
            nextday=timestamp+timedelta(hours=1)
            for hourstamp in datespan(timestamp, nextday, delta=timedelta(minutes=60)):
                if np.datetime64(hourstamp) not in ds.time.values:
                    print('',hourstamp.strftime(" - %Y-%m-%d %H:%M:%S"))
    else:
        print('',timestamp.strftime(" - %Y-%m-%d %H:00:00"))
