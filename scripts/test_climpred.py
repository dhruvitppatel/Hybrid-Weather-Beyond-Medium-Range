import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import numpy as mp

from climpred import HindcastEnsemble
import climpred

def decode_cf(ds, time_var):
    """Decode time dimension to CFTime standards."""

    if ds[time_var].attrs["calendar"] == "360":
       ds[time_var].attrs["calendar"] = "360_day"
 
    ds = xr.decode_cf(ds, decode_times=True)
    return ds

url = 'http://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/NCEP-CFSv2/.HINDCAST/.MONTHLY/.sst/X/190/240/RANGEEDGES/Y/-5/5/RANGEEDGES/[X%20Y%20M]average/dods'
#url = 'http://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/NCEP-CFSv2/.FORECAST/.MONTHLY/.sst/X/190/240/RANGEEDGES/Y/-5/5/RANGEEDGES/[X%20Y%20M]average//dods'
fcstds = xr.open_dataset(url, decode_times=False)
fcstds = decode_cf(fcstds, 'S').compute()
#print(f'fcstds: {fcstds}\n')

fcstds = fcstds.rename({"S": "init", "L": "lead"})
fcstds["lead"] = (fcstds["lead"] - 0.5).astype("int")
fcstds["lead"].attrs = {"units": "months"}
print(f'fcstds: {fcstds}\n')

obsurl = 'http://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.OIv2_SST/.sst/X/190/240/RANGEEDGES/Y/-5/5/RANGEEDGES/[X%20Y]average/dods' 
verifds = xr.open_dataset(obsurl, decode_times=False)
verifds = decode_cf(verifds, 'T').compute()
#print(f'verifds: {verifds}')

verifds = verifds.rename({"T": "time"})
verifds["time"] = xr.cftime_range(start="1982-01", periods=verifds["time"].size, freq="MS", calendar="360_day")
print(f'verifds: {verifds}\n')

#fcstds = fcstds.sel(init=slice("1982-01-01", "2010-12-01"))
#verifds = verifds.sel(time=slice("1982-01-01", "2010-12-01"))

hindcast = HindcastEnsemble(fcstds)
print(f'hindcast: {hindcast}')
hindcast = hindcast.add_observations(verifds)
hindcast = hindcast.remove_seasonality("month")
result = hindcast.verify(metric="rmse", comparison="e2o", dim="init", alignment="maximize", groupby="month")

result.sst.plot(y="lead", cmap="YlOrRd", vmin=0.0, vmax=1.0)
print(f'result.sst: {result.sst}\n')

sst_rmse = result.sst.mean("month")
print(f'sst_rmse: {sst_rmse}\n')
fig, ax = plt.subplots()
ax.plot(result["lead"],sst_rmse,'-k')
ax.set_xlabel("Lead Time (months)")
ax.set_ylabel("RMSE")
ax.grid()
plt.show()

plt.title("NCEP-CFSv2 Nino3.4 RMSE")
plt.xlabel("Initial Month")
plt.ylabel("Lead Time (Months)")
plt.show()
