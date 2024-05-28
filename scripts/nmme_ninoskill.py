import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import xarray as xr
import pandas as pd
import numpy as np

from climpred import HindcastEnsemble
import climpred


def decode_cf(ds, time_var):
    """Decodes time dimension to CFTime standards."""

    if ds[time_var].attrs["calendar"] == "360":
       ds[time_var].attrs["calendar"] = "360_day"
    ds = xr.decode_cf(ds, decode_times=True)
    return ds

# server-side average over enso region and ensemble members
url = 'http://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/NCEP-CFSv2/.HINDCAST/.MONTHLY/.sst/X/190/240/RANGEEDGES/Y/-5/5/RANGEEDGES/[X%20Y%20M]average/dods'
#url = 'http://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/NCEP-CFSv2/.FORECAST/.PENTAD_SAMPLES/.MONTHLY/.sst/X/190/240/RANGEEDGES/Y/-5/5/RANGEEDGES/[X%20Y%20M]average/dods'
fcstds = xr.open_dataset(url, decode_times=False)
fcstds = decode_cf(fcstds, 'S').compute()
print(f'\nfcstds: {fcstds}\n')

# reformat dimensions/attributes according to climpred convention
fcstds = fcstds.rename({"S": "init", "L": "lead"})
fcstds["lead"] = (fcstds["lead"] - 0.5).astype("int")
fcstds["lead"].attrs = {"units": "months"}

# obtain the verification SST data
obsurl = 'http://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.OIv2_SST/.sst/X/190/240/RANGEEDGES/Y/-5/5/RANGEEDGES/[X%20Y]average/dods'
verifds = xr.open_dataset(obsurl, decode_times=False)
verifds = decode_cf(verifds, 'T').compute()
print(f'\nverifds: {verifds}\n')

# reformat dimensions to climpred convention
verifds = verifds.rename({"T": "time"})
verifds["time"] = xr.cftime_range(start="1982-01", periods=verifds["time"].size, freq="MS", calendar="360_day")

# Select appropriate subset of verification data
fcstds = fcstds.sel(init=slice("2003-01-01", "2009-12-01"))
verifds = verifds.sel(time=slice("2003-01-01", "2009-12-01"))

# calculate acc
hindcast = HindcastEnsemble(fcstds)
hindcast = hindcast.add_observations(verifds)
hindcast = hindcast.remove_seasonality("month")
result = hindcast.verify(metric="acc", comparison="e2o", dim="init", alignment="maximize", groupby="month")

result.sst.plot(y="lead", cmap="YlOrRd", vmin=0.0, vmax=1.0)
print(f'\n\nresult.sst: {result.sst}')
print(f'\n\nresult.sst.to_numpy(): {result.sst.to_numpy()}\n')
pcorr = result.sst.to_numpy()
#pcorr = np.transpose(pcorr)

# make contour plot of ninoskill vs lead time and start month
lead_max = pcorr.shape[1]
fig, ax = plt.subplots()
ax.contourf(pcorr, levels=np.arange(0,1.01,0.1), extend="both", cmap="RdBu_r")
ct = ax.contour(pcorr, [0.5, 0.6, 0.7, 0.8, 0.9], colors="k", linewidths=1)
ax.clabel(ct, fontsize=8, colors="k", fmt="%.1f")
ax.set_xlim(0, lead_max - 1)
ax.set_xticks(np.array([1, 3, 6, 9]) - 1) #1,5,10,15,20
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.set_xticklabels(np.array([1,3,6,9]), fontsize=9)
ax.set_xlabel("Prediction lead (months)", fontsize=9)
ax.set_yticks(np.arange(0,12,1))
y_ticklabels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", \
                "Sep", "Oct", "Nov", "Dec"]
ax.set_yticklabels(y_ticklabels, fontsize=9)
ax.set_ylabel("Month", fontsize=9)
plt.show()
