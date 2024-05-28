import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import xarray as xr
from eofs.xarray import Eof 
from datetime import datetime, timedelta
from scipy.ndimage import uniform_filter1d


def make_ds_time_dim(ds,timestep,startdate):
    begin_year_str = startdate.strftime("%Y-%m-%d")

    attrs = {"units": f"hours since {begin_year_str} "}

    ds = ds.assign_coords({"Timestep": ("Timestep", ds.Timestep.values*timestep, attrs)})
    ds = xr.decode_cf(ds)

    return ds

def get_obs_sst_timeseries(startdate,enddate,timestep):
    start_year = startdate.year
    end_year = enddate.year

    currentdate = startdate
    counter = 0
    while currentdate.year <= enddate.year:
        print('year',currentdate.year)
        ds_era = xr.open_dataset(f'/scratch/user/troyarcomano/ERA_5/{currentdate.year}/era_5_y{currentdate.year}_sst_regridded_fixed_var_gcc.nc') 

        #begin_year = datetime(currentdate.year,1,1,0)
        #begin_year_str = begin_year.strftime("%Y-%m-%d")
        #attrs = {"units": f"hours since {begin_year_str} "}
        #ds_era = ds_era.assign_coords({"Timestep": ("Timestep", ds_era.time.values, attrs)})
        #ds_era = xr.decode_cf(ds_era)

        if start_year == currentdate.year:
           ds_merged = ds_era
        else:    
           ds_merged = xr.merge([ds_merged,ds_era]) 

        currentdate = currentdate + timedelta(hours=ds_era.sizes['time'])

    time_slice = slice(startdate.strftime("%Y-%m-%d"),enddate.strftime("%Y-%m-%d"),timestep)
    return ds_merged.sel(time=time_slice)

def get_anom_specified_climo_hybrid(ds_era,ds_hybrid):

    try:
       monthly_era = ds_era.resample(Timestep='1MS').mean(dim='Timestep')
       monthly_hybrid = ds_hybrid.resample(Timestep='1MS').mean(dim='Timestep')

       climatology = monthly_era.groupby('Timestep.month').mean('Timestep')
       monthly_hybrid_anoms = monthly_hybrid.groupby('Timestep.month') - climatology
    except: 
       monthly_era = ds_era.resample(time='1MS').mean(dim='time')
       monthly_hybrid = ds_hybrid.resample(time='1MS').mean(dim='time')

       climatology = monthly_era.groupby('time.month').mean('time')
       monthly_hybrid_anoms = monthly_hybrid.groupby('time.month') - climatology

    return monthly_hybrid_anoms

def get_pdo_index(ds, ds_climo):
    """Get PDO index given xarray DataArrays."""
    
    lat_slice = slice(20,70)
    lon_slice = slice(360-250,360-100)

    # Remove global mean, get anomalies, crop to North Pacific
    global_mean = ds_climo.mean().values
    ds_climo = ds_climo - global_mean
    ds = ds - global_mean
    ds_anom = get_anom_specified_climo_hybrid(ds_climo, ds)
    ds_anom = ds_anom.sel(Lat=lat_slice, Lon=lon_slice)

    # Compute EOF/PC
    # need to rename Timestep->time to work with eofs package
    ds_anom = ds_anom.rename({'Timestep':'time'})
    solver = Eof(ds_anom)
    pcs = solver.pcs(npcs=1)
    print(f'pcs: {pcs}\n')

    return pcs


date_i, date_f = datetime(1958,1,1,0), datetime(2002,12,1,0)
ds_climo = get_obs_sst_timeseries(date_i,date_f,6)['sst']
ds_climo = ds_climo.rename({'time':'Timestep','lat':'Lat','lon':'Lon'})

date = datetime(2003,1,16,0)
hybrid_root = "/scratch/user/dpp94/Predictions/Hybrid/hybrid_prediction_era6000_20_20_20_sigma0.5_beta_res0.001_beta_model_1.0_prior_0.0_overlap1_vertlevel_1_precip_epsilon0.001_ohtc_multiple_leakage_test_oceantimestep_72hr_train1981_2002_oldcal__pred_newcal_trial_"
date_str = date.strftime("%m_%d_%Y_%H")
filepath = hybrid_root + date_str + ".nc"

ds = xr.open_dataset(filepath)
ds = make_ds_time_dim(ds, 6, date)['SST']

ds_climo['Lat'] = ds['Lat'].values

print(f'ds: {ds}\n')
print(f'ds_climo: {ds_climo}\n')

pcs = get_pdo_index(ds_climo, ds_climo)
time = pcs['time']
pc = pcs.sel(mode=0).values
pc = (pc - pc.mean()) / pc.std()
pc = uniform_filter1d(pc, size=12) #, origin=1)

fig, ax = plt.subplots()
ax.plot(time,pc)
plt.show()
