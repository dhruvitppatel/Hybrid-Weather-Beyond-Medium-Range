import numpy as np
import xarray as xr
import pandas as pd
# our local module:
import wavenumber_frequency_functions as wf
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def wf_analysis(x, **kwargs):
    """Return normalized spectra of x using standard processing parameters."""
    # Get the "raw" spectral power
    # OPTIONAL kwargs:
    # segsize, noverlap, spd, latitude_bounds (tuple: (south, north)), dosymmetries, rmvLowFrq

    z2 = wf.spacetime_power(x, **kwargs)
    z2avg = z2.mean(dim='component')
    z2.loc[{'frequency':0}] = np.nan # get rid of spurious power at \nu = 0
    # the background is supposed to be derived from both symmetric & antisymmetric
    background = wf.smooth_wavefreq(z2avg, kern=wf.simple_smooth_kernel(), nsmooth=50, freq_name='frequency')
    # separate components
    z2_sym = z2[0,...]
    z2_asy = z2[1,...]
    # normalize
    nspec_sym = z2_sym / background
    nspec_asy = z2_asy / background
    return nspec_sym, nspec_asy

def plot_normalized_symmetric_spectrum(s, title_pre, ofil=None):
    """Basic plot of normalized symmetric power spectrum with shallow water curves."""
    fb = [0, .8]  # frequency bounds for plot
    # get data for dispersion curves:
    swfreq,swwn = wf.genDispersionCurves()
    # swfreq.shape # -->(6, 3, 50)
    swf = np.where(swfreq == 1e20, np.nan, swfreq)
    swk = np.where(swwn == 1e20, np.nan, swwn)

    fig, ax = plt.subplots()
    c = 'darkgray' # COLOR FOR DISPERSION LINES/LABELS
    z = s.transpose().sel(frequency=slice(*fb), wavenumber=slice(-15,15))
    z.loc[{'frequency':0}] = np.nan
    kmesh0, vmesh0 = np.meshgrid(z['wavenumber'], z['frequency'])
    img = ax.contourf(kmesh0, vmesh0, z, levels=np.linspace(0.2, 3.0, 16), cmap='Spectral_r',  extend='both')
    for ii in range(3,6):
        ax.plot(swk[ii, 0,:], swf[ii,0,:], color=c)
        ax.plot(swk[ii, 1,:], swf[ii,1,:], color=c)
        ax.plot(swk[ii, 2,:], swf[ii,2,:], color=c)
    ax.axvline(0, linestyle='dashed', color='lightgray')
    ax.set_xlim([-15,15])
    ax.set_ylim(fb)
    title = title_pre + 'Normalilzed_Symmetric Component'
    ax.set_title(title)
    fig.colorbar(img)
    if ofil is not None:
        fig.savefig(ofil, bbox_inches='tight', dpi=144)
    else:
        plt.show()

def plot_normalized_asymmetric_spectrum(s, title_pre, ofil=None):
    """Basic plot of normalized symmetric power spectrum with shallow water curves."""

    fb = [0, .8]  # frequency bounds for plot
    # get data for dispersion curves:
    swfreq,swwn = wf.genDispersionCurves()
    # swfreq.shape # -->(6, 3, 50)
    swf = np.where(swfreq == 1e20, np.nan, swfreq)
    swk = np.where(swwn == 1e20, np.nan, swwn)

    fig, ax = plt.subplots()
    c = 'darkgray' # COLOR FOR DISPERSION LINES/LABELS
    z = s.transpose().sel(frequency=slice(*fb), wavenumber=slice(-15,15))
    z.loc[{'frequency':0}] = np.nan
    kmesh0, vmesh0 = np.meshgrid(z['wavenumber'], z['frequency'])
    img = ax.contourf(kmesh0, vmesh0, z, levels=np.linspace(0.2, 1.8, 16), cmap='Spectral_r', extend='both')
    for ii in range(0,3):
        ax.plot(swk[ii, 0,:], swf[ii,0,:], color=c)
        ax.plot(swk[ii, 1,:], swf[ii,1,:], color=c)
        ax.plot(swk[ii, 2,:], swf[ii,2,:], color=c)
    ax.axvline(0, linestyle='dashed', color='lightgray')
    ax.set_xlim([-15,15])
    ax.set_ylim(fb)
    title = title_pre + 'Normalized Anti-symmetric Component'
    ax.set_title(title)
    fig.colorbar(img)
    if ofil is not None:
        fig.savefig(ofil, bbox_inches='tight', dpi=144)
    else:
        plt.show()

def make_ds_time_dim(ds,timestep,startdate):
    begin_year_str = startdate.strftime("%Y-%m-%d")

    attrs = {"units": f"hours since {begin_year_str} "}

    ds = ds.assign_coords({"Timestep": ("Timestep", ds.Timestep.values*timestep, attrs)})
    ds = xr.decode_cf(ds)

    return ds

def get_6hr_precip_era5_timeseries(startdate,enddate,timestep,lat_slice,lon_slice):
    start_year = startdate.year
    end_year = enddate.year

    currentdate = startdate
    counter = 0
    while currentdate.year <= enddate.year:
        print(currentdate.year)
        try:
           ds_era = xr.open_dataset(f'/scratch/user/troyarcomano/ERA_5/{currentdate.year}/era_5_y{currentdate.year}_precip_regridded_mpi_fixed_var_gcc.nc')
        except:
           ds_era = xr.open_dataset(f'/scratch/user/troyarcomano/ERA_5/{currentdate.year}/era_5_y{currentdate.year}_regridded_mpi_fixed_var.nc')

        begin_year = datetime(currentdate.year,1,1,0)
        begin_year_str = begin_year.strftime("%Y-%m-%d")
        attrs = {"units": f"hours since {begin_year_str} "}
        ds_era = ds_era.assign_coords({"Timestep": ("Timestep", ds_era.Timestep.values, attrs)})
        ds_era = xr.decode_cf(ds_era)

        var = ['tp']
        ds_era = ds_era[var].sel(Lat=lat_slice,Lon=lon_slice)

        if start_year == currentdate.year:
           ds_merged = ds_era
        else:
           ds_merged = xr.merge([ds_merged,ds_era])

        currentdate = currentdate + timedelta(hours=ds_era.sizes['Timestep'])

    time_slice = slice(startdate.strftime("%Y-%m-%d"),enddate.strftime("%Y-%m-%d"),timestep)
    return  ds_merged.resample(Timestep = "6H").sum() #ds_merged.sel(Timestep=time_slice)

def get_data(filename, variablename):
    try:
        ds = xr.open_dataset(filename)
    except ValueError:
        ds = xr.open_dataset(filename, decode_times=False)

    return ds #[variablename]

def get_obs_precip_timeseries(startdate,enddate,timestep):
    start_year = startdate.year
    end_year = enddate.year

    currentdate = startdate
    counter = 0
    while currentdate.year <= enddate.year:
        print(currentdate.year)
        try:
           ds_era = xr.open_dataset(f'/scratch/user/troyarcomano/ERA_5/{currentdate.year}/era_5_y{currentdate.year}_precip_regridded_mpi.nc')
        except:
           ds_era = xr.open_dataset(f'/scratch/user/troyarcomano/ERA_5/{currentdate.year}/era_5_y{currentdate.year}_precip_regridded_mpi_fixed_var_gcc.nc')

        begin_year = datetime(currentdate.year,1,1,0)
        begin_year_str = begin_year.strftime("%Y-%m-%d")
        attrs = {"units": f"hours since {begin_year_str} "}
        ds_era = ds_era.assign_coords({"Timestep": ("Timestep", ds_era.Timestep.values, attrs)})
        ds_era = xr.decode_cf(ds_era)

        ds_era = ds_era['tp']

        if start_year == currentdate.year:
           ds_merged = ds_era
        else:
           print(f'ds_era: {ds_era}\n')
           #lon_ind = ds_era.indexes['Lon']
           #ds_era = ds_era.assign_coords(Lon=lon_ind.values)
           ds_merged = xr.merge([ds_merged,ds_era])

        currentdate = currentdate + timedelta(hours=ds_era.sizes['Timestep'])

    time_slice = slice(startdate.strftime("%Y-%m-%d"),enddate.strftime("%Y-%m-%d"),timestep)
    return ds_merged.sel(Timestep=time_slice)

def calculate_plot_wheeler_kiladis(startdates,prediction_length,timestep):
    """Calculate the average Wheeler-Kiladis wavenumber-frequency spectrum across
       forecasts, and plot it."""

    hybrid_root = '/scratch/user/dpp94/Predictions/Hybrid/hybrid_prediction_era6000_20_20_20_sigma0.5_beta_res0.001_beta_model_1.0_prior_0.0_overlap1_vertlevel_1_precip_epsilon0.001_ohtc_multiple_leakage_test_oceantimestep_72hr_train1981_2002_oldcal__pred_newcal_trial_' 
 
    latBound = (-15,15)      # latitude bounds for analysis
    spd = 2                  # samples per day
    nDayWin= 96              # Wheeler-Kiladis [WK] temporal window length (days)
    nDaySkip = -65           # time (days) between temporal windows. negative means there will be overlap between segments.
    twoMonthOverlap = 65
    opt = {'segsize': nDayWin,
           'noverlap': twoMonthOverlap,
           'spd': spd,
           'latitude_bounds': latBound,
           'dosymmetries': True,
           'rmvLowFrq': True}

    hybrid_sym, hybrid_asym = [], []
    obs_sym, obs_asym = [], []
    for startdate in startdates:
        # Read in hybrid and era5 data
        date_str = startdate.strftime("%m_%d_%Y_%H")
        filepath = hybrid_root + date_str + ".nc"
        ds_hybrid = xr.open_dataset(filepath)
        ds_hybrid = make_ds_time_dim(ds_hybrid, timestep, startdate)['p6hr']
        date_i, date_f = startdate, startdate + timedelta(hours=prediction_length) 
        ds_obs = get_obs_precip_timeseries(date_i,date_f,timestep)['tp']

        # Resample timeseries
        ds_hybrid = ds_hybrid.resample(Timestep='12H').sum()
        ds_hybrid = ds_hybrid.rename({'Lat':'lat','Lon':'lon','Timestep':'time'})
        ds_obs = ds_obs.resample(Timestep='12H').sum()
        ds_obs = ds_obs.rename({'Lat':'lat','Lon':'lon','Timestep':'time'})
       
        # Perform wavenumber-frequency analysis
        hybrid_sym_temp, hybrid_asym_temp = wf_analysis(ds_hybrid, **opt) 
        print(f'hybrid_sym_temp: {hybrid_sym_temp}\n')
        obs_sym_temp, obs_asym_temp = wf_analysis(ds_obs, **opt)
        print(f'obs_sym_temp: {obs_sym_temp}\n')
        hybrid_sym.append(hybrid_sym_temp)
        hybrid_asym.append(hybrid_asym_temp)
        obs_sym.append(obs_sym_temp)
        obs_asym.append(obs_asym_temp)

    # Average sym/asym components across all forecasts
    hybrid_sym = xr.concat(hybrid_sym, dim='samples').mean(dim='samples')
    hybrid_asym = xr.concat(hybrid_asym, dim='samples').mean(dim='samples')
    obs_sym = xr.concat(obs_sym, dim='samples').mean(dim='samples')
    obs_asym = xr.concat(obs_asym, dim='samples').mean(dim='samples')

    # Plotting routine
    plot_normalized_symmetric_spectrum(hybrid_sym,'Hybrid: ')
    plot_normalized_asymmetric_spectrum(hybrid_asym,'Hybrid: ')
    plot_normalized_symmetric_spectrum(obs_sym,'ERA5: ')
    plot_normalized_asymmetric_spectrum(obs_asym,'ERA5: ') 


startdates = pd.date_range(start='1/16/2003', end='12/23/2018', freq='30D')
#startdates = pd.date_range(start='1/16/2003', end='2/13/2007', freq='30D')

prediction_length = 8760 * 2
timestep = 6

calculate_plot_wheeler_kiladis(startdates,prediction_length,timestep)
