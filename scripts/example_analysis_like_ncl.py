import numpy as np
import xarray as xr
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


def plot_normalized_symmetric_spectrum(s, ofil=None):
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
    ax.set_title("Normalized Symmetric Component")
    fig.colorbar(img)
    if ofil is not None:
        fig.savefig(ofil, bbox_inches='tight', dpi=144)


def plot_normalized_asymmetric_spectrum(s, ofil=None):
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
    ax.set_title("Normalized Anti-symmetric Component")
    fig.colorbar(img)
    if ofil is not None:
        fig.savefig(ofil, bbox_inches='tight', dpi=144)

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

#
# LOAD DATA, x = DataArray(time, lat, lon), e.g., daily mean precipitation
#
def get_data(filename, variablename):
    try: 
        ds = xr.open_dataset(filename)
    except ValueError:
        ds = xr.open_dataset(filename, decode_times=False)
    
    return ds #[variablename]


if __name__ == "__main__":
    #
    # input file -- could make this a CLI argument
    #
    #fili = '/scratch/user/troyarcomano/Predictions/Hybrid/hybrid_prediction_era6000_20_20_20_sigma0.5_beta_res0.001_beta_model_1.0_prior_0.0_overlap1_vertlevel_1_precip_epsilon0.001_multi_gaussian_noise_newest_version_32_processors_root_ssttrial_12_29_2006_00.nc' #"OLR.12hr_2yrs.wheeler.nc" 
    #fili = '/scratch/user/dpp94/Predictions/Hybrid/hybrid_prediction_era6000_20_20_20_sigma0.5_beta_res0.001_beta_model_1.0_prior_0.0_overlap1_vertlevel_1_precip_epsilon0.001_ohtc_multiple_leakage_test_proper_leaks_70yr_trial_11_16_2007_00.nc'
    fili = '/scratch/user/dpp94/Predictions/Hybrid/hybrid_prediction_era6000_20_20_20_sigma0.5_beta_res0.001_beta_model_1.0_prior_0.0_overlap1_vertlevel_1_precip_epsilon0.001_ohtc_multiple_leakage_test_oceantimestep_72hr_atmo_multileaks_70yr_trial_03_09_2007_00.nc'
    vari = 'p6hr' #"olr"
    #
    # Loading data ... example is very simple
    #
    data = get_data(fili, vari)  # returns OLR

    startdate_hybrid = datetime(2007,1,1,0)
    enddate_hybrid = datetime(2021,1,1,0)

    print(type(data))

    data =  make_ds_time_dim(data,6,startdate_hybrid) 

    data = data.resample(Timestep = "12H").sum()

    data=data.rename({'Lat': 'lat','Lon': 'lon','Timestep':'time'})

    data = data[vari]
    print(data)
    '''
    vari = 'tp'

    lat_slice = slice(-30,30)
    lon_slice = slice(0,365)
    sigma = 7

    startdate = datetime(1981,1,1,0)
    enddate = datetime(2015,1,1,0)

    data = get_6hr_precip_era5_timeseries(startdate,enddate,6,lat_slice,lon_slice)
    data = data.resample(Timestep = "12H").sum()

    data=data.rename({'Lat': 'lat','Lon': 'lon','Timestep':'time'})

    data = data[vari]
   
    vari = 'prec'

    lat_slice = slice(-30,30)
    lon_slice = slice(0,365)
    sigma = 7

    startdate = datetime(1981,1,1,0)
    enddate = datetime(2015,1,1,0)

    data = xr.open_dataset('/scratch/user/troyarcomano/QOCN_DATA/true0iteroutdaily.nc')
    data = data.sel(time=slice(startdate.strftime("%Y-%m-%d"),enddate.strftime("%Y-%m-%d")),lon=lon_slice,lat=lat_slice)

    data = data[vari] 

    '''
    #
    # Options ... right now these only go into wk.spacetime_power()
    #
    latBound = (-15,15)  # latitude bounds for analysis
    spd      = 2 #1    # SAMPLES PER DAY
    nDayWin  = 96   # Wheeler-Kiladis [WK] temporal window length (days)
    nDaySkip = -65  # time (days) between temporal windows [segments]
                    # negative means there will be overlapping temporal segments
    twoMonthOverlap = 65
    opt      = {'segsize': nDayWin, 
                'noverlap': twoMonthOverlap, 
                'spd': spd, 
                'latitude_bounds': latBound, 
                'dosymmetries': True, 
                'rmvLowFrq':True}
    # in this example, the smoothing & normalization will happen and use defaults
    symComponent, asymComponent = wf_analysis(data, **opt)
    #
    # Plots ... sort of matching NCL, but not worrying much about customizing.
    #
    outPlotName = "hybrid_prediction_era6000_20_20_20_sigma0.5_beta_res0.001_beta_model_1.0_prior_0.0_overlap1_vertlevel_1_precip_epsilon0.001_ohtc_multiple_leakage_test_oceantimestep_72hr_atmo_multileaks_symmetric_plot.png"
    plot_normalized_symmetric_spectrum(symComponent, outPlotName)

    outPlotName = "hybrid_prediction_era6000_20_20_20_sigma0.5_beta_res0.001_beta_model_1.0_prior_0.0_overlap1_vertlevel_1_precip_epsilon0.001_ohtc_multiple_leakage_test_oceantimestep_72hr_atmo_multileaks_asymmetric_plot.png"
    plot_normalized_asymmetric_spectrum(asymComponent, outPlotName)
