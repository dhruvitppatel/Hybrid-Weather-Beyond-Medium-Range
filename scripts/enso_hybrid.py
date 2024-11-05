import numpy as np
import numpy.ma as ma
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm
from matplotlib.ticker import NullFormatter, MultipleLocator
from netCDF4 import Dataset
import cartopy as cart
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.geoaxes import GeoAxes
from collections import deque
import xarray as xr
import glob
from datetime import date as datetimedate
from datetime import datetime, timedelta
from numba import jit
import calendar
from mpl_toolkits.axes_grid1 import AxesGrid
#import fiona
import cartopy.io.shapereader as shpreader
import shapely.geometry as sgeom
from shapely.prepared import prep
from scipy.ndimage.filters import uniform_filter1d
from scipy.signal import find_peaks, welch, correlate#, correlation_lags, detrend
from scipy.stats import linregress, pearsonr#, PermutationMethod
import seaborn as sns
import pandas as pd
from xskillscore import pearson_r_p_value, pearson_r_eff_p_value
import cftime
from dateutil.relativedelta import relativedelta
 
import pycwt as wavelet
from pycwt.helpers import find

#from climpred import HindcastEnsemble
#import climpred

#geoms = fiona.open(shpreader.natural_earth(resolution='10m', category='physical', name='land'))
#land_geom = sgeom.MultiPolygon([sgeom.shape(geom['geometry']) for geom in geoms])
#land = prep(land_geom)

#def is_land(lon,lat#):
#    return land.contains(sgeom.Point(lon, lat)) 

def get_e3sm_diff_colormap_greyer():
    e3sm_colors_grayer = [
        '#1b1c40',
        '#2a41a1',
        '#237cbc',
        '#75a9be',
        '#cad3d9',
        '#f1eceb',
        '#e5ccc4',
        '#d08b73',
        '#b9482d',
        '#860e27',
        '#3d0712'
    ]

    cmap = mpl.colors.ListedColormap(e3sm_colors_grayer)
    return cmap

def get_e3sm_diff_colormap():
    file_cm = 'e3sm.rgb'
    rgb_arr = np.loadtxt(file_cm)
    rgb_arr = rgb_arr / 255.0

    colormap = 'test_cmap'
    cmap = LinearSegmentedColormap.from_list(name=colormap, colors=rgb_arr)
    return cmap

@jit()
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

@jit()
def rms(true,prediction):
    return np.sqrt(np.nanmean((prediction-true)**2))

def latituded_weighted_rmse(true,prediction,lats):
    diff = prediction-true
    weights = np.cos(np.deg2rad(lats))
    weights2d = np.zeros(np.shape(diff))
    diff_squared = diff**2.0
    #weights = np.ones((10,96))
    weights2d = np.tile(weights,(96,1))
    weights2d = np.transpose(weights2d)
    masked = np.ma.MaskedArray(diff_squared, mask=np.isnan(diff_squared))
    weighted_average = np.ma.average(masked,weights=weights2d)
    return np.sqrt(weighted_average)

def latitude_weighted_average(data,lats):
    weights = np.cos(np.deg2rad(lats)).reshape((-1,1))
    weights2d = np.tile(weights,(1,96)) #(96,1))
    #weights2d = np.transpose(weights2d)
    masked = np.ma.MaskedArray(data, mask=np.isnan(data))
    weighted_avg = np.ma.average(masked,weights=weights2d)
    return weighted_avg

@jit()
def anomly_cc(forecast,observed,climo):
    top = np.mean((forecast-climo)*(observed-climo))
    bottom = np.sqrt(np.mean((observed-climo)**2)*np.mean((observed-climo)**2))
    ACC = top/bottom
    return ACC

def climo_sst(start_year,end_year,lon_slice,region_slice):
    root_path = '/scratch/user/troyarcomano/ERA/'
    hours_in_year = 24*365
    xgrid = 96
    ygrid = 48
    average_sst = np.zeros((hours_in_year,ygrid,xgrid)) 

    for current_year in range(start_year,end_year + 1):
        ds_era = xr.open_dataset(f'/scratch/user/troyarcomano/ERA_5/{current_year}/era_5_y{current_year}_sst_regridded_fixed_var_gcc.nc')

        if current_year == start_year:
           shape = np.shape(ds_era['sst'].sel(lon=lon_slice,lat=region_slice))
           xgrid = shape[2]
           ygrid = shape[1]
           average_sst = np.zeros((hours_in_year,ygrid,xgrid)) 

        if calendar.isleap(current_year):
           startdate = datetime(current_year,1,1,0)
           enddate = datetime(current_year,2,29,0)
           time_slice = slice(startdate.strftime("%Y-%m-%d"),enddate.strftime("%Y-%m-%d"))
           average_sst[0:240*6,:,:] += ds_era['sst'].sel(time=time_slice,lon=lon_slice,lat=region_slice).values

           startdate = datetime(current_year,3,2,0)
           enddate = datetime(current_year,12,31,23)
           time_slice = slice(startdate.strftime("%Y-%m-%d"),enddate.strftime("%Y-%m-%d"))
           average_sst[240*6:1460*6,:,:] += ds_era['sst'].sel(time=time_slice,lon=lon_slice,lat=region_slice).values
        else:
           startdate = datetime(current_year,1,1,0)
           enddate = datetime(current_year,12,31,23)
           time_slice = slice(startdate.strftime("%Y-%m-%d"),enddate.strftime("%Y-%m-%d"))
           average_sst += ds_era['sst'].sel(time=time_slice,lon=lon_slice,lat=region_slice).values

    return average_sst/(end_year-start_year+1)

def climo_atmo_var(start_year,end_year,lon_slice,region_slice,varname):
    root_path = '/scratch/user/troyarcomano/ERA/'
    hours_in_year = 24*365
    xgrid = 96
    ygrid = 48
    average_var = np.zeros((hours_in_year,ygrid,xgrid))

    for current_year in range(start_year,end_year + 1):
        ds_era = xr.open_dataset(f'/scratch/user/troyarcomano/ERA_5/{current_year}/era_5_y{current_year}_regridded_mpi_fixed_var_gcc.nc')

        if current_year == start_year:
           shape = np.shape(ds_era[varname].sel(Lon=lon_slice,Lat=region_slice))
           xgrid = shape[2]
           ygrid = shape[1]
           average_var = np.zeros((hours_in_year,ygrid,xgrid))

        if calendar.isleap(current_year):
           time_slice = slice(0,240*6)
           average_var[0:240*6,:,:] += ds_era[varname].sel(Timestep=time_slice,Lon=lon_slice,Lat=region_slice)

           time_slice = slice(244*6,1464*6)
           average_var[240*6:1460*6,:,:] += ds_era[varname].sel(Timestep=time_slice,Lon=lon_slice,Lat=region_slice)
        else:
           time_slice = slice(0,hours_in_year)
           average_var += ds_era[varname].sel(Timestep=time_slice,Lon=lon_slice,Lat=region_slice)

    return average_var/(end_year-start_year+1)

def nino_index(sst_data,region):
    if region == "1+2":
       lat_slice = slice(-10,0)
       lon_slice = slice(360-90,360-80)
    elif region == "3":
       lat_slice = slice(-5,5)
       lon_slice = slice(360-150,360-90)
    elif region == "3.4" :
       lat_slice = slice(-5,5)
       lon_slice = slice(360-170,360-120)
    elif region == "4" :
       lat_slice = slice(-5,5)
       lon_slice = slice(360-200,360-150)
  
    start_year = 1981
    end_year = 2010

    try: 
      climate_data = climo_sst(start_year,end_year,lon_slice,lat_slice)
    
      sst_data = sst_data.sel(Lon=lon_slice,Lat=lat_slice)
     
      sst_anom = np.zeros((sst_data.sizes['Timestep'],sst_data.sizes['Lat'],sst_data.sizes['Lon'] ))
    except: 
      climate_data = climo_sst(start_year,end_year,lon_slice,lat_slice)

      sst_data = sst_data.sel(lon=lon_slice,lat=lat_slice)

      sst_anom = np.zeros((sst_data.sizes['time'],sst_data.sizes['lat'],sst_data.sizes['lon']))

    size = np.shape(climate_data)[0]
    print(np.shape(climate_data))
    print(size)
    try:
       for i in range(sst_data.sizes['Timestep']-1): 
           sst_anom[i,:,:] = sst_data[i,:,:] - climate_data[(i*6)%size,:,:]
    except:
       for i in range(sst_data.sizes['time']-1):
           sst_anom[i,:,:] = sst_data[i,:,:] - climate_data[(i*6)%size,:,:]
 
    return np.mean(sst_anom,axis=(1,2))

def nino_index_monthly(sst_data,region):
    if region == "1+2":
       lat_slice = slice(-10,0)
       lon_slice = slice(360-90,360-80)
    elif region == "3":
       lat_slice = slice(-5,5)
       lon_slice = slice(360-150,360-90)
    elif region == "3.4" :
       lat_slice = slice(-5,5)
       lon_slice = slice(360-170,360-120)
    elif region == "4" :
       lat_slice = slice(-5,5)
       lon_slice = slice(360-200,360-150)

    start_year = 1981
    end_year = 2010
   
    try:
       sst_data_copy = sst_data.sel(Lon=lon_slice,Lat=lat_slice)

       sst_data_copy = sst_data_copy.resample(Timestep="1MS").mean(dim="Timestep")
       gb = sst_data_copy.groupby('Timestep.month')
       tos_nino34_anom = gb - gb.mean(dim='Timestep')
       index_nino34 = tos_nino34_anom.mean(dim=['Lat', 'Lon'])
    except:
       sst_data_copy = sst_data.sel(lon=lon_slice,lat=lat_slice)
       sst_data_copy = sst_data_copy.resample(time="1MS").mean(dim="time")
       gb = sst_data_copy.groupby('time.month')
       tos_nino34_anom = gb - gb.mean(dim='time')
       index_nino34 = tos_nino34_anom.mean(dim=['lat', 'lon'])
    return index_nino34 

def nino_index_speedy(sst_data,startdate,enddate,region):
    if region == "1+2":
       lat_slice = slice(-10,0)
       lon_slice = slice(360-90,360-80)
    elif region == "3":
       lat_slice = slice(-5,5)
       lon_slice = slice(360-150,360-90)
    elif region == "3.4" :
       lat_slice = slice(-5,5)
       lon_slice = slice(360-170,360-120)
    elif region == "4" :
       lat_slice = slice(-5,5)
       lon_slice = slice(360-200,360-150)

    climate_start_year = 1981
    climate_end_year = 2000

    climate_data = climo_sst(climate_start_year,climate_end_year,lon_slice,lat_slice)

    sst_data = sst_data.sel(time=slice(startdate,enddate),lon=lon_slice,lat=lat_slice)
    print(sst_data)

    sst_anom = np.zeros((sst_data.sizes['time'],sst_data.sizes['lat'],sst_data.sizes['lon'] ))

    size = np.shape(climate_data)[0]
    for i in range(sst_data.sizes['time']-1):
        sst_anom[i,:,:] = sst_data[i,:,:].values - climate_data[(i*730)%size,:,:]

    return np.mean(sst_anom,axis=(1,2))

def get_obs_atmo_timeseries(startdate,enddate,timestep):
    start_year = startdate.year
    end_year = enddate.year

    currentdate = startdate
    counter = 0
    while currentdate.year <= enddate.year:
        print(f'year: {currentdate.year}\n')
        ds_era = xr.open_dataset(f'/scratch/user/troyarcomano/ERA_5/{currentdate.year}/era_5_y{currentdate.year}_regridded_mpi_fixed_var_gcc.nc')

        begin_year = datetime(currentdate.year,1,1,0)
        begin_year_str = begin_year.strftime("%Y-%m-%d")
        attrs = {"units": f"hours since {begin_year_str} "}
        ds_era = ds_era.assign_coords({"Timestep": ("Timestep", ds_era.Timestep.values, attrs)})
        ds_era = xr.decode_cf(ds_era)

        if start_year == currentdate.year:
           ds_merged = ds_era
        else:
           ds_merged = xr.merge([ds_merged,ds_era])

        currentdate = currentdate + timedelta(hours=ds_era.sizes['Timestep'])

    time_slice = slice(startdate.strftime("%Y-%m-%d"),enddate.strftime("%Y-%m-%d"),timestep)
    return ds_merged.sel(Timestep=time_slice)

def get_obs_atmo_timeseries_var(startdate,enddate,timestep,var,sigma_lvl=None):
    start_year = startdate.year
    end_year = enddate.year

    currentdate = startdate
    counter = 0
    while currentdate.year <= enddate.year:
        print(currentdate.year)
        try:
           ds_era = xr.open_dataset(f'/scratch/user/troyarcomano/ERA_5/{currentdate.year}/era_5_y{currentdate.year}_regridded_mpi.nc')
        except: 
           ds_era = xr.open_dataset(f'/scratch/user/troyarcomano/ERA_5/{currentdate.year}/era_5_y{currentdate.year}_regridded_mpi_fixed_var_gcc.nc')

        begin_year = datetime(currentdate.year,1,1,0)
        begin_year_str = begin_year.strftime("%Y-%m-%d")
        attrs = {"units": f"hours since {begin_year_str} "}
        ds_era = ds_era.assign_coords({"Timestep": ("Timestep", ds_era.Timestep.values, attrs)})
        ds_era = xr.decode_cf(ds_era)

        if sigma_lvl:
           ds_era = ds_era[var].sel(Sigma_Level=sigma_lvl)
        else:
           ds_era = ds_era[var]

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

def southern_oscillation_index(ds_hybrid,ds_era,startdate,enddate):
    darwin_lat, darwin_long = -12.4637, 130.8444
    tahiti_lat, tahiti_long = -17.6509, 210.426

    
    lon_slice = slice(darwin_long - 10, tahiti_long + 10)
    lat_slice = slice(tahiti_lat - 5, darwin_lat + 5)

    climate_start = datetime(1981,1,1,0)
    climate_end = datetime(2006,12,31,23)

    climo_time_slice = slice(startdate.strftime("%Y-%m-%d"),enddate.strftime("%Y-%m-%d"))
   
    data_tahiti = ds_era.sel(Lat = tahiti_lat, Lon = tahiti_long, method = 'nearest')['logp']
    data_darwin = ds_era.sel(Lat = darwin_lat, Lon = darwin_long, method = 'nearest')['logp']
   
    #data_tahiti = data_tahiti.resample(Timestep="1MS").mean("Timestep")
    #data_darwin = data_darwin.resample(Timestep="1MS").std("Timestep")

    data_tahiti = np.exp(data_tahiti)*1000.0
    data_darwin = np.exp(data_darwin)*1000.0

    diff_data = data_tahiti - data_darwin
   
    diff_climo = diff_data.sel(Timestep=climo_time_slice)

    climatology_mean = diff_climo.groupby("Timestep.month").mean("Timestep")
    climatology_std  = diff_climo.groupby("Timestep.month").std("Timestep")

    print('climatology_std',climatology_std)
    print('climatology_mean',climatology_mean)

    print('ds_hybrid',ds_hybrid)
    hybrid_data_tahiti = ds_hybrid.sel(Lat = tahiti_lat, Lon = tahiti_long, method = 'nearest')['logp']
    hybrid_data_darwin = ds_hybrid.sel(Lat = darwin_lat, Lon = darwin_long, method = 'nearest')['logp']

    #hybrid_data_tahiti = hybrid_data_tahiti.resample(Timestep="1MS")
    #hybrid_data_darwin = hybrid_data_darwin.resample(Timestep="1MS")

    hybrid_data_tahiti = np.exp(hybrid_data_tahiti)*1000.0
    hybrid_data_darwin = np.exp(hybrid_data_darwin)*1000.0

    hybrid_diff_data = hybrid_data_tahiti - hybrid_data_darwin

    hybrid_soi = xr.apply_ufunc(
        lambda x, m, s: (x - m) / s,
        hybrid_diff_data.groupby("Timestep.month"),
        climatology_mean,
        climatology_std,
    )

    print('hybrid_soi',hybrid_soi)
    hybrid_soi = hybrid_soi * 10.0
    #hybrid_soi = 10 * (hybrid_diff_data.groupby("Timestep.month") - climatology_mean)/climatology_std
  
    print('np.shape(hybrid_soi)',np.shape(hybrid_soi)) 

    era_soi = xr.apply_ufunc(
        lambda x, m, s: (x - m) / s,
        diff_climo.groupby("Timestep.month"),
        climatology_mean,
        climatology_std,
    )
    era_soi = era_soi * 10.0
    return hybrid_soi

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

def get_obs_ohtc_timeseries(startdate,enddate,timestep):
    start_year = startdate.year
    end_year = enddate.year

    time_slice = slice(startdate.strftime("%Y-%m-%d"),enddate.strftime("%Y-%m-%d"),timestep)
    ds_era = xr.open_dataset(f'/scratch/user/troyarcomano/ORAS5/regridded_sohtc300_control_hourly_full_data_gcc.nc')

    return ds_era.sel(time_counter=time_slice)

def get_obs_mld_timeseries(startdate,enddate,timestep):
    start_year = startdate.year
    end_year = enddate.year

    time_slice = slice(startdate.strftime("%Y-%m-%d"),enddate.strftime("%Y-%m-%d"),timestep)
    ds_era = xr.open_dataset(f'/scratch/user/troyarcomano/ORAS5/regridded_somxl010_control_monthly_highres_2D_CONS_v0.1_hourly_invert.nc')

    return ds_era.sel(time_counter=time_slice)

def sst_annual_variability(ds_model,ds_era,ds_speedy):
    lat_slice = slice(-90,90)
    lon_slice = slice(0,360)

    startdate_era = datetime(1981,1,1,0)
    enddate_era = datetime(2020,12,31,23)

    startdate_hybrid = datetime(2007,1,1,0)
    enddate_hybrid = datetime(2047,12,31,23)

    startdate_speedy = datetime(1980,1,1,0)
    enddate_speedy = datetime(2008,12,31,23)

    ds_model = ds_model.sel(Lon=lon_slice,Lat=lat_slice,Timestep=slice(startdate_hybrid.strftime("%Y-%m-%d"),enddate_hybrid.strftime("%Y-%m-%d")))['SST']
    ds_era = ds_era.sel(lon=lon_slice,lat=lat_slice,time=slice(startdate_era.strftime("%Y-%m-%d"),enddate_era.strftime("%Y-%m-%d")))['sst']
    ds_speedy = ds_speedy.sel(lon=lon_slice,lat=lat_slice,time=slice(startdate_speedy.strftime("%Y-%m-%d"),enddate_speedy.strftime("%Y-%m-%d")))['sst']
  
    ds_model = ds_model.groupby("Timestep.month").std("Timestep")
    ds_era = ds_era.groupby("time.month").std("time")
    ds_speedy = ds_speedy.groupby("time.month").std("time")

    lons = ds_model.Lon.values 
    lats = ds_model.Lat.values
    print('lons',lons)

    mean_hybrid = np.mean(ds_model.values,axis=(0))
    mean_climo = np.mean(ds_era.values,axis=(0))
    mean_speedy = np.mean(ds_speedy.values,axis=(0))

    projection = ccrs.PlateCarree(central_longitude=-179)
    axes_class = (GeoAxes,dict(map_projection=projection))
    plt.rc('font', family='serif')
    plt.rcParams['figure.constrained_layout.use'] = True


    cmap_e3sm_diff_colormap_greyer = get_e3sm_diff_colormap_greyer()

    fig = plt.figure(figsize=(6,10))
    axgr = AxesGrid(fig, 111, axes_class=axes_class,
                    nrows_ncols=(3, 1),
                    axes_pad=0.7,
                    cbar_location='right',
                    cbar_mode='each',
                    cbar_pad=0.2,
                    cbar_size='3%',
                    label_mode='')  # note the empty label_mode


    cyclic_data, cyclic_lons = add_cyclic_point(mean_hybrid, coord=lons)
    lons2d,lats2d = np.meshgrid(cyclic_lons,lats)

    ax1 = axgr[0]#plt.subplot(221,projection=ccrs.PlateCarree())
    ax1.coastlines()

    ax1.set_xticks([-180,-120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree())
    ax1.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax1.xaxis.set_major_formatter(lon_formatter)
    ax1.yaxis.set_major_formatter(lat_formatter)
    gl = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                       linewidth=2, color='gray', alpha=0.5, linestyle='--')

    levels = [0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6]
    temp_colormap = sns.color_palette("Spectral", as_cmap=True)
    temp_colormap = temp_colormap.reversed()
    cmap = temp_colormap
    plot = ax1.contourf(lons2d,lats2d,cyclic_data,transform=ccrs.PlateCarree(),levels = levels,cmap=cmap,extend='both')#'seismic',extend='both')

    cbar = axgr.cbar_axes[0].colorbar(plot, extend='both')
    cbar.set_ticks(levels)
    cbar.set_label(r'$\degree C$',fontsize=16)
    cbar.ax.tick_params(labelsize=16)
    ax1.add_feature(cart.feature.LAND, zorder=100, edgecolor='k',facecolor='#808080')

    ax1.set_title(f"Coupled Model SST Standard Deviation of Monthly Means",fontsize=18,fontweight="bold")
    ax1.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())

    cyclic_data, cyclic_lons = add_cyclic_point(mean_climo, coord=lons)
    lons2d,lats2d = np.meshgrid(cyclic_lons,lats)
    ax2 = axgr[1]#plt.subplot(221,projection=ccrs.PlateCarree())
    ax2.coastlines()

    ax2.set_xticks([-180,-120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree())
    ax2.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax2.xaxis.set_major_formatter(lon_formatter)
    ax2.yaxis.set_major_formatter(lat_formatter)
    gl = ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                        linewidth=2, color='gray', alpha=0.5, linestyle='--')

    levels = [0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6]
    plot = ax2.contourf(lons2d,lats2d,cyclic_data,transform=ccrs.PlateCarree(),levels = levels,cmap=cmap,extend='both')#'seismic',extend='both')

    cbar = axgr.cbar_axes[1].colorbar(plot, extend='both')
    cbar.set_ticks(levels)
    cbar.set_label(r'$\degree C$',fontsize=16)
    cbar.ax.tick_params(labelsize=16)
    ax2.add_feature(cart.feature.LAND, zorder=100, edgecolor='k',facecolor='#808080')

    ax2.set_title(f"ERA5 SST Standard Deviation",fontsize=18,fontweight="bold")
    ax2.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())

    #### Bias
    cyclic_data, cyclic_lons = add_cyclic_point(mean_hybrid - mean_climo, coord=lons)
    lons2d,lats2d = np.meshgrid(cyclic_lons,lats)
    ax3 = axgr[2]#plt.subplot(221,projection=ccrs.PlateCarree())
    ax3.coastlines()
   
    ax3.set_xticks([-180,-120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree())
    ax3.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax3.xaxis.set_major_formatter(lon_formatter)
    ax3.yaxis.set_major_formatter(lat_formatter)
    gl = ax3.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                       linewidth=2, color='gray', alpha=0.5, linestyle='--')

    levels = [-0.6,-0.4,-0.2,-0.05,0.05,0.2,0.4,0.6]

    plot = ax3.contourf(lons2d,lats2d,cyclic_data,transform=ccrs.PlateCarree(),levels = levels,cmap=cmap_e3sm_diff_colormap_greyer,extend='both')#'seismic',extend='both')

    cbar = axgr.cbar_axes[2].colorbar(plot, extend='both')
    cbar.set_ticks(levels)
    cbar.set_label(r'$\degree C$',fontsize=16)
    cbar.ax.tick_params(labelsize=16)
    ax3.add_feature(cart.feature.LAND, zorder=100, edgecolor='k',facecolor='#808080')

    ax3.set_title(f"Standard Deviation Bias (Model - ERA5)",fontsize=18,fontweight="bold")
    ax3.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())

    plt.show()

def sst_climatology_error(ds_model,ds_era,ds_speedy):  
    lat_slice = slice(-90,90)
    lon_slice = slice(0,360) 

    startdate_era = datetime(1981,1,1,0)
    enddate_era = datetime(2020,12,31,23) 
    
    startdate_hybrid = datetime(2007,1,1,0)
    enddate_hybrid = datetime(2047,12,31,23)

    startdate_speedy = datetime(1980,1,1,0)
    enddate_speedy = datetime(2008,12,31,23)
   
    mean_hybrid = ds_model.sel(Lon=lon_slice,Lat=lat_slice,Timestep=slice(startdate_hybrid.strftime("%Y-%m-%d"),enddate_hybrid.strftime("%Y-%m-%d"))).mean("Timestep")['SST']
    mean_climo = ds_era.sel(lon=lon_slice,lat=lat_slice,time=slice(startdate_era.strftime("%Y-%m-%d"),enddate_era.strftime("%Y-%m-%d"))).mean("time")['sst']
    mean_speedy = ds_speedy.sel(lon=lon_slice,lat=lat_slice,time=slice(startdate_speedy.strftime("%Y-%m-%d"),enddate_speedy.strftime("%Y-%m-%d"))).mean("time")['sst']

    lons = mean_hybrid.Lon.values
    lats = mean_hybrid.Lat.values

    mean_hybrid = mean_hybrid.where(mean_hybrid > 272.0)

    mean_speedy = mean_speedy.where(mean_speedy > 272.0)

    mean_climo = mean_climo.where(mean_climo > 272.0)

    mean_hybrid = mean_hybrid.values
    mean_climo = mean_climo.values
    mean_speedy = mean_speedy.values

    print('mean_climo')
    print(mean_climo) 
    print(mean_hybrid)
    print(mean_speedy)
    #mean_hybrid = mean_hybrid.where(mean_hybrid > 273.0)
    #mean_speedy = mean_speedy.where(mean_speedy > 273.0)


    print('lons',lons) 
    diff_max = 5

    cmap = get_e3sm_diff_colormap()
 
    cmap_e3sm_diff_colormap_greyer = get_e3sm_diff_colormap_greyer()

    cyclic_data, cyclic_lons = add_cyclic_point(mean_hybrid - mean_climo, coord=lons)
    lons2d,lats2d = np.meshgrid(cyclic_lons,lats)
    projection = ccrs.PlateCarree(central_longitude=-179)
    axes_class = (GeoAxes,dict(map_projection=projection))
    plt.rc('font', family='serif')

    fig = plt.figure()
    axgr = AxesGrid(fig, 111, axes_class=axes_class,
                    nrows_ncols=(2, 1),
                    axes_pad=0.5,
                    cbar_location='right',
                    cbar_mode='each',
                    cbar_pad=0.2,
                    cbar_size='3%',
                    label_mode='')  # note the empty label_mode


    ax1 = axgr[0]#plt.subplot(221,projection=ccrs.PlateCarree())
    ax1.coastlines()

    ax1.set_xticks([-180,-120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree())
    ax1.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax1.xaxis.set_major_formatter(lon_formatter)
    ax1.yaxis.set_major_formatter(lat_formatter) 
    gl = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                       linewidth=2, color='gray', alpha=0.5, linestyle='--')

    levels = [-5,-3,-2,-1,-0.1,0.1,1,2,3,5]
    plot = ax1.contourf(lons2d,lats2d,cyclic_data,transform=ccrs.PlateCarree(),levels = levels,cmap=cmap,extend='both')#'seismic',extend='both')

    cbar = axgr.cbar_axes[0].colorbar(plot, extend='both')
    cbar.set_ticks(levels)
    cbar.set_label(r'$\degree C$',fontsize=16)
    cbar.ax.tick_params(labelsize=16)
    ax1.add_feature(cart.feature.LAND, zorder=100, edgecolor='k',facecolor='#808080')

    ax1.set_title(f"Hybrid Sea Surface Temp. 40 yr. Avg. Bias",fontsize=18,fontweight="bold")
    ax1.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())

    print('Hybrid Sea Surface Temperature Bias',np.nanmin(cyclic_data),np.nanmax(cyclic_data),np.nanmean(cyclic_data))
    rms_hybrid = rms(mean_climo,mean_hybrid)
    print('Hybrid SST RMS',np.nanmin(rms_hybrid),np.nanmax(rms_hybrid),np.nanmean(rms_hybrid))
    ###########

    diff_max = 5
 
    cyclic_data, cyclic_lons = add_cyclic_point(mean_speedy - mean_climo, coord=lons)
    lons2d,lats2d = np.meshgrid(cyclic_lons,lats)

    print(np.shape(cyclic_data))
    print(np.shape(lons2d))

    ax2 = axgr[1] #plt.subplot(222,projection=ccrs.PlateCarree())
    ax2.coastlines()

    ax2.set_xticks([-180,-120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree())
    ax2.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax2.xaxis.set_major_formatter(lon_formatter)
    ax2.yaxis.set_major_formatter(lat_formatter)
    gl = ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                       linewidth=2, color='gray', alpha=0.5, linestyle='--')

    levels = [-5,-3,-2,-1,-0.1,0.1,1,2,3,5]
    plot2 = ax2.contourf(lons2d,lats2d,cyclic_data,transform=ccrs.PlateCarree(),levels=levels,cmap=cmap,extend='both')#'seismic',extend='both')

    cbar = axgr.cbar_axes[1].colorbar(plot2, extend='both')
    cbar.set_ticks(levels)
    cbar.set_label(r'$\degree C$',fontsize=16)
    cbar.ax.tick_params(labelsize=16)
    ax2.add_feature(cart.feature.LAND, zorder=100, edgecolor='k',facecolor='#808080')

    ax2.set_title(f"SPEEDY Slab Ocean SST 40 yr. Avg. Bias",fontsize=18,fontweight="bold")
    ax2.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())

    print('SPEEDY Sea Surface Temperature Bias',np.nanmin(cyclic_data),np.nanmax(cyclic_data),np.nanmean(cyclic_data))
    rms_hybrid = rms(mean_climo,mean_speedy)
    print('SPEEDY SST RMS',np.nanmin(rms_hybrid),np.nanmax(rms_hybrid),np.nanmean(rms_hybrid))

    plt.show()

    fig = plt.figure(figsize=(6,10))
    axgr = AxesGrid(fig, 111, axes_class=axes_class,
                    nrows_ncols=(3, 1),
                    axes_pad=0.7,
                    cbar_location='right',
                    cbar_mode='each',
                    cbar_pad=0.2,
                    cbar_size='3%',
                    label_mode='')  # note the empty label_mode


    cyclic_data, cyclic_lons = add_cyclic_point(mean_hybrid - 273.15, coord=lons)
    lons2d,lats2d = np.meshgrid(cyclic_lons,lats)

    ax1 = axgr[0]#plt.subplot(221,projection=ccrs.PlateCarree())
    ax1.coastlines()

    ax1.set_xticks([-180,-120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree())
    ax1.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax1.xaxis.set_major_formatter(lon_formatter)
    ax1.yaxis.set_major_formatter(lat_formatter)
    gl = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                       linewidth=2, color='gray', alpha=0.5, linestyle='--')

    levels = np.arange(0,33,4)
    temp_colormap = sns.color_palette("Spectral", as_cmap=True)
    temp_colormap = temp_colormap.reversed() 
    cmap = temp_colormap
    plot = ax1.contourf(lons2d,lats2d,cyclic_data,transform=ccrs.PlateCarree(),levels = levels,cmap=cmap,extend='both')#'seismic',extend='both')

    cbar = axgr.cbar_axes[0].colorbar(plot, extend='both')
    cbar.set_ticks(levels)
    cbar.set_label(r'$\degree C$',fontsize=16)
    cbar.ax.tick_params(labelsize=16)
    ax1.add_feature(cart.feature.LAND, zorder=100, edgecolor='k',facecolor='#808080')

    ax1.set_title(f"Coupled Model Mean SST",fontsize=18,fontweight="bold")
    ax1.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
  
 
    cyclic_data, cyclic_lons = add_cyclic_point(mean_climo - 273.15, coord=lons)
    lons2d,lats2d = np.meshgrid(cyclic_lons,lats) 
    ax2 = axgr[1]#plt.subplot(221,projection=ccrs.PlateCarree())
    ax2.coastlines()

    ax2.set_xticks([-180,-120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree())
    ax2.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax2.xaxis.set_major_formatter(lon_formatter)
    ax2.yaxis.set_major_formatter(lat_formatter)
    gl = ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                       linewidth=2, color='gray', alpha=0.5, linestyle='--')

    #plot = ax1.pcolormesh(lons2d,lats2d,cyclic_data,transform=ccrs.PlateCarree(),vmin=-1*diff_max,vmax=diff_max,cmap='seismic')
    levels = np.arange(0,33,4)
    plot = ax2.contourf(lons2d,lats2d,cyclic_data,transform=ccrs.PlateCarree(),levels = levels,cmap=cmap,extend='both')#'seismic',extend='both')

    cbar = axgr.cbar_axes[1].colorbar(plot, extend='both')
    cbar.set_ticks(levels)
    cbar.set_label(r'$\degree C$',fontsize=16)
    cbar.ax.tick_params(labelsize=16)
    ax2.add_feature(cart.feature.LAND, zorder=100, edgecolor='k',facecolor='#808080')

    ax2.set_title(f"ERA5 Mean SST",fontsize=18,fontweight="bold")
    ax2.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
 
    #### Bias 
    cyclic_data, cyclic_lons = add_cyclic_point(mean_hybrid - mean_climo, coord=lons)
    lons2d,lats2d = np.meshgrid(cyclic_lons,lats) 
    ax3 = axgr[2]#plt.subplot(221,projection=ccrs.PlateCarree())
    ax3.coastlines()

    ax3.set_xticks([-180,-120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree())
    ax3.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax3.xaxis.set_major_formatter(lon_formatter)
    ax3.yaxis.set_major_formatter(lat_formatter)
    gl = ax3.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                       linewidth=2, color='gray', alpha=0.5, linestyle='--')

    #plot = ax1.pcolormesh(lons2d,lats2d,cyclic_data,transform=ccrs.PlateCarree(),vmin=-1*diff_max,vmax=diff_max,cmap='seismic')
    levels = [-5,-3,-2,-1,-0.1,0.1,1,2,3,5]
    plot = ax3.contourf(lons2d,lats2d,cyclic_data,transform=ccrs.PlateCarree(),levels = levels,cmap=cmap_e3sm_diff_colormap_greyer,extend='both')#'seismic',extend='both')

    cbar = axgr.cbar_axes[2].colorbar(plot, extend='both')
    cbar.set_ticks(levels)
    cbar.set_label(r'$\degree C$',fontsize=16)
    cbar.ax.tick_params(labelsize=16)
    ax3.add_feature(cart.feature.LAND, zorder=100, edgecolor='k',facecolor='#808080')

    ax3.set_title(f"Sea Surface Temp. Bias (Model - ERA5)",fontsize=18,fontweight="bold")
    ax3.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())

    plt.show()

    ###########

def sst_annual_climo_and_var_grl2022_paper(ds_model,ds_era,ds_speedy):
    lat_slice = slice(-90,90)
    lon_slice = slice(0,360)

    startdate_era = datetime(1981,1,1,0)
    enddate_era = datetime(2020,12,31,23)

    startdate_hybrid = datetime(2007,1,1,0)
    enddate_hybrid = datetime(2047,12,31,23)

    startdate_speedy = datetime(1980,1,1,0)
    enddate_speedy = datetime(2008,12,31,23)

    mean_hybrid = ds_model.sel(Lon=lon_slice,Lat=lat_slice,Timestep=slice(startdate_hybrid.strftime("%Y-%m-%d"),enddate_hybrid.strftime("%Y-%m-%d"))).mean("Timestep")['SST']
    mean_climo = ds_era.sel(lon=lon_slice,lat=lat_slice,time=slice(startdate_era.strftime("%Y-%m-%d"),enddate_era.strftime("%Y-%m-%d"))).mean("time")['sst']
    mean_speedy = ds_speedy.sel(lon=lon_slice,lat=lat_slice,time=slice(startdate_speedy.strftime("%Y-%m-%d"),enddate_speedy.strftime("%Y-%m-%d"))).mean("time")['sst']

    lons = mean_hybrid.Lon.values
    lats = mean_hybrid.Lat.values

    mean_hybrid = mean_hybrid.where(mean_hybrid > 272.0)

    mean_speedy = mean_speedy.where(mean_speedy > 272.0)

    mean_climo = mean_climo.where(mean_climo > 272.0)

    mean_hybrid = mean_hybrid.values
    mean_climo = mean_climo.values
    mean_speedy = mean_speedy.values

    ds_model = ds_model.sel(Lon=lon_slice,Lat=lat_slice,Timestep=slice(startdate_hybrid.strftime("%Y-%m-%d"),enddate_hybrid.strftime("%Y-%m-%d")))['SST']
    ds_era = ds_era.sel(lon=lon_slice,lat=lat_slice,time=slice(startdate_era.strftime("%Y-%m-%d"),enddate_era.strftime("%Y-%m-%d")))['sst']
    ds_speedy = ds_speedy.sel(lon=lon_slice,lat=lat_slice,time=slice(startdate_speedy.strftime("%Y-%m-%d"),enddate_speedy.strftime("%Y-%m-%d")))['sst']

    ds_model_std = ds_model.groupby("Timestep.month").std("Timestep")
    ds_era_std = ds_era.groupby("time.month").std("time")
    ds_speedy_std = ds_speedy.groupby("time.month").std("time")

    mean_hybrid_std = np.mean(ds_model_std.values,axis=(0))
    mean_climo_std = np.mean(ds_era_std.values,axis=(0))
    mean_speedy_std = np.mean(ds_speedy_std.values,axis=(0))

    print('mean_climo')
    print(mean_climo)
    print(mean_hybrid)
    print(mean_speedy)
    #mean_hybrid = mean_hybrid.where(mean_hybrid > 273.0)
    #mean_speedy = mean_speedy.where(mean_speedy > 273.0)


    print('lons',lons)
    diff_max = 5

    cmap = get_e3sm_diff_colormap()

    cmap_e3sm_diff_colormap_greyer = get_e3sm_diff_colormap_greyer()

    cyclic_data, cyclic_lons = add_cyclic_point(mean_hybrid - mean_climo, coord=lons)
    lons2d,lats2d = np.meshgrid(cyclic_lons,lats)
    projection = ccrs.PlateCarree(central_longitude=-179)
    axes_class = (GeoAxes,dict(map_projection=projection))
    plt.rc('font', family='serif')
    plt.rcParams['figure.constrained_layout.use'] = True

    fig = plt.figure(figsize=(10,18))
    axgr = AxesGrid(fig, 111, axes_class=axes_class,
                    nrows_ncols=(3, 2),
                    axes_pad=(1.55,0.9),
                    cbar_location='right',
                    cbar_mode='each',
                    cbar_pad=0.2,
                    cbar_size='3%',
                    label_mode='')  # note the empty label_mode


    cyclic_data, cyclic_lons = add_cyclic_point(mean_hybrid - 273.15, coord=lons)
    lons2d,lats2d = np.meshgrid(cyclic_lons,lats)

    ax1 = axgr[2]#plt.subplot(221,projection=ccrs.PlateCarree())
    ax1.coastlines()

    ax1.set_xticks([-180,-120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree())
    ax1.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax1.xaxis.set_major_formatter(lon_formatter)
    ax1.yaxis.set_major_formatter(lat_formatter)
    gl = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                       linewidth=2, color='gray', alpha=0.5, linestyle='--')

    levels = np.arange(0,33,4)
    temp_colormap = sns.color_palette("Spectral", as_cmap=True)
    temp_colormap = temp_colormap.reversed()
    cmap = temp_colormap
    plot = ax1.contourf(lons2d,lats2d,cyclic_data,transform=ccrs.PlateCarree(),levels = levels,cmap=cmap,extend='both')#'seismic',extend='both')

    cbar = axgr.cbar_axes[0].colorbar(plot, extend='both')
    cbar.set_ticks(levels)
    cbar.set_label(r'$\degree C$',fontsize=16)
    cbar.ax.tick_params(labelsize=16)
    ax1.add_feature(cart.feature.LAND, zorder=100, edgecolor='k',facecolor='#808080')

    ax1.set_title(f"Hybrid Model Mean SST",fontsize=18,fontweight="bold")
    ax1.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())


    cyclic_data, cyclic_lons = add_cyclic_point(mean_climo - 273.15, coord=lons)
    lons2d,lats2d = np.meshgrid(cyclic_lons,lats)
    ax2 = axgr[0]#plt.subplot(221,projection=ccrs.PlateCarree())
    ax2.coastlines()

    ax2.set_xticks([-180,-120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree())
    ax2.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
    lat_formatter = LatitudeFormatter()
    ax2.xaxis.set_major_formatter(lon_formatter)
    ax2.yaxis.set_major_formatter(lat_formatter)
    gl = ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                       linewidth=2, color='gray', alpha=0.5, linestyle='--')

    #plot = ax1.pcolormesh(lons2d,lats2d,cyclic_data,transform=ccrs.PlateCarree(),vmin=-1*diff_max,vmax=diff_max,cmap='seismic')
    levels = np.arange(0,33,4)
    plot = ax2.contourf(lons2d,lats2d,cyclic_data,transform=ccrs.PlateCarree(),levels = levels,cmap=cmap,extend='both')#'seismic',extend='both')

    cbar = axgr.cbar_axes[2].colorbar(plot, extend='both')
    cbar.set_ticks(levels)
    cbar.set_label(r'$\degree C$',fontsize=16)
    cbar.ax.tick_params(labelsize=16)
    ax2.add_feature(cart.feature.LAND, zorder=100, edgecolor='k',facecolor='#808080')

    ax2.set_title(f"ERA5 Mean SST",fontsize=18,fontweight="bold")
    ax2.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())

    #### Bias
    cyclic_data, cyclic_lons = add_cyclic_point(mean_hybrid - mean_climo, coord=lons)
    lons2d,lats2d = np.meshgrid(cyclic_lons,lats)
    ax3 = axgr[4]#plt.subplot(221,projection=ccrs.PlateCarree())
    ax3.coastlines()

    ax3.set_xticks([-180,-120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree())
    ax3.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax3.xaxis.set_major_formatter(lon_formatter)
    ax3.yaxis.set_major_formatter(lat_formatter)
    gl = ax3.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                       linewidth=2, color='gray', alpha=0.5, linestyle='--')

    #plot = ax1.pcolormesh(lons2d,lats2d,cyclic_data,transform=ccrs.PlateCarree(),vmin=-1*diff_max,vmax=diff_max,cmap='seismic')
    levels = [-5,-3,-2,-1,-0.1,0.1,1,2,3,5]
    plot = ax3.contourf(lons2d,lats2d,cyclic_data,transform=ccrs.PlateCarree(),levels = levels,cmap=cmap_e3sm_diff_colormap_greyer,extend='both')#'seismic',extend='both')

    cbar = axgr.cbar_axes[4].colorbar(plot, extend='both')
    cbar.set_ticks(levels)
    cbar.set_label(r'$\degree C$',fontsize=16)
    cbar.ax.tick_params(labelsize=16)
    ax3.add_feature(cart.feature.LAND, zorder=100, edgecolor='k',facecolor='#808080')

    ax3.set_title(f"Sea Surface Temp. Bias\n(Model - ERA5)",fontsize=18,fontweight="bold")
    ax3.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())

    ###Monthly mean std###

    cyclic_data, cyclic_lons = add_cyclic_point(mean_hybrid_std, coord=lons)
    lons2d,lats2d = np.meshgrid(cyclic_lons,lats)

    ax4 = axgr[3]#plt.subplot(221,projection=ccrs.PlateCarree())
    ax4.coastlines()

    ax4.set_xticks([-180,-120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree())
    ax4.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax4.xaxis.set_major_formatter(lon_formatter)
    ax4.yaxis.set_major_formatter(lat_formatter)
    gl = ax4.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                       linewidth=2, color='gray', alpha=0.5, linestyle='--')

    levels = [0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6]
    temp_colormap = sns.color_palette("Spectral", as_cmap=True)
    temp_colormap = temp_colormap.reversed()
    cmap = temp_colormap
    plot = ax4.contourf(lons2d,lats2d,cyclic_data,transform=ccrs.PlateCarree(),levels = levels,cmap=cmap,extend='max')#'seismic',extend='both')

    cbar = axgr.cbar_axes[1].colorbar(plot, extend='both')
    cbar.set_ticks(levels)
    cbar.set_label(r'$\degree C$',fontsize=16)
    cbar.ax.tick_params(labelsize=16)
    ax4.add_feature(cart.feature.LAND, zorder=100, edgecolor='k',facecolor='#808080')

    ax4.set_title(f"Hybrid Model SST Standard Deviation",fontsize=18,fontweight="bold")
    ax4.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())

    cyclic_data, cyclic_lons = add_cyclic_point(mean_climo_std, coord=lons)
    lons2d,lats2d = np.meshgrid(cyclic_lons,lats)
    ax5 = axgr[1]#plt.subplot(221,projection=ccrs.PlateCarree())
    ax5.coastlines()

    ax5.set_xticks([-180,-120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree())
    ax5.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax5.xaxis.set_major_formatter(lon_formatter)
    ax5.yaxis.set_major_formatter(lat_formatter)
    gl = ax5.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                        linewidth=2, color='gray', alpha=0.5, linestyle='--')

    levels = [0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6]
    plot = ax5.contourf(lons2d,lats2d,cyclic_data,transform=ccrs.PlateCarree(),levels = levels,cmap=cmap,extend='max')#'seismic',extend='both')

    cbar = axgr.cbar_axes[3].colorbar(plot, extend='both')
    cbar.set_ticks(levels)
    cbar.set_label(r'$\degree C$',fontsize=16)
    cbar.ax.tick_params(labelsize=16)
    ax5.add_feature(cart.feature.LAND, zorder=100, edgecolor='k',facecolor='#808080')

    ax5.set_title(f"ERA5 SST Standard Deviation",fontsize=18,fontweight="bold")
    ax5.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())

    #### Bias
    cyclic_data, cyclic_lons = add_cyclic_point(mean_hybrid_std - mean_climo_std, coord=lons)
    lons2d,lats2d = np.meshgrid(cyclic_lons,lats)
    ax6 = axgr[5]#plt.subplot(221,projection=ccrs.PlateCarree())
    ax6.coastlines()

    ax6.set_xticks([-180,-120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree())
    ax6.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax6.xaxis.set_major_formatter(lon_formatter)
    ax6.yaxis.set_major_formatter(lat_formatter)
    gl = ax6.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                       linewidth=2, color='gray', alpha=0.5, linestyle='--')

    levels = [-0.6,-0.4,-0.2,-0.05,0.05,0.2,0.4,0.6]

    plot = ax6.contourf(lons2d,lats2d,cyclic_data,transform=ccrs.PlateCarree(),levels = levels,cmap=cmap_e3sm_diff_colormap_greyer,extend='both')#'seismic',extend='both')

    cbar = axgr.cbar_axes[5].colorbar(plot, extend='both')
    cbar.set_ticks(levels)
    cbar.set_label(r'$\degree C$',fontsize=16)
    cbar.ax.tick_params(labelsize=16)
    ax6.add_feature(cart.feature.LAND, zorder=100, edgecolor='k',facecolor='#808080')

    ax6.set_title(f"Standard Deviation Difference\n(Model - ERA5)",fontsize=18,fontweight="bold")
    ax6.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())

    plt.show()

def make_ds_time_dim(ds,timestep,startdate):
    begin_year_str = startdate.strftime("%Y-%m-%d")

    attrs = {"units": f"hours since {begin_year_str} "}

    ds = ds.assign_coords({"Timestep": ("Timestep", ds.Timestep.values*timestep, attrs)})
    ds = xr.decode_cf(ds)

    return ds

def count_enso_peaks(array,distance,height):
    print(np.shape(array))
    peaks, _ = find_peaks(array, height=height,distance=distance)

    plt.plot(array)
    plt.plot(peaks,array[peaks], "x")
    plt.show()

def power_spectra_enso(data):
    time_step = 1/1460#4.0#1/14540
    power_spectra_era = np.zeros((np.shape(data)))

    hamming = np.hamming(np.shape(data)[0])

    fft_data = np.fft.fft(data*hamming)
    power_spectra_data = np.abs(fft_data)**2.0

    freq = np.fft.fftfreq(len(data),time_step)

    idx = np.argsort(freq)
    idx = idx[int(len(idx)/2)::]

    return power_spectra_data, freq, idx
    
def sst_trend(ds):
    result = ds.groupby("time.month") - ds.groupby("time.month").mean("time")  
    temp = result['sst'][:].values


    average_temp = np.nanmean(temp,axis=(1,2))
    print(np.shape(average_temp))

    time = np.arange(1,np.shape(average_temp)[0]+1)
    time = time/1460
    plt.plot(time,uniform_filter1d(average_temp,size=1460))

    ticks = np.arange(1,31)
    plt.xticks(ticks)

    plt.show()

def acf(x, length=20):
    return np.array([1]+[np.corrcoef(x[:-i], x[i:])[0,1]  \
        for i in range(1, length)])

def autocorr_plot(ds_hybrid,ds_era):
    hybrid_nino3_4 = nino_index_monthly(ds_hybrid,"3.4")
    
    observed_nino3_4 = nino_index_monthly(ds_observed,"3.4")
    print(hybrid_nino3_4)
    print(observed_nino3_4)

     
    hybrid_nino = uniform_filter1d(hybrid_nino3_4['SST'].values, size=3, origin=1)
    era5_nino = uniform_filter1d(observed_nino3_4['sst'].values, size=3, origin=1)

    hybrid_acf = acf(hybrid_nino,length=49)
    era_acf = acf(era5_nino,length=49)

    plt.rc('font', family='serif')

    x = np.arange(0,49)
    ticks = np.arange(0,49,6)
    plt.plot(x,hybrid_acf,label='Coupled Model')
    plt.plot(x,era_acf,label='ERA5') 
    plt.hlines(0.0,x[0],x[-1],linewidth=0.5,color='tab:gray',ls='--')
   
    plt.title('Nino3.4 Autocorrelation',fontsize=18) 
    plt.ylim([-1,1])
    plt.xlim([0,48])
    plt.yticks(fontsize=16)
    plt.xticks(ticks,fontsize=16)

    plt.xlabel('lag (months)',fontsize=16)
    plt.ylabel('autocorrelation',fontsize=16)
    plt.legend(fontsize=16)
    plt.show()
    

   
def oni_soi_timeseries(ds_hybrid,ds_observed,ds_era,ds_speedy):
    hybrid_soi = southern_oscillation_index(ds_hybrid,ds_era,startdate,enddate)

    hybrid_nino3_4 = nino_index(ds_hybrid['SST'],"3.4")
    observed_nino3_4 = nino_index(ds_observed['sst'],"3.4")
    speedy_nino = nino_index_speedy(ds_speedy['sst'],startdate,enddate,"3.4")
    #print(hybrid_nino3_4)
    #print(np.shape(hybrid_nino3_4))

    #print(np.shape(speedy_nino))
    time = np.arange(0,np.shape(hybrid_nino3_4)[0])
    time = time/1461
    time = time + startdate_hybrid.year

    time_soi = np.arange(0,np.shape(hybrid_soi)[0])
    time_soi = time_soi/1461
    time_soi = time_soi + startdate_hybrid.year

    speedy_time = np.arange(0,np.shape(speedy_nino)[0])
    speedy_time = speedy_time/12
    speedy_time = speedy_time + startdate_hybrid.year

    era_time = np.arange(0,np.shape(observed_nino3_4)[0])
    era_time = era_time/1461
    era_time = era_time + startdate_hybrid.year

    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()

    p1, = ax1.plot(time,uniform_filter1d(hybrid_nino3_4, size=360),color='k',ls='-',linewidth=2.0,label="Coupled Model ONI 3 Month Average")
    ax1.fill_between(time, 0.8, uniform_filter1d(hybrid_nino3_4, size=360), where = uniform_filter1d(hybrid_nino3_4, size=360) >= 0.8, alpha = 0.2, color = 'red')
    ax1.fill_between(time,uniform_filter1d(hybrid_nino3_4, size=360), -0.8, where = uniform_filter1d(hybrid_nino3_4, size=360) <= -0.8, alpha = 0.2, color = 'blue')
    #p2, = ax1.plot(era_time,uniform_filter1d(observed_nino3_4,size=360),color='#377eb8',ls='-',linewidth=2.0,label="ERA5 3 Month Average")
    #ax1.plot(speedy_time,uniform_filter1d(speedy_nino,size=3),color='#4daf4a',ls='-',linewidth=2.0,label="SPEEDY 3 Month Average")

    #ax1.hlines(-0.5,time[0],time[-1],color='b',ls='--')
    #ax1.hlines(0.5,time[0],time[-1],color='r',ls='--')

    #p2, = ax2.plot(time_soi,uniform_filter1d(hybrid_soi,size=120),color='#4daf4a',ls='-',linewidth=1.0,label="Hybrid SOI 1 Month Average")
    #p3, = ax2.plot(time_soi,uniform_filter1d(hybrid_soi,size=600),color='#4daf4a',ls='--',linewidth=2.0,label="Hybrid SOI 5 Month Average")
    p2, = ax2.plot(time_soi,uniform_filter1d(hybrid_soi,size=600),color='#4daf4a',ls='--',linewidth=2.0,label="Coupled SOI 5 Month Average")
    print(hybrid_soi)

    ticks = np.arange(startdate_hybrid.year,2047,5)#enddate_hybrid.year+1)#[0,1460,2920,4380,5840,7300]
    labels_ticks = ticks - startdate_hybrid.year

    ax1.set_xlim([np.min(ticks),np.max(ticks)])
    ax1.set_ylim([-3.2,3.2])
    ax2.set_ylim([-20,20]) 

    plt.title("Oceanic Nino Index / Southern Oscillation",fontsize=18,fontweight="bold")
    #plt.title(r"Oceanic Niño Index",fontsize=28,fontweight="bold")

    ax1.set_xticks(ticks)
    ax1.set_xticklabels(labels_ticks,fontsize=18)
    ax1.set_xlabel("Years into Simulation",fontsize=20)

    ax1.set_ylabel("ONI",fontsize=20)
    ax2.set_ylabel("SOI",fontsize=16)

    ax1.tick_params(axis='both', which='major', labelsize=18)
    ax1.tick_params(axis='both', which='minor', labelsize=18)

    ax2.tick_params(axis='both', which='major', labelsize=14)
    ax2.tick_params(axis='both', which='minor', labelsize=14)

    #ax1.legend(handles=[p1, p2, p3],fontsize=16)
    ax1.legend(handles=[p1, p2],fontsize=16)

    plt.show()
    
def enso_combined_plots(ds_observed_sst,ds_era,ds_hybrid):
     
    print('startdate,enddate',startdate,enddate)
    hybrid_soi = southern_oscillation_index(ds_hybrid,ds_era,startdate,enddate)

    hybrid_nino3_4 = nino_index(ds_hybrid['SST'],"3.4")
    observed_nino3_4 = nino_index(ds_observed_sst['sst'],"3.4")
    speedy_nino = nino_index_speedy(ds_speedy['sst'],startdate,enddate,"3.4")

    ##Monthly or seasonal data
    hybrid_nino3_4_monthly = nino_index_monthly(ds_hybrid,"3.4")

    observed_nino3_4_monthly = nino_index_monthly(ds_observed_sst,"3.4")

    hybrid_nino_3monthly = uniform_filter1d(hybrid_nino3_4_monthly['SST'].values, size=3, origin=1)
    era5_nino_3monthly = uniform_filter1d(observed_nino3_4_monthly['sst'].values, size=3, origin=1)

    hybrid_acf = acf(hybrid_nino_3monthly,length=49)
    era_acf = acf(era5_nino_3monthly,length=49)

    file_name = '/home/troyarcomano/FortranReservoir/vert_loc_hybridspeedy_leakage/psl.noaa.gov/gcos_wgsp/Timeseries/Data/nino34.long.anom.data'
    A = np.genfromtxt(file_name,dtype=None,usecols=np.arange(1,13),skip_header=1,skip_footer=8)
    dat = A.flatten()

    t0 = 1870.
    dt = 1/12.0  # In years

    f_Had, Pxx_spec_Had = welch(observed_nino3_4_monthly['sst'].values,nperseg=128*2, scaling='spectrum')

    f_hybrid, Pxx_spec_hybrid = welch(hybrid_nino3_4_monthly['SST'].values,nperseg=128*2, scaling='spectrum')

    print('f_hybrid',f_hybrid,1/f_hybrid)

    HadISST_glbl_power, HadISST_fft_power, HadISST_period, HadISST_fftfreqs = get_wavelet_fft_power(dat,dt,t0)

    print('HadISST_fftfreqs',HadISST_fftfreqs,1/HadISST_fftfreqs)

    hybrid_glbl_power, hybrid_fft_power, hybrid_period, hybrid_fftfreqs = get_wavelet_fft_power(hybrid_nino3_4_monthly['SST'].values,dt,t0)


    hybrid_fft_power = Pxx_spec_hybrid
    HadISST_fft_power = Pxx_spec_Had
    HadISST_fftfreqs = f_Had*12
    hybrid_fftfreqs = f_hybrid*12

    time = np.arange(0,np.shape(hybrid_nino3_4)[0])
    time = time/1461
    time = time + startdate_hybrid.year

    time_soi = np.arange(0,np.shape(hybrid_soi)[0])
    time_soi = time_soi/1461
    time_soi = time_soi + startdate_hybrid.year

    speedy_time = np.arange(0,np.shape(speedy_nino)[0])
    speedy_time = speedy_time/12
    speedy_time = speedy_time + startdate_hybrid.year

    era_time = np.arange(0,np.shape(observed_nino3_4)[0])
    era_time = era_time/1461
    era_time = era_time + startdate_hybrid.year


    axd = plt.figure(constrained_layout=True).subplot_mosaic(
                     """
                     AA
                     BC
                     """
                     )
    print(axd)
    ax1 = axd['A']

    ax1b = ax1.twinx()

    p1, = ax1.plot(time,uniform_filter1d(hybrid_nino3_4, size=360),color='k',ls='-',linewidth=2.0,label="Hybrid Model ONI 3 Month Average")
    ax1.fill_between(time, 0.8, uniform_filter1d(hybrid_nino3_4, size=360), where = uniform_filter1d(hybrid_nino3_4, size=360) >= 0.8, alpha = 0.2, color = 'red')
    ax1.fill_between(time,uniform_filter1d(hybrid_nino3_4, size=360), -0.8, where = uniform_filter1d(hybrid_nino3_4, size=360) <= -0.8, alpha = 0.2, color = 'blue')
    #p2, = ax1.plot(era_time,uniform_filter1d(observed_nino3_4.sel(Timestep=slice(,size=360),color='#377eb8',ls='-',linewidth=2.0,label="ERA5 3 Month Average")
    #ax1.plot(speedy_time,uniform_filter1d(speedy_nino,size=3),color='#4daf4a',ls='-',linewidth=2.0,label="SPEEDY 3 Month Average")

    #ax1.hlines(-0.5,time[0],time[-1],color='b',ls='--')
    #ax1.hlines(0.5,time[0],time[-1],color='r',ls='--')

    #p2, = ax2.plot(time_soi,uniform_filter1d(hybrid_soi,size=120),color='#4daf4a',ls='-',linewidth=1.0,label="Hybrid SOI 1 Month Average")
    #p3, = ax2.plot(time_soi,uniform_filter1d(hybrid_soi,size=600),color='#4daf4a',ls='--',linewidth=2.0,label="Hybrid SOI 5 Month Average")
    p2, = ax1b.plot(time_soi,uniform_filter1d(hybrid_soi,size=600),color='#4daf4a',ls='--',linewidth=2.0,label="Hybrid Model SOI 5 Month Average")
    print(hybrid_soi)

    ticks = np.arange(startdate_hybrid.year,2057,5)#enddate_hybrid.year+1)#[0,1460,2920,4380,5840,7300]
    labels_ticks = ticks - startdate_hybrid.year

    ax1.set_xlim([np.min(ticks),np.max(ticks)])
    ax1.set_ylim([-3.2,3.2])
    ax1b.set_ylim([-20,20])

    plt.title("Oceanic Nino Index / Southern Oscillation",fontsize=18,fontweight="bold")
    #plt.title(r"Oceanic Niño Index",fontsize=28,fontweight="bold")

    ax1.set_xticks(ticks)
    ax1.set_xticklabels(labels_ticks,fontsize=18)
    ax1.set_xlabel("Years into Simulation",fontsize=20)

    ax1.set_ylabel("ONI",fontsize=20)
    ax1b.set_ylabel("SOI",fontsize=16)

    ax1.tick_params(axis='both', which='major', labelsize=18)
    ax1.tick_params(axis='both', which='minor', labelsize=18)

    ax1b.tick_params(axis='both', which='major', labelsize=14)
    ax1b.tick_params(axis='both', which='minor', labelsize=14)

    #ax1.legend(handles=[p1, p2, p3],fontsize=16)
    ax1.legend(handles=[p1, p2],fontsize=16)

    #####AAutocorrelation#####

    ax2 = axd['B']
    x = np.arange(0,49)
    ticks = np.arange(0,49,6)
    ax2.plot(x,era_acf,label='ERA5')
    ax2.plot(x,hybrid_acf,label='Hybrid Model')
    ax2.hlines(0.0,x[0],x[-1],linewidth=0.5,color='tab:gray',ls='--')

    ax2.set_title('Nino3.4 Autocorrelation',fontsize=18)
    ax2.set_ylim([-1,1])
    ax2.set_xlim([0,48])
    ax2.set_xticks(ticks)
    ax2.tick_params(axis='both', which='major', labelsize=16)

    ax2.set_xlabel('lag (months)',fontsize=16)
    ax2.set_ylabel('autocorrelation',fontsize=16)
    ax2.legend(fontsize=16)


    xticks = np.arange(0,11,1)

    ax3 = axd['C']
    ax3.plot(1/HadISST_fftfreqs,HadISST_fft_power,linewidth=1.5,label='ERA5')
    ax3.plot(1/hybrid_fftfreqs,hybrid_fft_power,linewidth=1.5,label='Hybrid Model')

    ax3.set_title('Nino 3.4 Power Spectrum',fontsize=20)
    ax3.set_ylabel(r'Power[$\degree C^{2}$]',fontsize=16)
    ax3.set_xlabel('Period (Years)',fontsize=16)
    #cx.set_yticks(np.arange(0,40,5))
    #cx.set_yticklabels(np.arange(0,40,5),fontsize=16)
    #cx.set_ylim([0, hybrid_glbl_power.max()])

    ax3.set_xticks(xticks)
    ax3.set_xticklabels(xticks,fontsize=16)
    ax3.set_xlim([0,xticks.max()])
    ax3.legend(fontsize=16)

    plt.show()

def get_wavelet_fft_power(dat,dt,t0):
    N = dat.size
    t = np.arange(0, N) * dt + t0
    #We write the following code to detrend and normalize the input data by its
    # standard deviation. Sometimes detrending is not necessary and simply
    # removing the mean value is good enough. However, if your dataset has a well
    # defined trend, such as the Mauna Loa CO\ :sub:`2` dataset available in the
    # above mentioned website, it is strongly advised to perform detrending.
    # Here, we fit a one-degree polynomial function and then subtract it from the
    # original data.
    p = np.polyfit(t - t0, dat, 1)
    dat_notrend = dat - np.polyval(p, t - t0)
    std = dat_notrend.std()  # Standard deviation
    var = std ** 2  # Variance
    dat_norm = dat_notrend / std  # Normalized dataset

    # The next step is to define some parameters of our wavelet analysis. We
    # select the mother wavelet, in this case the Morlet wavelet with
    # :math:`\omega_0=6`.
    mother = wavelet.Morlet(6)
    #s0 = 2 * dt  # Starting scale, in this case 2 * 0.25 years = 6 months
    s0 = 6 * dt
    dj = 1 / 12  # Twelve sub-octaves per octaves
    J = 7 / dj  # Seven powers of two with dj sub-octaves
    alpha, _, _ = wavelet.ar1(dat)  # Lag-1 autocorrelation for red noise

    # The following routines perform the wavelet transform and inverse wavelet
    # transform using the parameters defined above. Since we have normalized our
    # input time-series, we multiply the inverse transform by the standard
    # deviation.
    wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(dat_norm, dt, dj, s0, J,
                                                          mother)
    iwave = wavelet.icwt(wave, scales, dt, dj, mother) * std

    # We calculate the normalized wavelet and Fourier power spectra, as well as
    # the Fourier equivalent periods for each wavelet scale.
    power = (np.abs(wave)) ** 2
    fft_power = np.abs(fft) ** 2
    period = 1 / freqs

    # We could stop at this point and plot our results. However we are also
    # interested in the power spectra significance test. The power is significant
    # where the ratio ``power / sig95 > 1``.
    signif, fft_theor = wavelet.significance(1.0, dt, scales, 0, alpha,
                                             significance_level=0.95,
                                             wavelet=mother)
    sig95 = np.ones([1, N]) * signif[:, None]
    sig95 = power / sig95

    # Then, we calculate the global wavelet spectrum and determine its
    # significance level.
    glbl_power = power.mean(axis=1)
    dof = N - scales  # Correction for padding at edges
    glbl_signif, tmp = wavelet.significance(var, dt, scales, 1, alpha,
                                            significance_level=0.95, dof=dof,
                                            wavelet=mother)

    # We also calculate the scale average between 2 years and 8 years, and its
    # significance level.
    sel = find((period >= 2) & (period < 8))
    Cdelta = mother.cdelta
    scale_avg = (scales * np.ones((N, 1))).transpose()
    scale_avg = power / scale_avg  # As in Torrence and Compo (1998) equation 24
    scale_avg = var * dj * dt / Cdelta * scale_avg[sel, :].sum(axis=0)
    scale_avg_signif, tmp = wavelet.significance(var, dt, scales, 2, alpha,
                                                 significance_level=0.90,#5,
                                                 dof=[scales[sel[0]],
                                                      scales[sel[-1]]],
                                                 wavelet=mother)


    return var * glbl_power, var * fft_power, period, fftfreqs


def nino_index_monthly_specified_climo(sst_data,region,monthly_climo):
    # monthly_climo are monthly climotologies to use when calculating anomalies.
    # they are xarray DataArray grouped by month 

    if region == "1+2":
        lat_slice = slice(-10,0)
        lon_slice = slice(360-90,360-80)
    elif region == "3":
        lat_slice = slice(-5,5)
        lon_slice = slice(360-150,360-90)
    elif region == "3.4" :
        lat_slice = slice(-5,5)
        lon_slice = slice(360-170,360-120)
    elif region == "4" :
        lat_slice = slice(-5,5)
        lon_slice = slice(360-200,360-150)
 
    start_year = 1981
    end_year = 2010

    if ('Timestep' in sst_data.dims):
       sst_data_copy = sst_data.sel(Lon=lon_slice,Lat=lat_slice)
       sst_data_copy = sst_data_copy.resample(Timestep="1MS").mean(dim="Timestep")
       gb = sst_data_copy.groupby('Timestep.month')
        
       if ('time' in monthly_climo.dims):
          monthly_climo = monthly_climo.rename({"time": "Timestep", "lat": "Lat", "lon": "Lon"})
         
       monthly_climo_copy = monthly_climo.sel(Lon=lon_slice,Lat=lat_slice)
       monthly_climo_copy = monthly_climo_copy.resample(Timestep="1MS").mean(dim="Timestep")
       monthly_climo = monthly_climo_copy.groupby("Timestep.month")
       #print(f'\n\nDims of monthly_climo PRED_PATH: {monthly_climo_copy.dims}\nShape: {monthly_climo_copy.shape}\n\n')
       #print(f'\n\nDims of sst_data PRED_PATH: {sst_data_copy.dims}\nShape: {sst_data_copy.shape}\n\n')
       #print(f'\n\nDims of monthly_climo_grouped_mean PRED_PATH: {monthly_climo.mean("Timestep").dims}\nContent: {monthly_climo.mean("Timestep")}\n\n')
       #print(f'\n\nDims of sst_data_grouped PRED_PATH: {gb.dims}\nContents of month 1: {gb[1]}\n\n')
       tos_nino34_anom = gb - monthly_climo.mean('Timestep')
       #print(f'\n\ntos_nino34_anom: {tos_nino34_anom}\n\n')
       index_nino34 = tos_nino34_anom.mean(dim=['Lat', 'Lon'], skipna=True)
       #print(f'\n\nindex_nino34: {index_nino34}\n\n')
 
    else:
        sst_data_copy = sst_data.sel(lon=lon_slice,lat=lat_slice)
        sst_data_copy = sst_data_copy.resample(time="1MS").mean(dim="time")
        gb = sst_data_copy.groupby('time.month')
        
        if ('Timestep' in monthly_climo.dims):
           monthly_climo = monthly_climo.rename({"Timestep": "time", "Lat": "lat", "Lon": "lon"})

        print(f'\n\nmonthly_climo_OBSERVED_PATH: {monthly_climo}\n\n')
        monthly_climo_copy = monthly_climo.sel(lon=lon_slice,lat=lat_slice)
        monthly_climo_copy = monthly_climo_copy.resample(time="1MS").mean(dim="time")
        monthly_climo = monthly_climo_copy.groupby("time.month").mean('time')
        tos_nino34_anom = gb - monthly_climo
        index_nino34 = tos_nino34_anom.mean(dim=['lat', 'lon'])
    
    return index_nino34


def nino_index_monthly_specified_climo_hybrid(ds_era,region,ds_hybrid):
    # monthly_climo are monthly climotologies to use when calculating anomalies.
    # they are xarray DataArray grouped by month

    if region == "1+2":
        lat_slice = slice(-10,0)
        lon_slice = slice(360-90,360-80)
    elif region == "3":
        lat_slice = slice(-5,5)
        lon_slice = slice(360-150,360-90)
    elif region == "3.4" :
        lat_slice = slice(-5,5)
        lon_slice = slice(360-170,360-120)
    elif region == "4" :
        lat_slice = slice(-5,5)
        lon_slice = slice(360-200,360-150)

    start_year = 1981
    end_year = 2010
 
    ds_era = ds_era.sel(lon=lon_slice,lat=lat_slice)
    monthly_era = ds_era.resample(time='1MS').mean(dim='time')

    ds_hybrid = ds_hybrid.sel(Lon=lon_slice,Lat=lat_slice)
    monthly_hybrid = ds_hybrid.resample(Timestep='1MS').mean(dim='Timestep')
    
    monthly_hybrid = monthly_hybrid['SST']
    
    climatology = monthly_era.groupby('time.month').mean('time')
    monthly_hybrid_anoms = monthly_hybrid.groupby('Timestep.month') - climatology
    #print(f'\n\nmonthly_hybrid_anoms: {monthly_hybrid_anoms}\n\n') 
   
    nino34_index = monthly_hybrid_anoms.mean(dim=['Lat','Lon','lon','lat'])
    return nino34_index

def precip_anom_specified_climo_hybrid(ds_era,ds_hybrid):
    
    monthly_era = ds_era.resample(Timestep='1MS').mean(dim='Timestep')
    monthly_hybrid = ds_hybrid.resample(Timestep='1MS').mean(dim='Timestep')

    climatology = monthly_era.groupby('Timestep.month').mean('Timestep')
    monthly_hybrid_anoms = monthly_hybrid.groupby('Timestep.month') - climatology

    return monthly_hybrid_anoms

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

def get_anom_specified_climo_hybrid_daily(ds_era,ds_hybrid):

    try:
       monthly_era = ds_era.resample(Timestep='1D').sum(dim='Timestep')
       monthly_hybrid = ds_hybrid.resample(Timestep='1D').sum(dim='Timestep')

       climatology = monthly_era.groupby('Timestep.dayofyear').mean('Timestep')
       monthly_hybrid_anoms = monthly_hybrid.groupby('Timestep.dayofyear') - climatology
    except:
       monthly_era = ds_era.resample(time='1D').sum(dim='time')
       monthly_hybrid = ds_hybrid.resample(time='1D').sum(dim='time')

       climatology = monthly_era.groupby('time.dayofyear').mean('time')
       monthly_hybrid_anoms = monthly_hybrid.groupby('time.dayofyear') - climatology

    return monthly_hybrid_anoms


def get_anom_specified_climo_hybrid_weekly(ds_era,ds_hybrid):

    try:
       monthly_era = ds_era.resample(Timestep='1W').mean(dim='Timestep') 
       monthly_hybrid = ds_hybrid.resample(Timestep='1W').mean(dim='Timestep')

       climatology = monthly_era.groupby('Timestep.weekofyear').mean('Timestep')
       monthly_hybrid_anoms = monthly_hybrid.groupby('Timestep.weekofyear') - climatology
    except:
       monthly_era = ds_era.resample(time='1W').mean(dim='time')
       monthly_hybrid = ds_hybrid.resample(time='1W').mean(dim='time')

       climatology = monthly_era.groupby('time.weekofyear').mean('time')
       monthly_hybrid_anoms = monthly_hybrid.groupby('time.weekofyear') - climatology

    return monthly_hybrid_anoms

def get_anom_specified_climo_hybrid_seasonal(ds_era,ds_hybrid):

    try:
       monthly_era = ds_era.resample(Timestep='QS-DEC').mean(dim='Timestep')
       monthly_hybrid = ds_hybrid.resample(Timestep='QS-DEC').mean(dim='Timestep')

       climatology = monthly_era.groupby('Timestep.season').mean('Timestep')
       monthly_hybrid_anoms = monthly_hybrid.groupby('Timestep.season') - climatology
    except:
       monthly_era = ds_era.resample(time='QS-DEC').mean(dim='time')
       monthly_hybrid = ds_hybrid.resample(time='QS-DEC').mean(dim='time')

       climatology = monthly_era.groupby('time.season').mean('time')
       monthly_hybrid_anoms = monthly_hybrid.groupby('time.season') - climatology

    return monthly_hybrid_anoms


def get_predicted_nino34_ens(startdates, prediction_length, timestep, outlen=24):
    """Load in all observed, Hybrid, and SPEEDY predicted SST datasets. Obtain (monthly) nino3.4 index from each dataset.
    Inputs:
        startdates: list of datetime objects of prediction start dates.
        prediction_length: length of forecast (in hours)
        timestep: timestep of hybrid model
        outlen: temporal length of predictions
    Returns:
        ds_hybrid: [xarray DataArray] hybrid-predicted monthly nino3.4 indexes (3-month average).
        ds_observed: [xarray DataArray] observed monthly nino3.4 index (3-month averaged).
        ds_per: [xarray DataArray] persistence nino3.4 index."""

    hybrid_root = "/scratch/user/dpp94/Predictions/Hybrid/hybrid_prediction_era6000_20_20_20_sigma0.5_beta_res0.001_beta_model_1.0_prior_0.0_overlap1_vertlevel_1_precip_epsilon0.001_ohtc_multiple_leakage_test_oceantimestep_72hr_train1981_2002_oldcal__pred_newcal_trial_"
   
    #hybrid_climo_ref_path = "/scratch/user/dpp94/Predictions/Hybrid/hybrid_prediction_era6000_20_20_20_sigma0.5_beta_res0.001_beta_model_1.0_prior_0.0_overlap1_vertlevel_1_precip_epsilon0.001_multi_gaussian_noisempi_res_test_final_before_committrial_12_29_2006_00.nc"
    #hybrid_climo_ref_startdate = datetime(2006,12,29,0)

    obs_climo_ref_startdate = datetime(1981,1,1,0) #59,1,1,0)   
    obs_climo_ref_enddate = datetime(2002,12,1,0) #2006,12,1,0

    # Obtain observed nino index
    date, enddate = startdates[0] - timedelta(hours=24*60), startdates[-1] + timedelta(hours=prediction_length)
    ds_observed = get_obs_sst_timeseries(date, enddate, timestep)
    ds_observed_climo = get_obs_sst_timeseries(obs_climo_ref_startdate, obs_climo_ref_enddate, timestep)
    ds_observed = nino_index_monthly_specified_climo(ds_observed, "3.4", ds_observed_climo)
    ds_observed_unsmoothed = ds_observed["sst"]
    time_index = ds_observed["time"]
    ds_observed = uniform_filter1d(ds_observed["sst"].values, size=3, origin=1)
    ds_observed = xr.DataArray(ds_observed, dims="time", coords={"time": time_index.values}, name=date.strftime("%m_%d_%Y_%H"))
    print(f'observed SSTs: {ds_observed}')
    # Note, I believe groupby month sorts the result by calendar month
    # so that, ds_observed_unq[1] is for Jan, etc
 
    # Obtain data for calculating monthly climotology for predictions
    #ds_hybrid_climo = xr.open_dataset(hybrid_climo_ref_path)
    #ds_hybrid_climo = make_ds_time_dim(ds_hybrid_climo, timestep, hybrid_climo_ref_startdate)

    ds_hybrid = []
    ds_per = []
    for date in startdates:
        # Load in hybrid predictions
        date_str = date.strftime("%m_%d_%Y_%H")
        filepath = hybrid_root + date_str + ".nc"
        print(f'Loading in file at: {filepath}')
        ds_hybrid_temp = xr.open_dataset(filepath)
        ds_hybrid_temp = make_ds_time_dim(ds_hybrid_temp, timestep, date)
        ds_hybrid_temp = nino_index_monthly_specified_climo_hybrid(ds_observed_climo, "3.4", ds_hybrid_temp) #ds_hybrid_climo["SST"])
        time_index = ds_hybrid_temp["Timestep"]
        print(f'ds_hybrid_temp time_index: {time_index}\n')
        print(f'ds_observed time_index: {ds_observed_unsmoothed["time"]}\n')
        date_temp = np.argwhere(ds_observed_unsmoothed["time"].values == time_index[0].values).squeeze() - 1
        print(f'date_temp, date_temp-1: {date_temp}, {date_temp-1}\n')
        print(f'\nds_observed_unsmoothed.isel(): {ds_observed_unsmoothed.isel(time=slice(date_temp-1,date_temp)).values}\n')
        ds_hybrid_temp = np.concatenate((ds_observed_unsmoothed.isel(time=slice(date_temp-1,date_temp+1)).values, ds_hybrid_temp['sst'].values))
        ds_hybrid_temp = uniform_filter1d(ds_hybrid_temp, size=3, origin=1) #ds_hybrid_temp['sst'].values
        print(f'len(ds_hybrid_temp): {len(ds_hybrid_temp)}\n')
        ds_hybrid_temp = ds_hybrid_temp[2:]
        ds_hybrid_temp = xr.DataArray(ds_hybrid_temp, dims="time", coords={"time": time_index.values}, name=date_str)
        ds_hybrid_temp.coords["lead"] = ("time", np.arange(ds_hybrid_temp.size))
        #print(f'\n\nds_hybrid_temp: {ds_hybrid_temp}\n\n')

        # Load in observed SST for persistence
        #print(f'ds_observed["time"].values: {ds_observed["time"].values}\n')
        #print(f'time_index[0]: {time_index[0].values}\n')
        ds_per_temp_time = np.argwhere(ds_observed["time"].values == time_index[0].values).squeeze() - 1
        print(f'\n\ndata start date: {date}\npersistence start date: {ds_observed["time"].isel(time=ds_per_temp_time)}\n') 
        ds_per_temp = xr.ones_like(ds_hybrid_temp) * ds_observed.isel(time=ds_per_temp_time)
        ds_per_temp.coords["lead"] = ("time", np.arange(ds_per_temp.size)) 
        print(f'\n\nds_per_temp: {ds_per_temp}\n\n')

        ds_hybrid.append(ds_hybrid_temp.isel(time=slice(0,outlen)))
        ds_per.append(ds_per_temp.isel(time=slice(0,outlen)))
 
    # Concatenate predictions into single xarray DataArray
    ds_hybrid = xr.concat(ds_hybrid, dim="ens")
    ds_per = xr.concat(ds_per, dim='ens')

    return ds_hybrid, ds_observed, ds_per
   
def get_nino_index_rmse(startdates, prediction_length, timestep):
    """Load in all observed, Hybrid, and SPEEDY predicted SST datasets. Calculate RMSE of nino indexes from each dataset.
    Inputs:
        startdates: list of datetime objects of prediction start dates.
        prediction_length: length of forecast (in hours)
        timestep: timestep of hybrid model
    Returns:
        ds_hybrid: [xarray DataArray] hybrid-predicted monthly nino3.4 indexes (3-month average).
        ds_observed: [xarray DataArray] observed monthly nino3.4 index (3-month averaged).
        ds_per: [xarray DataArray] persistence nino3.4 index."""

    hybrid_root = "/scratch/user/dpp94/Predictions/Hybrid/hybrid_prediction_era6000_20_20_20_sigma0.5_beta_res0.001_beta_model_1.0_prior_0.0_overlap1_vertlevel_1_precip_epsilon0.001_ohtc_multiple_leakage_test_oceantimestep_72hr_uvwindtemplogponly_reg10_trial_"

    #hybrid_climo_ref_path = "/scratch/user/dpp94/Predictions/Hybrid/hybrid_prediction_era6000_20_20_20_sigma0.5_beta_res0.001_beta_model_1.0_prior_0.0_overlap1_vertlevel_1_precip_epsilon0.001_multi_gaussian_noisempi_res_test_final_before_committrial_12_29_2006_00.nc"
    #hybrid_climo_ref_startdate = datetime(2006,12,29,0)

    obs_climo_ref_startdate = datetime(1981,1,1,0)
    obs_climo_ref_enddate = datetime(2006,12,1,0)  #Should change this to 2002,12,1,0

    # Obtain observed nino index
    date, enddate = startdates[0] - timedelta(hours=24*60), startdates[-1] + timedelta(hours=prediction_length)
    ds_observed = get_obs_sst_timeseries(date, enddate, timestep)
    ds_observed_climo = get_obs_sst_timeseries(obs_climo_ref_startdate, obs_climo_ref_enddate, timestep)
    ds_observed = nino_index_monthly_specified_climo(ds_observed, "3.4", ds_observed_climo)
    ds_observed_unsmoothed = ds_observed["sst"]
    time_index = ds_observed["time"]
    ds_observed = uniform_filter1d(ds_observed["sst"].values, size=3, origin=1)
    ds_observed = xr.DataArray(ds_observed, dims="time", coords={"time": time_index.values}, name=date.strftime("%m_%d_%Y_%H"))
    print(f'observed SSTs: {ds_observed}')
    # Note, I believe groupby month sorts the result by calendar month
    # so that, ds_observed_unq[1] is for Jan, etc

    # Obtain data for calculating monthly climotology for predictions
    #ds_hybrid_climo = xr.open_dataset(hybrid_climo_ref_path)
    #ds_hybrid_climo = make_ds_time_dim(ds_hybrid_climo, timestep, hybrid_climo_ref_startdate)

    ds_hybrid = []
    ds_per = []
    for date in startdates:
        # Load in hybrid predictions
        date_str = date.strftime("%m_%d_%Y_%H")
        filepath = hybrid_root + date_str + ".nc"
        print(f'Loading in file at: {filepath}')
        ds_hybrid_temp = xr.open_dataset(filepath)
        ds_hybrid_temp = make_ds_time_dim(ds_hybrid_temp, timestep, date)
        ds_hybrid_temp = nino_index_monthly_specified_climo_hybrid(ds_observed_climo, "3.4", ds_hybrid_temp) #ds_hybrid_climo["SST"])
        time_index = ds_hybrid_temp["Timestep"]
        print(f'ds_hybrid_temp time_index: {time_index}\n')
        print(f'ds_observed_unsmoothed time_index: {ds_observed_unsmoothed["time"]}\n')
        date_temp = np.argwhere(ds_observed_unsmoothed["time"].values == time_index[0].values).squeeze() - 1
        print(f'date_temp, date_temp-1: {date_temp}, {date_temp-1}\n')
        print(f'\nds_observed_unsmoothed.isel(): {ds_observed_unsmoothed.isel(time=slice(date_temp-1,date_temp)).values}\n')
        ds_hybrid_temp = np.concatenate((ds_observed_unsmoothed.isel(time=slice(date_temp-1,date_temp+1)).values, ds_hybrid_temp['sst'].values))
        ds_hybrid_temp = uniform_filter1d(ds_hybrid_temp, size=3, origin=1) #ds_hybrid_temp['sst'].values
        print(f'len(ds_hybrid_temp): {len(ds_hybrid_temp)}\n')
        ds_hybrid_temp = ds_hybrid_temp[2:]
        ds_hybrid_temp = xr.DataArray(ds_hybrid_temp, dims="time", coords={"time": time_index.values}, name=date_str)
        #ds_hybrid_temp.coords["lead"] = ("time", np.arange(ds_hybrid_temp.size))
        print(f'\n\nds_hybrid_temp: {ds_hybrid_temp}\n\n')

        # Load in observed SST for persistence
        #print(f'ds_observed["time"].values: {ds_observed["time"].values}\n')
        #print(f'time_index[0]: {time_index[0].values}\n')
        ds_per_temp_time = np.argwhere(ds_observed["time"].values == time_index[0].values).squeeze() - 1
        print(f'\n\ndata start date: {date}\npersistence start date: {ds_observed["time"].isel(time=ds_per_temp_time)}\n')
        ds_per_temp = xr.ones_like(ds_hybrid_temp) * ds_observed.isel(time=ds_per_temp_time)
        #ds_per_temp.coords["lead"] = ("time", np.arange(ds_per_temp.size))
        print(f'\n\nds_per_temp: {ds_per_temp}\n\n')

        ds_hybrid.append(ds_hybrid_temp)
        ds_per.append(ds_per_temp)

    rmse = []
    rmse_per = []
    rmse_climo = []
    for i in range(len(ds_hybrid)):
        time_ind = ds_hybrid[i].indexes["time"]
        rmse_temp, rmse_per_temp, rmse_climo_temp = [], [], []
        for time in time_ind:
            rmse_temp.append(np.abs(ds_hybrid[i].sel(time=time).values - ds_observed.sel(time=time).values))
            rmse_per_temp.append(np.abs(ds_per[i].sel(time=time).values - ds_observed.sel(time=time).values))
            rmse_climo_temp.append(np.abs(ds_observed.sel(time=time).values))
        rmse.append(np.array(rmse_temp[:24]))
        rmse_per.append(np.array(rmse_per_temp[:24]))
        rmse_climo.append(np.array(rmse_climo_temp[:24]))
    rmse = np.mean(rmse, axis=0)
    rmse_per = np.mean(rmse_per, axis=0)
    rmse_climo = np.mean(rmse_climo, axis=0)

    return rmse, rmse_per, rmse_climo

def get_nino_index_rmse_new(startdates, prediction_length, timestep, lead_max=24):
    """Load in all observed, Hybrid, and SPEEDY predicted SST datasets. Calculate RMSE of nino indexes from each dataset.
    Inputs:
        startdates: list of datetime objects of prediction start dates.
        prediction_length: length of forecast (in hours)
        timestep: timestep of hybrid model
    Returns:
        ds_hybrid: [xarray DataArray] hybrid-predicted monthly nino3.4 indexes (3-month average).
        ds_observed: [xarray DataArray] observed monthly nino3.4 index (3-month averaged).
        ds_per: [xarray DataArray] persistence nino3.4 index."""

    hybrid_root = "/scratch/user/dpp94/Predictions/Hybrid/hybrid_prediction_era6000_20_20_20_sigma0.5_beta_res0.001_beta_model_1.0_prior_0.0_overlap1_vertlevel_1_precip_epsilon0.001_ohtc_multiple_leakage_test_oceantimestep_72hr_train1981_2002_oldcal__pred_newcal_trial_"

    obs_climo_ref_startdate = datetime(1981,1,1,0)
    obs_climo_ref_enddate = datetime(2002,12,1,0)  

    # Obtain observed nino index
    date, enddate = startdates[0] - timedelta(hours=24*60), startdates[-1] + timedelta(hours=prediction_length)
    ds_observed = get_obs_sst_timeseries(date, enddate, timestep)
    ds_observed_climo = get_obs_sst_timeseries(obs_climo_ref_startdate, obs_climo_ref_enddate, timestep)
    ds_observed = nino_index_monthly_specified_climo(ds_observed, "3.4", ds_observed_climo)
    ds_observed_unsmoothed = ds_observed["sst"]
    time_index = ds_observed["time"]
    ds_observed = uniform_filter1d(ds_observed["sst"].values, size=3, origin=1)
    ds_observed = xr.DataArray(ds_observed, dims="time", coords={"time": time_index.values}, name=date.strftime("%m_%d_%Y_%H"))
    print(f'observed SSTs: {ds_observed}')
    # Note, I believe groupby month sorts the result by calendar month
    # so that, ds_observed_unq[1] is for Jan, etc

    # Obtain data for calculating monthly climotology for predictions
    #ds_hybrid_climo = xr.open_dataset(hybrid_climo_ref_path)
    #ds_hybrid_climo = make_ds_time_dim(ds_hybrid_climo, timestep, hybrid_climo_ref_startdate)

    ds_hybrid = []
    ds_per = []
    for date in startdates:
        # Load in hybrid predictions
        date_str = date.strftime("%m_%d_%Y_%H")
        filepath = hybrid_root + date_str + ".nc"
        print(f'Loading in file at: {filepath}')
        ds_hybrid_temp = xr.open_dataset(filepath)
        ds_hybrid_temp = make_ds_time_dim(ds_hybrid_temp, timestep, date)
        ds_hybrid_temp = nino_index_monthly_specified_climo_hybrid(ds_observed_climo, "3.4", ds_hybrid_temp) #ds_hybrid_climo["SST"])
        time_index = ds_hybrid_temp["Timestep"]
        print(f'ds_hybrid_temp time_index: {time_index}\n')
        print(f'ds_observed_unsmoothed time_index: {ds_observed_unsmoothed["time"]}\n')
        date_temp = np.argwhere(ds_observed_unsmoothed["time"].values == time_index[0].values).squeeze() - 1
        print(f'date_temp, date_temp-1: {date_temp}, {date_temp-1}\n')
        print(f'\nds_observed_unsmoothed.isel(): {ds_observed_unsmoothed.isel(time=slice(date_temp-1,date_temp)).values}\n')
        ds_hybrid_temp = np.concatenate((ds_observed_unsmoothed.isel(time=slice(date_temp-1,date_temp+1)).values, ds_hybrid_temp['sst'].values))
        ds_hybrid_temp = uniform_filter1d(ds_hybrid_temp, size=3, origin=1) #ds_hybrid_temp['sst'].values
        print(f'len(ds_hybrid_temp): {len(ds_hybrid_temp)}\n')
        ds_hybrid_temp = ds_hybrid_temp[2:]
        ds_hybrid_temp = xr.DataArray(ds_hybrid_temp, dims="time", coords={"time": time_index.values}, name=date_str)
        #ds_hybrid_temp.coords["lead"] = ("time", np.arange(ds_hybrid_temp.size))
        print(f'\n\nds_hybrid_temp: {ds_hybrid_temp}\n\n')

        # Load in observed SST for persistence
        #print(f'ds_observed["time"].values: {ds_observed["time"].values}\n')
        #print(f'time_index[0]: {time_index[0].values}\n')
        ds_per_temp_time = np.argwhere(ds_observed["time"].values == time_index[0].values).squeeze() - 1
        print(f'\n\ndata start date: {date}\npersistence start date: {ds_observed["time"].isel(time=ds_per_temp_time)}\n')
        ds_per_temp = xr.ones_like(ds_hybrid_temp) * ds_observed.isel(time=ds_per_temp_time)
        #ds_per_temp.coords["lead"] = ("time", np.arange(ds_per_temp.size))
        print(f'\n\nds_per_temp: {ds_per_temp}\n\n')

        ds_hybrid.append(ds_hybrid_temp)
        ds_per.append(ds_per_temp)

    rmse, rmse_per, rmse_climo = [], [], []
    std, std_per, std_climo = [], [], []
    bias = []
    var = []
    for i in range(lead_max):
        rmse_temp, rmse_per_temp, rmse_climo_temp = [], [], []
        bias_temp, var_temp = [], []
        for j in range(len(ds_hybrid)):
            time_ind = ds_hybrid[j].indexes["time"].values[i]
            rmse_temp.append(ds_hybrid[j].sel(time=time_ind).values - ds_observed.sel(time=time_ind).values)
            rmse_per_temp.append(ds_per[j].sel(time=time_ind).values - ds_observed.sel(time=time_ind).values)
            rmse_climo_temp.append(ds_observed.sel(time=time_ind).values)
            bias_temp.append(rmse_temp[-1])
            var_temp.append(ds_hybrid[j].sel(time=time_ind).values)
        rmse.append(np.sqrt(np.mean(np.square(rmse_temp))))
        rmse_per.append(np.sqrt(np.mean(np.square(rmse_per_temp))))
        rmse_climo.append(np.sqrt(np.mean(np.square(rmse_climo_temp))))
        std.append(np.std(rmse_temp))
        std_per.append(np.std(rmse_per_temp))
        std_climo.append(np.std(rmse_climo_temp))
        bias.append(np.mean(bias_temp))
        var.append(np.std(bias_temp))    #var.append(np.var(var_temp))

    return np.array(rmse), np.array(rmse_per), np.array(rmse_climo), np.array(std), np.array(std_per), np.array(std_climo), np.array(bias), np.array(var)

def get_nino34_mean_std_vs_lead_time(ds_hybrid, ds_obs):
    """Get mean and var vs lead time."""

    hybrid_mean, obs_mean = [], []
    hybrid_var, obs_var = [], []
    leadtimes = ds_hybrid.groupby("lead").groups.keys()
    for lead in leadtimes:
        ds_hybrid_temp = ds_hybrid.groupby("lead")[lead]
        #print(f'ds_hybrid_temp: {ds_hybrid_temp}\n')
        #print(f'ds_hybrid_temp["time"]: {ds_hybrid_temp["time"]}\n')
        ds_obs_temp = ds_obs.sel(time=ds_hybrid_temp['time']).values
        #print(f'ds_obs_temp: {ds_obs_temp}\n')
        ds_hybrid_temp = ds_hybrid_temp.values

        hybrid_mean.append(np.mean(ds_hybrid_temp))
        obs_mean.append(np.mean(ds_obs_temp))
        hybrid_var.append(np.var(ds_hybrid_temp))
        obs_var.append(np.var(ds_obs_temp))

    return hybrid_mean, hybrid_var, obs_mean, obs_var

def plot_nino34_forecasts_vs_leadtime(ds_hybrid, ds_obs):
    """Plot the 1) forecast values vs lead time, and 2) signed forecast 
       errors vs lead time."""

    vals, errors = [], []
    leadtimes = ds_hybrid.groupby("lead").groups.keys()
    for lead in leadtimes:
        ds_hybrid_temp = ds_hybrid.groupby("lead")[lead]
        ds_obs_temp = ds_obs.sel(time=ds_hybrid_temp['time']).values
        ds_hybrid_temp = ds_hybrid_temp.values

        vals.append(ds_hybrid_temp)
        errors.append(ds_hybrid_temp - ds_obs_temp)

    # Plot forecast values vs lead time
    fig, ax = plt.subplots(figsize=(11,8.5))
    ens = len(vals[0])
    for i in range(ens):
        data = ds_hybrid.sel(ens=i).dropna('time').values
        ax.plot(np.arange(1,1+len(data)),data,'-k',alpha=0.05)
        ax.plot(np.arange(1,1+len(data)),data,'ok',markerfacecolor=(1,1,1,1))
    for i, lead in enumerate(leadtimes):
        vp = ax.violinplot(vals[i],[lead+1],showmeans=True)
        vp['bodies'][-1].set_color((1,0,0,0.5))
        vp['cmeans'].set_color((1,0,0,0.5))
        vp['cbars'].set_color((1,0,0,0.5)) 
    ax.set_xlim([0,25])
    ax.set_ylim([-4,5])
    ax.set_xlabel('Lead time (months)', fontsize=16)
    ax.set_ylabel('Forecast Values', fontsize=16)
    ax.grid()
    plt.tight_layout()
    plt.show()

    # Plot forecast error vs lead time
    fig, ax = plt.subplots(figsize=(11,8.5))
    for i, lead in enumerate(leadtimes):
        x = [lead+1]*len(errors[i])
        ax.scatter(x,errors[i],c='k')
        vp = ax.violinplot(errors[i],[lead+1],showmeans=True)
        vp['bodies'][-1].set_color((1,0,0,0.5))
        vp['cmeans'].set_color((1,0,0,0.5))
        vp['cbars'].set_color((1,0,0,0.5))
    ax.set_xlim([0,25])
    ax.set_ylim([-6,6])
    ax.set_xlabel('Lead time (months)', fontsize=16)
    ax.set_ylabel('Forecast Error', fontsize=16)
    ax.grid()
    plt.tight_layout()
    plt.show() 

def get_acc(ds_hybrid, ds_obs):
    """Calculate the temporal anamoly correlation coefficient vs lead time.
    Inputs:
       ds_hybrid: [xarray DataArray] hybrid-predicted monthly nino3.4 indexes.
       ds_obs: [xarray DataArray] observed monthly nino3.4 index.
    Returns:
       leadtimes: lead times (in months) 
       C: temporal anomaly correlation coefficient."""
    
    leadtimes = ds_hybrid.groupby("lead").groups.keys()
    months = ds_hybrid.groupby("time.month").groups.keys()
    years = ds_hybrid.groupby("time.year").groups.keys()
    print(f'Lead time index: {leadtimes}\n')
    print(f'Months: {months}\n')
    print(f'Years: {years}\n')
    C = []
    for lead in leadtimes:
        frac = 0
        for month in months:
            print(f'lead: {lead}, month: {month}\n')
            Y_m = ds_obs.groupby("time.month")[month]
            Ybar_m = Y_m.mean("time").values
            print(f'Ybar_m: {Ybar_m}\n')
            P_lm = ds_hybrid.groupby("lead")[lead].groupby("time.month")[month]
            Pbar_lm = P_lm.mean("stacked_ens_time").values
            print(f'Pbar_lm: {Pbar_lm}\n')
            numm = 0
            denom_Y, denom_P = 0, 0
            for year in years:
                try:
                    print(f'lead: {lead}, month: {month}, year: {year}\n')
                    dY_ym = np.mean(Y_m.groupby("time.year")[year].values) - Ybar_m
                    dP_ylm = np.mean(P_lm.groupby("time.year")[year].values) - Pbar_lm
                    numm += dY_ym * dP_ylm
                    denom_Y += dY_ym**2
                    denom_P += dP_ylm**2
                except:
                    pass
            denom = np.sqrt(denom_Y * denom_P)
            print(f'numerator, denominator: {numm}, {denom}\n')
            frac += numm / denom
        C.append(frac/12)

    return leadtimes, np.array(C)

def get_pearson_corr(startdates,prediction_length,timestep,lead_max=24):
    """Obtain pearson correlation coefficient."""
 
    hybrid_root = "/scratch/user/dpp94/Predictions/Hybrid/hybrid_prediction_era6000_20_20_20_sigma0.5_beta_res0.001_beta_model_1.0_prior_0.0_overlap1_vertlevel_1_precip_epsilon0.001_ohtc_multiple_leakage_test_oceantimestep_72hr_train1981_2002_oldcal__pred_newcal_trial_"

    obs_climo_ref_startdate = datetime(1981,1,1,0)
    obs_climo_ref_enddate = datetime(2002,12,1,0)

    # Obtain observed nino index
    date, enddate = startdates[0] - timedelta(hours=24*60), startdates[-1] + timedelta(hours=prediction_length)
    ds_observed = get_obs_sst_timeseries(date, enddate, timestep)
    ds_observed_climo = get_obs_sst_timeseries(obs_climo_ref_startdate, obs_climo_ref_enddate, timestep)
    ds_observed = nino_index_monthly_specified_climo(ds_observed, "3.4", ds_observed_climo)
    ds_observed_unsmoothed = ds_observed["sst"]
    print(f'i\nds_observed_unsmoothed: {ds_observed_unsmoothed}\n')
    time_index = ds_observed["time"]
    ds_observed = uniform_filter1d(ds_observed["sst"].values, size=3, origin=1)
    ds_observed = xr.DataArray(ds_observed, dims="time", coords={"time": time_index.values}, name=date.strftime("%m_%d_%Y_%H"))
    print(f'observed SSTs: {ds_observed}')
    # Note, I believe groupby month sorts the result by calendar month
    # so that, ds_observed_unq[1] is for Jan, etc

    # Obtain data for calculating monthly climotology for predictions
    #ds_hybrid_climo = xr.open_dataset(hybrid_climo_ref_path)
    #ds_hybrid_climo = make_ds_time_dim(ds_hybrid_climo, timestep, hybrid_climo_ref_startdate)

    ds_hybrid = []
    ds_per = []
    for date in startdates:
        # Load in hybrid predictions
        date_str = date.strftime("%m_%d_%Y_%H")
        filepath = hybrid_root + date_str + ".nc"
        print(f'Loading in file at: {filepath}')
        ds_hybrid_temp = xr.open_dataset(filepath)
        ds_hybrid_temp = make_ds_time_dim(ds_hybrid_temp, timestep, date)
        ds_hybrid_temp = nino_index_monthly_specified_climo_hybrid(ds_observed_climo, "3.4", ds_hybrid_temp) #ds_hybrid_climo["SST"])
        time_index = ds_hybrid_temp["Timestep"]
        print(f'ds_hybrid_temp time_index: {time_index}\n')
        print(f'ds_observed time_index: {ds_observed_unsmoothed["time"]}\n')
        date_temp = np.argwhere(ds_observed_unsmoothed["time"].values == time_index[0].values).squeeze() - 1
        print(f'date_temp, date_temp-1: {date_temp}, {date_temp-1}\n')
        print(f'\nds_observed_unsmoothed.isel(): {ds_observed_unsmoothed.isel(time=slice(date_temp-1,date_temp)).values}\n')
        ds_hybrid_temp = np.concatenate((ds_observed_unsmoothed.isel(time=slice(date_temp-1,date_temp+1)).values, ds_hybrid_temp['sst'].values))
        ds_hybrid_temp = uniform_filter1d(ds_hybrid_temp, size=3, origin=1) #ds_hybrid_temp['sst'].values
        print(f'len(ds_hybrid_temp): {len(ds_hybrid_temp)}\n')
        ds_hybrid_temp = ds_hybrid_temp[2:]
        ds_hybrid_temp = xr.DataArray(ds_hybrid_temp, dims="time", coords={"time": time_index.values}, name=date_str)
        #ds_hybrid_temp.coords["lead"] = ("time", np.arange(ds_hybrid_temp.size))
        print(f'\n\nds_hybrid_temp: {ds_hybrid_temp}\n\n')

        # Load in observed SST for persistence
        #print(f'ds_observed["time"].values: {ds_observed["time"].values}\n')
        #print(f'time_index[0]: {time_index[0].values}\n')
        ds_per_temp_time = np.argwhere(ds_observed["time"].values == time_index[0].values).squeeze() - 1
        print(f'\n\ndata start date: {date}\npersistence start date: {ds_observed["time"].isel(time=ds_per_temp_time)}\n')
        ds_per_temp = xr.ones_like(ds_hybrid_temp) * ds_observed.isel(time=ds_per_temp_time)
        #ds_per_temp.coords["lead"] = ("time", np.arange(ds_per_temp.size))
        print(f'\n\nds_per_temp: {ds_per_temp}\n\n')

        ds_hybrid.append(ds_hybrid_temp)
        ds_per.append(ds_per_temp)

    C, C_per = [], []
    std, std_per = [], []
    #time_ind = ds_hybrid[0].indexes["time"].values
    for i in range(lead_max):  
        C_temp, C_per_temp, C_obs_temp = [], [], []
        for j in range(len(ds_hybrid)):
            time_ind = ds_hybrid[j].indexes["time"].values[i]
            C_temp.append(ds_hybrid[j].sel(time=time_ind).values)
            C_per_temp.append(ds_per[j].sel(time=time_ind).values)
            C_obs_temp.append(ds_observed.sel(time=time_ind).values)
        print(f'\n\n pearsonr(): {pearsonr(C_temp,C_obs_temp)}\n')
        pr = pearsonr(C_temp,C_obs_temp)
        pr_per = pearsonr(C_per_temp,C_obs_temp)
        ci = pr.confidence_interval() #get_pearsonr_ci(C_temp,C_obs_temp)
        ci_per = pr_per.confidence_interval() #get_pearsonr_ci(C_per_temp,C_obs_temp)
        C.append(pr.statistic)
        C_per.append(pr_per.statistic)
        std.append((ci.low, ci.high))
        std_per.append((ci_per.low, ci_per.high))       

    return np.array(C), np.array(C_per), std, std_per

def decode_cf(ds, time_var):
    """Decodes time dimension to CFTime standards."""

    if ds[time_var].attrs["calendar"] == "360":
       ds[time_var].attrs["calendar"] = "360_day"
    ds = xr.decode_cf(ds, decode_times=True)
    return ds

def get_anoms(ds, ds_clim):

    # get monthly clim
    ds_clim = ds_clim.groupby('time.month').mean(dim='time')
    #print(f'ds_clim: {ds_clim}\n')

    # get anoms
    ds = ds.groupby('time.month') - ds_clim
    #print(f'ds_anom: {ds}')

    return ds

def get_acc_nmme(x, y):
    """
    x, y: [num forecasts, lead times]
    """

    leads = x.shape[1]
    acc = []
    for lead in range(leads):
        acc.append(np.corrcoef(x[:, lead], y[:, lead])[0, 1])
    
    return acc

def get_rmse_nmme(x, y):
    """
    x, y: [num forecasts, lead times]
    """
    
    leads = x.shape[1]
    rmse = []
    for lead in range(leads):
        err = (x[:, lead] - y[:, lead]) ** 2
        rmse.append(np.sqrt(np.mean(err)))
    
    return rmse

def get_avg_nmme_skill(models, startdates, nmme_startdate, nmme_enddate):

    # Lat/Lon 
    lat_slice = slice(-5, 5)
    lon_slice = slice(360-170, 360-150)

    # Get ERA5 climatology / Nino 3.4 index
    obs_climo_ref_startdate = datetime(1981,1,1,0)
    obs_climo_ref_enddate = datetime(2002,12,1,0)
    timestep = 6

    ds_observed_climo = get_obs_sst_timeseries(obs_climo_ref_startdate, obs_climo_ref_enddate, timestep)['sst']
    ds_observed_unsmoothed = ds_observed_climo
    date, enddate = startdates[0] - timedelta(hours=24*60), startdates[-1] + timedelta(hours=prediction_length)
    ds_observed = get_obs_sst_timeseries(date, enddate, timestep)
    ds_verif_nino = nino_index_monthly_specified_climo(ds_observed, "3.4", ds_observed_climo)
    time_index = ds_verif_nino["time"]
    print(f'ds_verif_nino["time"]: {ds_verif_nino["time"]}\n')
    ds_verif_nino = uniform_filter1d(ds_verif_nino["sst"].values, size=3, origin=1)
    ds_verif_nino = xr.DataArray(ds_verif_nino, dims="time", coords={"time": time_index.values}, name=date.strftime("%m_%d_%Y_%H"))

    # reformat time dimensions to match nmme data
    date  = ds_verif_nino["time"].values[0]
    year = date.astype('datetime64[Y]').astype(int) + 1970
    month = date.astype('datetime64[M]').astype(int) % 12 + 1
    date_cft = cftime.DatetimeGregorian(year, month, 1)
    print(f'date_cft: {date_cft}\n')
    ds_verif_nino["time"] = xr.cftime_range(start=date_cft, periods=ds_verif_nino["time"].size, freq="MS", calendar="360_day")

    # Spatially average and temporally resample climo data and reformat dime dim
    ds_observed_climo = ds_observed_climo.sel(lon=lon_slice, lat=lat_slice).mean(dim=['lon', 'lat'])
    ds_observed_climo = ds_observed_climo.resample(time="1MS").mean(dim="time")
    ds_observed_climo["time"] = xr.cftime_range(start="1981-01", periods=ds_observed_climo["time"].size, freq="MS", calendar="360_day")
    print(f'ds_observed_climo reformat time: {ds_observed_climo}\n')

    # Data URLs
    url_1 = 'http://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/'
    url_2 = '/.HINDCAST/.MONTHLY/.sst/X/190/240/RANGEEDGES/Y/-5/5/RANGEEDGES/[X%20Y%20M]average/dods'

    accs = {}
    rmses = {}
    biases = {}
    varr = {}
    for model in models:
        print(f'Fetching model: {model}\n')

        # Fetch data
        url = url_1 + model + url_2
        fcstds = xr.open_dataset(url, decode_times=False)
        fcstds = decode_cf(fcstds, 'S').compute()['sst']

        # reformat dimensions/attributes according to climpred convention
        fcstds = fcstds.rename({"S": "init", "L": "lead"})
        fcstds["lead"] = (fcstds["lead"] - 0.5).astype("int")
        fcstds["lead"].attrs = {"units": "months"}

        # Select appropriate subset of verification data
        fcstds = fcstds.sel(init=slice(nmme_startdate, nmme_enddate))

        preds = []
        obs = []
        # Calculate acc of each forecast
        init_dates = fcstds['init'].values
        leads = fcstds['lead'].values
        for date in init_dates:
            # assign the valid_dates to 'lead' coordinate values
            year, month, day = date.year, date.month, date.day
            init_date = datetimedate(year, month, day) + relativedelta(months=+1) 
            valid_dates = pd.date_range(start=init_date, periods=len(leads), freq='MS')
            ds = fcstds.sel(init=date)
            ds = ds.assign_coords(lead=valid_dates)

            # rename 'lead' as 'time'
            ds = ds.rename({'lead': 'time'})

            # get appropriate subset of verification data
            obs_anom = ds_verif_nino.sel(time=valid_dates)
            print(f'valid_dates: {valid_dates}\n')
            print(f'obs_anom: {obs_anom}\n')
            obs_anom = obs_anom.values

            # get anomalies of prediction
            ds_anom = get_anoms(ds, ds_observed_climo).values
            print(f'ds_anom: {ds_anom}\n')
            print(f'ds_anom.shape obs_anom.shape: {ds_anom.shape}, {obs_anom.shape}')
            # perform 3-month avg
            # prepend true 2 months of nino3.4vals, smooth, remove true 2 months
            padding_dates = pd.date_range(start=init_date+relativedelta(months=-1), periods=2, freq='MS')
            padding = ds_verif_nino.sel(time=padding_dates).values
            print(f'padding size: {len(padding)}\n')
            ds_anom = np.concatenate([padding, ds_anom])
            ds_anom = uniform_filter1d(ds_anom, size=3, origin=1)
            ds_anom = ds_anom[2:]

            preds.append(ds_anom)
            obs.append(obs_anom)
        
        preds = np.array(preds)
        obs = np.array(obs)

        # get acc
        acc = get_acc_nmme(preds, obs)
        print(f'acc: {acc}')
        
        # get rmse
        rmse = get_rmse_nmme(preds, obs)

        # get bias
        bias = np.mean(preds - obs, axis=0) 

        # get var
        var = np.std(preds - obs, axis=0)

        accs[model] = acc
        rmses[model] = rmse
        biases[model] = bias
        varr[model] = var
    return accs, rmses, biases, varr

def get_pearson_corr_mjo_precip(startdates,prediction_length,timestep,lead_max=10):
    """Calculate PCC skill of MJO-related WP precip vs lead time."""
   
    hybrid_root = "/scratch/user/dpp94/Predictions/Hybrid/hybrid_prediction_era6000_20_20_20_sigma0.5_beta_res0.001_beta_model_1.0_prior_0.0_overlap1_vertlevel_1_precip_epsilon0.001_ohtc_multiple_leakage_test_oceantimestep_72hr_train1981_2002_oldcal__pred_newcal_trial_" 

    lats = slice(-15,15)
    lons = slice(360-180,360-130) #WP:(360-180,360-130),EP:(130,180)

    obs_climo_ref_startdate = datetime(1981,1,1,0)
    obs_climo_ref_enddate = datetime(2002,12,1,0)

    # Get observed precip climatology
    ds_observed_climo = get_obs_precip_timeseries(obs_climo_ref_startdate,obs_climo_ref_enddate,1)['tp'] * 39.37
    # Go from 1hr accumulated to daily accumulated
    ds_observed_climo = ds_observed_climo.resample(Timestep='1D').sum(dim='Timestep')
    # pentad precip anom for climo
    ds_climo_anom = get_anom_specified_climo_hybrid_daily(ds_observed_climo,ds_observed_climo)
    ds_climo_anom = ds_climo_anom.rolling(Timestep=5,center=False).sum().dropna(dim='Timestep')
    ds_climo_anom = ds_climo_anom.sel(Lat=lats,Lon=lons).mean(dim=['Lat','Lon'])

    # Obtain anomalies of forecasts
    ds_hybrid_all, ds_obs_all, ds_per_all = [], [], []
    for t, startdate in enumerate(startdates):
        # Read in hybrid forecast, calculate anomaly, and retain up to lead_max
        date_str = startdate.strftime("%m_%d_%Y_%H")
        filepath = hybrid_root + date_str + ".nc"
        #print(f'Loading in file at: {filepath}\n')
        ds_hybrid = xr.open_dataset(filepath)
        ds_hybrid = make_ds_time_dim(ds_hybrid, timestep, startdate)
        ds_hybrid = ds_hybrid['p6hr']
        # Go from 6hr accumulated to daily accumulated
        ds_hybrid = ds_hybrid.resample(Timestep='1D').sum(dim='Timestep')
        ds_hybrid_anom = get_anom_specified_climo_hybrid_daily(ds_observed_climo, ds_hybrid)
        #print(f'daily ds_hybrid_anom: {ds_hybrid_anom}\n')
        ds_hybrid_anom = ds_hybrid_anom.rolling(Timestep=5,center=False).sum().dropna(dim='Timestep')
        print(f'pentad ds_hybrid_anom: {ds_hybrid_anom}\n')
        ds_hybrid_anom = ds_hybrid_anom.isel(Timestep=slice(0,lead_max-5,1))
        ds_hybrid_anom = ds_hybrid_anom.sel(Lat=lats,Lon=lons).mean(dim=['Lat','Lon'])

        # Read in observed precip, calculate anomaly
        date_i, date_f = startdate, startdate + timedelta(hours=prediction_length) 
        ds_obs = get_obs_precip_timeseries(date_i,date_f,1)['tp'] * 39.37
        # Go from 1hr accumulated to daily accumulated
        ds_obs = ds_obs.resample(Timestep='1D').sum(dim='Timestep')
        ds_obs_anom = get_anom_specified_climo_hybrid_daily(ds_observed_climo, ds_obs)
        #print(f'daily ds_obs_anom: {ds_obs_anom}\n')
        ds_obs_anom = ds_obs_anom.rolling(Timestep=5,center=False).sum().dropna(dim='Timestep')
        print(f'pentad ds_obs_anom: {ds_obs_anom}\n')
        ds_obs_anom = ds_obs_anom.sel(Lat=lats,Lon=lons).mean(dim=['Lat','Lon'])
        ds_obs_anom = ds_obs_anom.isel(Timestep=slice(0,lead_max+1,1))
  
        # Persistence forecast
        date_i, date_f = startdate - timedelta(days=5), startdate + timedelta(hours=prediction_length)
        ds_per = get_obs_precip_timeseries(date_i,date_f,1)['tp'] * 39.37
        # Go from 1hr accumulated to daily accumulated
        ds_per = ds_per.resample(Timestep='1D').sum(dim='Timestep')
        ds_per_anom = get_anom_specified_climo_hybrid_daily(ds_observed_climo, ds_per)
        ds_per_anom = ds_per_anom.rolling(Timestep=5,center=False).sum().dropna(dim='Timestep')
        ds_per_anom = ds_per_anom.sel(Lat=lats,Lon=lons).mean(dim=['Lat','Lon'])
        ds_per_anom = ds_per_anom.isel(Timestep=0)
        print(f'pentad ds_per_anom: {ds_per_anom}\n')

        ds_hybrid_all.append(ds_hybrid_anom)
        ds_obs_all.append(ds_obs_anom)
        ds_per_all.append(ds_per_anom)

    # Calculate PCC
    C, C_per = [], []
    std, std_per = [], []
    C_all, C_obs_all = [], []
    for i in range(lead_max-5):
        C_temp, C_per_temp, C_obs_temp = [], [], []
        for j in range(len(ds_hybrid_all)):
            time_ind = ds_hybrid_all[j].indexes["Timestep"].values[i]
            C_temp.append(ds_hybrid_all[j].sel(Timestep=time_ind).values)
            C_per_temp.append(ds_per_all[j].values)
            C_obs_temp.append(ds_obs_all[j].sel(Timestep=time_ind).values)
        C_temp = np.array(C_temp).flatten()
        C_per_temp = np.array(C_per_temp).flatten()
        C_obs_temp = np.array(C_obs_temp).flatten()
        #print(f'C_temp, where nan/inf: {C_temp},\n {np.where(np.isnan(C_temp))}\n')
        #print(f'C_per_temp, where nan/inf: {C_per_temp},\n {np.where(np.isnan(C_per_temp))}\n')
        #print(f'C_obs_temp, where nan/inf: {C_obs_temp},\n {np.where(np.isnan(C_obs_temp))}\n')
        idx = np.argwhere(~np.isnan(C_obs_temp)).flatten()
        bad_idx = np.argwhere(C_obs_temp<-1)
        #print(f'bad_idx: {bad_idx}\n')
        #print(f'C, C_obs at bad idx: {C_temp[bad_idx]}, {C_obs_temp[bad_idx]}\n')
        #print(f'idx: {idx}\n') 
        C_temp = C_temp[idx] 
        C_obs_temp = C_obs_temp[idx]
        C_per_temp = C_per_temp[idx]
        C_all.append(C_temp)
        C_obs_all.append(C_obs_temp) 
        pr = pearsonr(C_temp,C_obs_temp)
        pr_per = pearsonr(C_per_temp,C_obs_temp)
        ci = pr.confidence_interval()
        ci_per = pr_per.confidence_interval()
        C.append(pr.statistic)
        C_per.append(pr_per.statistic)
        std.append((ci.low, ci.high))
        std_per.append((ci_per.low, ci_per.high))
   
    #print(f'bad idx: {bad_idx}\n')
    #print(f'bad date: {ds_hybrid_all[bad_idx[0][0]].indexes["Timestep"].values[0]}\n')
    return np.array(C), np.array(C_per), std, std_per, C_all, C_obs_all, ds_climo_anom

def plot_pentad_precip_maps(startdates,prediction_length,timestep,lead_max=14):
    """Plot spatial maps of pentad precip anomalies."""

    hybrid_root = "/scratch/user/dpp94/Predictions/Hybrid/hybrid_prediction_era6000_20_20_20_sigma0.5_beta_res0.001_beta_model_1.0_prior_0.0_overlap1_vertlevel_1_precip_epsilon0.001_ohtc_multiple_leakage_test_oceantimestep_72hr_train1981_2002_oldcal__pred_newcal_trial_" 

    obs_climo_ref_startdate = datetime(1981,1,1,0)
    obs_climo_ref_enddate = datetime(2002,12,1,0)

    # Get observed precip climatology
    ds_observed_climo = get_obs_precip_timeseries(obs_climo_ref_startdate,obs_climo_ref_enddate,1)['tp'] * 39.37
    # Go from 1hr accumulated to daily accumulated
    ds_observed_climo = ds_observed_climo.resample(Timestep='1D').sum(dim='Timestep')

    startdate = startdates[24]

    # Get hybrid prediction
    date_str = startdate.strftime("%m_%d_%Y_%H")
    filepath = hybrid_root + date_str + ".nc"
    print(f'Loading in file at: {filepath}\n')
    ds_hybrid = xr.open_dataset(filepath)
    ds_hybrid = make_ds_time_dim(ds_hybrid, timestep, startdate)
    ds_hybrid = ds_hybrid['p6hr']
    # Go from 6hr accumulated to daily accumulated
    ds_hybrid = ds_hybrid.resample(Timestep='1D').sum(dim='Timestep')
    ds_hybrid_anom = get_anom_specified_climo_hybrid_daily(ds_observed_climo, ds_hybrid)
    print(f'daily ds_hybrid_anom: {ds_hybrid_anom}\n')
    ds_hybrid_anom = ds_hybrid_anom.rolling(Timestep=5,center=False).sum().dropna(dim='Timestep')
    print(f'pentad ds_hybrid_anom: {ds_hybrid_anom}\n')
    ds_hybrid_anom = ds_hybrid_anom.isel(Timestep=slice(0,lead_max,1))

    # Read in observed precip, calculate anomaly
    date_i, date_f = startdate - timedelta(days=5), startdate + timedelta(hours=prediction_length)
    ds_obs = get_obs_precip_timeseries(date_i,date_f,1)['tp'] * 39.37 
    # Go from 1hr accumulated to daily accumulated
    ds_obs = ds_obs.resample(Timestep='1D').sum(dim='Timestep')
    ds_obs_anom = get_anom_specified_climo_hybrid_daily(ds_observed_climo, ds_obs)
    print(f'daily ds_obs_anom: {ds_obs_anom}\n')
    ds_obs_anom = ds_obs_anom.rolling(Timestep=5,center=False).sum().dropna(dim='Timestep')
    #print(f'pentad ds_obs_anom: {ds_obs_anom.isel(Timestep=slice(1,lead_max+1,1))}\n')
    ds_obs_anom = ds_obs_anom.isel(Timestep=slice(5,lead_max+1,1))
    print(f'pentad ds_obs_anom: {ds_obs_anom}\n')


    # Debug
    print(f'spatially-avg ds_hybrid_anom: {ds_hybrid_anom.isel(Timestep=0).sel(Lat=slice(-15,15)).mean(dim=["Lat","Lon"])}\n')
    print(f'spatially-avg ds_obs_anom: {ds_obs_anom.isel(Timestep=0).sel(Lat=slice(-15,15)).mean(dim=["Lat","Lon"])}\n')

    # Plotting
    ##ds_hybid_anom = ds_hybrid_anom.sel(Lat=slice(-15,15),Lon=slice(120,150))
    #ds_hybrid_anom = ds_hybrid_anom.where((ds_hybrid_anom.Lat>-15)&(ds_hybrid_anom.Lat<15)&(ds_hybrid_anom.Lon>0)&(ds_hybrid_anom.Lon<180),drop=True)
    ##ds_obs_anom = ds_obs_anom.sel(Lat=slice(-15,15),Lon=slice(120,150))
    #ds_obs_anom = ds_obs_anom.where((ds_obs_anom.Lat>-15)&(ds_obs_anom.Lat<15)&(ds_obs_anom.Lon>0)&(ds_obs_anom.Lon<180),drop=True)
    #print(f'ds_hybrid_anom: {ds_hybrid_anom}\n')
    #print(f'ds_obs_anom: {ds_obs_anom}\n')
    lons = ds_hybrid_anom.Lon.values
    lats = ds_hybrid_anom.Lat.values
    print(f'lons: {lons}\n')
    print(f'lats: {lats}\n')

    nrows, ncols = 4, 2
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols,
                            subplot_kw={'projection':ccrs.PlateCarree(central_longitude=-179)},
                            figsize=(11,8.5), sharex='col', sharey='row')

    axs = axs.flatten()
    
    for lead in range(4):
        tstep = lead * 3
        data = ds_hybrid_anom.isel(Timestep=tstep)
        data_obs = ds_obs_anom.isel(Timestep=tstep)
        #data_lons = lons 
        #data_obs_lons = lons
        data, data_lons = add_cyclic_point(data, coord=lons)
        data_obs, data_obs_lons = add_cyclic_point(data_obs, coord=lons)
        cmap = sns.color_palette("Spectral", as_cmap=True)
        cs = axs[2*lead].pcolormesh(data_lons,lats,data,
                                  transform=ccrs.PlateCarree(),
                                  cmap=cmap,vmin=-3,vmax=3)
        cs_obs = axs[2*lead+1].pcolormesh(data_obs_lons,lats,data_obs,
                                          transform=ccrs.PlateCarree(),
                                          cmap=cmap,vmin=-3,vmax=3)

        axs[2*lead].coastlines()
        axs[2*lead+1].coastlines()

        if lead == 0:
           axs[lead].set_title('Hybrid')
        if lead == 1:
           axs[lead].set_title('ERA5')
        
    cbar_ax = fig.add_axes([0.9,0.2,0.01,0.6]) #left,bottom,width,height
    cbar = fig.colorbar(cs, cax=cbar_ax, orientation='vertical', label='') 
    plt.show() 
 
def get_pearson_corr_mjo_precip_ecmwf(startdates,prediction_length,timestep,lead_max=10):
    """Calculate pentad mjo-related precip correlation for ecmwf hindcast."""
 
    filepath = "/scratch/user/troyarcomano/ecmwf_precip_hindcast.nc"

    lat_slice = slice(-10,10)
    lon_slice = slice(120,150)

    # Get observed precip climatology
    obs_climo_start = datetime(1981,1,1,0)
    obs_climo_end = datetime(2002,12,1,0)
    ds_obs_climo = get_obs_precip_timeseries(obs_climo_start,obs_climo_end,timestep)['tp']
    # Resample it to daily accumulated
    ds_obs_climo = ds_obs_climo.resample(Timestep='1D').sum(dim='Timestep')
    print(f'obs climo precip: {ds_obs_climo}\n')

    # Data parameters
    realization = 1
    ds_all = xr.open_dataset(filepath)
    ds_all = ds_all.sortby('forecast_time')
    
    ds_anom_all, ds_obs_anom_all = [], [] 
    for startdate in startdates:
        # Read in ECMWF data
        ds = ds_all.sel(forecast_time=startdate, method='nearest')
        ds = ds.rename({'valid_time':'Timestep','longitude':'Lon','latitude':'Lat'}) 
        ecmwf_dates = ds.coords["Timestep"]
        ecmwf_dates = pd.to_datetime(ecmwf_dates)
        ds = ds.assign_coords(coords={'Timestep':ecmwf_dates.values})
        print(f'ds nearest {startdate}: {ds}\n')
        
        # Read in observations
        date_i, date_f = ecmwf_dates[0], ecmwf_dates[0] + timedelta(hours=prediction_length)
        ds_obs = get_obs_precip_timeseries(date_i, date_f, timestep)['tp']
        ds_obs = ds_obs.resample(Timestep='1D').sum(dim='Timestep')  # go from 1hr accumulated to daily accumulated precip
        ds_obs_anom = get_anom_specified_climo_hybrid_daily(ds_obs_climo, ds_obs)
        ds_obs_anom = ds_obs_anom.resample(Timestep='5D').sum(dim='Timestep') 
        print(f'pentad ds_obs_anom: {ds_obs_anom}\n')
        lats = ds_obs.coords['Lat'].values
        lons = ds_obs.coords['Lon'].values 

        # Interpolate ECMWF data and convert units
        ds = ds.interp(Lat=lats, Lon=lons, method='linear')
        ds = ds * 0.001 # (1 kg/m2, ECMWF units, = 1 mm of rain, ERA5 units)
        ds = ds.diff(dim='Timestep')
        print(f'ds interpolated: {ds}\n')
        ds_anom = get_anom_specified_climo_hybrid_daily(ds_obs_climo, ds)
        ds_anom = ds_anom.resample(Timestep='5D').sum(dim='Timestep')
        print(f'ds_anom: {ds_anom}\n')

        ## Debug plotting
        #debug_obs = ds_obs.isel(Timestep=0).plot(subplot_kws=dict(projection=ccrs.PlateCarree()),
        #                                         transform=ccrs.PlateCarree(), vmin=0, vmax=0.075)
        #plt.show()
        #debug_ecmwf = ds['tp'].isel(realization=0).isel(lead_time=1).plot(subplot_kws=dict(projection=ccrs.PlateCarree()),
        #                                       transform=ccrs.PlateCarree()) #, vmin=0, vmax=0.075)
        #plt.show()
 
        # Get pentad mjo-precip 
        ds_obs_anom = ds_obs_anom.sel(Lat=lat_slice,Lon=lon_slice).mean(dim=['Lat','Lon'])
        ds_obs_anom = ds_obs_anom.isel(Timestep=slice(0,lead_max,1))
        print(f'pentad ds_obs_anom: {ds_obs_anom}\n')
        ds_anom = ds_anom.sel(Lat=lat_slice,Lon=lon_slice).mean(dim=['Lat','Lon']).mean(dim='realization')
        ds_anom = ds_anom.isel(Timestep=slice(0,lead_max,1))['tp']
        print(f'pentad ds_anom: {ds_anom}\n')

        ds_anom_all.append(ds_anom)
        ds_obs_anom_all.append(ds_obs_anom)

    # Calculate PCC
    C, std = [], []
    for i in range(lead_max):
        C_temp, C_obs_temp = [], []
        for j in range(len(ds_anom_all)):
            C_temp.append(ds_anom_all[j].isel(Timestep=i).values)
            C_obs_temp.append(ds_obs_anom_all[j].isel(Timestep=i).values)
        C_temp = np.array(C_temp).flatten()
        C_obs_temp = np.array(C_obs_temp).flatten()
        idx = np.argwhere(~np.isnan(C_obs_temp)).flatten()
        C_temp = C_temp[idx]
        C_obs_temp = C_obs_temp[idx]
        pr = pearsonr(C_temp,C_obs_temp)
        ci = pr.confidence_interval()
        C.append(pr.statistic)
        std.append((ci.low, ci.high))
  
    return np.array(C), std 

def get_pearsonr_ci(x, y, ci=95, n_boots=10000):
    x = np.asarray(x)
    y = np.asarray(y)
    
    # (n_boots, n_observations) paired arrays
    rand_ixs = np.random.randint(0, x.shape[0], size=(n_boots, x.shape[0]))
    x_boots = x[rand_ixs]
    y_boots = y[rand_ixs]
    
    # differences from mean
    x_mdiffs = x_boots - x_boots.mean(axis=1)[:, None]
    y_mdiffs = y_boots - y_boots.mean(axis=1)[:, None]
    
    # sums of squares
    x_ss = np.einsum('ij, ij -> i', x_mdiffs, x_mdiffs)
    y_ss = np.einsum('ij, ij -> i', y_mdiffs, y_mdiffs)
    
    # pearson correlations
    r_boots = np.einsum('ij, ij -> i', x_mdiffs, y_mdiffs) / np.sqrt(x_ss * y_ss)
    
    # upper and lower bounds for confidence interval
    ci_low = np.percentile(r_boots, (100 - ci) / 2)
    ci_high = np.percentile(r_boots, (ci + 100) / 2)
    return ci_low, ci_high

def get_ninoskill_lead_stmon(ds_hybrid, ds_obs):
    """Get nino 3.4 skill for lead time vs start month.
    Input: 
      ds_hybrid: [xarray DataArray] set of predicted nino indices.
      ds_obs:    [xarray DataArray] actual nino indices.
    Return:
      pcorr: [np.array] matrix of correlation values (lead time vs start month)."""
    
    # Add dimension to "ens" with values being the forecast start month
    ens = ds_hybrid["ens"]
    startmonth = []
    for i in ens:
        data = ds_hybrid.sel(ens=i)
        data = data.dropna(dim="time")
        startmonth.append(data["time.month"][0].values)
    ds_hybrid["ens"] = startmonth
    print(f'\n\nds_hybrid: {ds_hybrid}\n\n')	
  
    leads = ds_hybrid.groupby("lead").groups.keys() #isel(ens=0)["lead"].dropna(dim="time")
    print(f'\nds_hybrid.groupby("lead").groups.keys(): {ds_hybrid.groupby("lead").groups.keys()}\n')
    startmonths = ds_hybrid.groupby("ens").groups.keys() #np.arange(1,13,1)
    print(f'startmonths: {startmonths}\n')
    pcorr = np.zeros((12,len(leads)))
    for i, lead in enumerate(leads):
        hybrid_lead = ds_hybrid.groupby("lead")[lead]
        hybrid_lead = hybrid_lead.groupby("ens") 
        for j, month in enumerate(startmonths):
            hybrid_lead_mon = hybrid_lead[month]
            print(f'\nhybrid_lead_mon: {hybrid_lead_mon}\n')
            truth = []
            time_ind = hybrid_lead_mon["time"]
            for time in time_ind:
                truth.append(ds_obs.sel(time=time).values)
            print(f'\ntruth: {truth}\n')
            pcorr[j,i] = pearsonr(hybrid_lead_mon.values,truth).statistic
    return pcorr
   
def plot_ninoskill_contour(pcorr, pcorr_per=None):
    """Make contour plot of ninoskill for lead time vs start month.
    Input: 
      pcorr: [np.array] [12 x lead_max] matrix of correlation coefficients."""

    lead_max = pcorr.shape[1]
    fig, ax = plt.subplots()
    img = ax.contourf(pcorr, levels=np.arange(0,1.01,0.1), extend="min", cmap="RdBu_r")
    ct = ax.contour(pcorr, [0.5, 0.6, 0.7, 0.8, 0.9], colors="k", linewidths=1)
    #ax.clabel(ct, fontsize=8, colors="k", fmt="%.1f")
    if pcorr_per is not None:
       ax.contour(pcorr_per, levels=[0.5], colors="k", linewidth=1, linestyles="dotted")
    ax.set_xticks(np.array([1, 5, 10, 15, 20]) - 1)
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.set_xticklabels(np.array([1,5,10,15,20]))
    ax.set_xlabel("Prediction lead (months)", fontsize=16)
    ax.set_yticks(np.arange(0,12,1))
    y_ticklabels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", \
                    "Sep", "Oct", "Nov", "Dec"]
    ax.set_yticklabels(y_ticklabels)
    ax.set_ylabel("Month", fontsize=16)
    ax.set_xlim([0, 9])  #lead_max-1])
    ax.tick_params(axis='both', which='major', labelsize=16)
    cbar = plt.colorbar(img, extend='min')
    cbar.set_label('PCC')
    plt.tight_layout()
    #plt.show()
    fig.savefig('ENSO_skill_startmonth_leadtime.svg')

def get_sst_rmse(startdates, prediction_length, timestep, ocean_timestep):
    """Load in all observed, and Hybrid, predicted SST datasets. Calculate RMSE vs lead time for all datasets.
    Inputs:
        startdates: list of datetime objects of prediction start dates.
        prediction_length: length of forecast (in hours)
        timestep: timestep of hybrid model
        ocean_timestep: timestep of ocean component
    Returns:
        time_vec: [list] prediction lead time (# of weeks)
        rmse: [list] list of lists of SST rmse vs lead time .
        rmse_per: [list] list of lists of persistence SST rmse vs lead time.
        rmse_climo: [list] list of lists of climo SST rmse vs lead time."""
        

    hybrid_root = "/scratch/user/dpp94/Predictions/Hybrid/hybrid_prediction_era6000_20_20_20_sigma0.5_beta_res0.001_beta_model_1.0_prior_0.0_overlap1_vertlevel_1_precip_epsilon0.001_ohtc_multiple_leakage_test_oceantimestep_72hr_trial_"


    lat_slice = slice(-5,5)
    lon_slice = slice(360-170,360-120)

    # Get SST climo
    obs_climo_ref_startdate = datetime(1981,1,1,0)
    obs_climo_ref_enddate = datetime(2006,12,1,0)
    ds_obs_climo = get_obs_sst_timeseries(obs_climo_ref_startdate, obs_climo_ref_enddate, timestep)['sst']
    ds_obs_climo = ds_obs_climo.sel(lat=lat_slice, lon=lon_slice)
    ds_obs_climo = ds_obs_climo.where(ds_obs_climo > 272.0) #.mean(dim=['lat','lon'], skipna=True)
    print(f'ds_obs_climo: {ds_obs_climo}\n\n')
    ds_obs_climo = ds_obs_climo.groupby("time.dayofyear").mean('time')

    rmse = []
    rmse_per = []
    rmse_climo = []
    for date in startdates:
        # Load in hybrid predictions
        date_str = date.strftime("%m_%d_%Y_%H")
        filepath = hybrid_root + date_str + ".nc"
        ds_hybrid = xr.open_dataset(filepath)
        ds_hybrid = make_ds_time_dim(ds_hybrid, timestep, date)['SST'].sel(Lat=lat_slice,Lon=lon_slice)
        ds_hybrid = ds_hybrid.where(ds_hybrid > 272.0) #.mean(dim=['Lat','Lon'], skipna=True)
        print(f'date: {date}\n')
        print(f'ds_climo: {ds_obs_climo.sel(dayofyear=ds_hybrid.indexes["Timestep"].dayofyear[0]).mean(dim=["lat","lon"])}\n')
        print(f'ds_hybrid: {ds_hybrid.sel(Timestep=date).mean(dim=["Lat","Lon"])}\n')
        
        # Load in observed SST
        startdate = date - timedelta(hours=2*ocean_timestep) 
        enddate = date + timedelta(hours=prediction_length) 
        ds_obs = get_obs_sst_timeseries(startdate, enddate, timestep)['sst'].sel(lat=lat_slice,lon=lon_slice)
        ds_obs = ds_obs.where(ds_obs > 272.0) #.mean(dim=['lat','lon'], skipna=True)
        print(f'ds_obs: {ds_obs.sel(time=date).mean(dim=["lat","lon"])}\n')
        ds_obs = ds_obs.rolling(time=int(ocean_timestep/timestep)).mean('time')
        
        # Persistence forecast
        t = int(ocean_timestep/timestep) 
        ds_per = xr.ones_like(ds_obs) * ds_obs.isel(time=t) 
        print(f'ds_per: {ds_per.sel(time=date).mean(dim=["lat","lon"])}\n')

        rmse_temp = []
        rmse_per_temp = []
        rmse_climo_temp = []
        time_vec = []
        time_indx_obs = ds_obs.indexes["time"]
        print(f'time_indx_obs: {time_indx_obs}\n')
        time_indx_hybrid = ds_hybrid.indexes["Timestep"]
        counter = 1
        print(f'time_indx_hybrid[0:(8760*2/timestep)]: {time_indx_hybrid[0:int(8760*2/timestep)]}\n')
        for i, time in enumerate(time_indx_hybrid[0:int(8760*2/timestep)]):
            if (i%(ocean_timestep/timestep)==0):
               print(f'main loop i, time: {i}, {time}\n')
               climo = 0*ds_obs_climo.sel(dayofyear=time.dayofyear).values
               rmse_temp.append(rms(ds_hybrid.sel(Timestep=time).values, ds_obs.sel(time=time).values)) #-timedelta(hours=ocean_timestep)
               rmse_per_temp.append(rms(ds_per.sel(time=time).values, ds_obs.sel(time=time).values))
               rmse_climo_temp.append(rms(ds_obs_climo.sel(dayofyear=time.dayofyear).values, ds_obs.sel(time=time).values))
               #rmse_temp.append(latituded_weighted_rmse(ds_hybrid.sel(Timestep=time).values, ds_obs.sel(time=time).values, ds_obs.lat.values))
               #rmse_per_temp.append(latituded_weighted_rmse(ds_per.sel(time=time).values, ds_obs.sel(time=time).values, ds_obs.lat.values))
               #rmse_climo_temp.append(latituded_weighted_rmse(ds_obs_climo.sel(dayofyear=time.dayofyear).values, ds_obs.sel(time=time).values, ds_obs.lat.values))
               time_vec.append(counter)
               counter += 1

        rmse.append(rmse_temp)
        rmse_per.append(rmse_per_temp)
        rmse_climo.append(rmse_climo_temp)
        print(f'rmse: {rmse_temp}\n')
        print(f'rmse_climo: {rmse_climo_temp}\n\n')

    time_vec = np.array(time_vec)
    #rmse = np.mean(np.array(rmse), axis=0)
    #rmse_per = np.mean(np.array(rmse_per), axis=0)
    #rmse_climo = np.mean(np.array(rmse_climo), axis=0)
    return time_vec, rmse, rmse_per, rmse_climo

def get_ohtc_rmse(startdates, prediction_length, timestep, ocean_timestep):
    """Load in all observed, and Hybrid, predicted ohtc datasets. Calculate RMSE vs lead time for all datasets.
    Inputs:
        startdates: list of datetime objects of prediction start dates.
        prediction_length: length of forecast (in hours)
        timestep: timestep of hybrid model
        ocean_timestep: timestep of ocean component
    Returns:
        time_vec: [list] prediction lead time (# of weeks)
        rmse: [list] list of lists of SST rmse vs lead time .
        rmse_per: [list] list of lists of persistence ohtc rmse vs lead time.
        rmse_climo: [list] list of lists of climo ohtc rmse vs lead time."""


    hybrid_root = "/scratch/user/dpp94/Predictions/Hybrid/hybrid_prediction_era6000_20_20_20_sigma0.5_beta_res0.001_beta_model_1.0_prior_0.0_overlap1_vertlevel_1_precip_epsilon0.001_ohtc_multiple_leakage_test_oceantimestep_72hr_trial_"


    lat_slice = slice(-5,5) #(-5,5)
    lon_slice = slice(360-170,360-120) 

    # Get SST climo
    obs_climo_ref_startdate = datetime(1981,1,1,0)
    obs_climo_ref_enddate = datetime(2006,12,1,0)
    ds_obs_climo = get_obs_ohtc_timeseries(obs_climo_ref_startdate, obs_climo_ref_enddate, timestep)['sohtc300']
    ds_obs_climo = ds_obs_climo.sel(lat=lat_slice, lon=lon_slice)
    ds_obs_climo = ds_obs_climo.where(ds_obs_climo > 272.0) #.mean(dim=['lat','lon'], skipna=True)
    print(f'ds_obs_climo: {ds_obs_climo}\n\n')
    ds_obs_climo = ds_obs_climo.groupby("time_counter.dayofyear").mean('time_counter')

    rmse = []
    rmse_per = []
    rmse_climo = []
    for date in startdates:
        # Load in hybrid predictions
        date_str = date.strftime("%m_%d_%Y_%H")
        filepath = hybrid_root + date_str + ".nc"
        ds_hybrid = xr.open_dataset(filepath)
        ds_hybrid = make_ds_time_dim(ds_hybrid, timestep, date)['sohtc300'].sel(Lat=lat_slice,Lon=lon_slice)
        ds_hybrid = ds_hybrid.where(ds_hybrid > 272.0) #.mean(dim=['Lat','Lon'], skipna=True)
        print(f'date: {date}\n')
        print(f'ds_climo: {ds_obs_climo.sel(dayofyear=ds_hybrid.indexes["Timestep"].dayofyear[0]).mean(dim=["lat","lon"])}\n')
        print(f'ds_hybrid: {ds_hybrid.sel(Timestep=date).mean(dim=["Lat","Lon"])}\n')
 
        # Load in observed SST
        startdate = date - timedelta(hours=2*ocean_timestep)
        enddate = date + timedelta(hours=prediction_length)
        ds_obs = get_obs_ohtc_timeseries(startdate, enddate, timestep)['sohtc300'].sel(lat=lat_slice,lon=lon_slice)
        ds_obs = ds_obs.where(ds_obs > 272.0) #.mean(dim=['lat','lon'], skipna=True)
        print(f'ds_obs: {ds_obs.sel(time_counter=date).mean(dim=["lat","lon"])}\n')
        ds_obs = ds_obs.rolling(time_counter=int(ocean_timestep/timestep)).mean('time_counter')

        # Persistence forecast
        t = int(ocean_timestep/timestep)
        ds_per = xr.ones_like(ds_obs) * ds_obs.isel(time_counter=t)
        print(f'ds_per: {ds_per.sel(time_counter=date).mean(dim=["lat","lon"])}\n')

        rmse_temp = []
        rmse_per_temp = []
        rmse_climo_temp = []
        time_vec = []
        time_indx_obs = ds_obs.indexes["time_counter"]
        print(f'time_indx_obs: {time_indx_obs}\n')
        time_indx_hybrid = ds_hybrid.indexes["Timestep"]
        counter = 1
        print(f'time_indx_hybrid[0:(8760*2/timestep)]: {time_indx_hybrid[0:int(8760*2/timestep)]}\n')
        for i, time in enumerate(time_indx_hybrid[0:int(8760*2/timestep)]):
            if (i%(ocean_timestep/timestep)==0):
               print(f'main loop i, time: {i}, {time}\n')
               #climo = 0*ds_obs_climo.sel(dayofyear=time.dayofyear).values
               rmse_temp.append(rms(ds_hybrid.sel(Timestep=time).values, ds_obs.sel(time_counter=time).values)) #-timedelta(hours=ocean_timestep)
               rmse_per_temp.append(rms(ds_per.sel(time_counter=time).values, ds_obs.sel(time_counter=time).values))
               rmse_climo_temp.append(rms(ds_obs_climo.sel(dayofyear=time.dayofyear).values, ds_obs.sel(time_counter=time).values))
               #rmse_temp.append(latituded_weighted_rmse(ds_hybrid.sel(Timestep=time).values, ds_obs.sel(time=time).values, ds_obs.lat.values))
               #rmse_per_temp.append(latituded_weighted_rmse(ds_per.sel(time=time).values, ds_obs.sel(time=time).values, ds_obs.lat.values))
               #rmse_climo_temp.append(latituded_weighted_rmse(ds_obs_climo.sel(dayofyear=time.dayofyear).values, ds_obs.sel(time=time).values, ds_obs.lat.values))
               time_vec.append(counter)
               counter += 1

        rmse.append(rmse_temp)
        rmse_per.append(rmse_per_temp)
        rmse_climo.append(rmse_climo_temp)
        print(f'rmse: {rmse_temp}\n')
        print(f'rmse_climo: {rmse_climo_temp}\n\n')

    time_vec = np.array(time_vec)
    #rmse = np.mean(np.array(rmse), axis=0)
    #rmse_per = np.mean(np.array(rmse_per), axis=0)
    #rmse_climo = np.mean(np.array(rmse_climo), axis=0)
    return time_vec, rmse, rmse_per, rmse_climo

def get_seasonal_spatial_corr_nino34_precip(startdate,prediction_length,timestep,ocean_timestep):
    """Obtain the seasonal spatial correlation between nino3.4 index and precipitation."""

    hybrid_root = "/scratch/user/troyarcomano/Predictions/Hybrid/hybrid_prediction_era6000_20_20_20_sigma0.5_beta_res0.001_beta_model_1.0_prior_0.0_overlap1_vertlevel_1_precip_epsilon0.001_multi_gaussian_noise_newest_version_32_processors_root_ssttrial_" #"/scratch/user/dpp94/Predictions/Hybrid/hybrid_prediction_era6000_20_20_20_sigma0.5_beta_res0.001_beta_model_1.0_prior_0.0_overlap1_vertlevel_1_precip_epsilon0.001_ohtc_multiple_leakage_test_trial_"

    lat_slice = slice(None,None) #(-50,50)
    lon_slice = slice(None,None)

    obs_climo_ref_startdate = datetime(1981,1,1,0)
    obs_climo_ref_enddate = datetime(2008,12,31,0)

    # Obtain observed sst climotology
    ds_sst_observed_climo = get_obs_sst_timeseries(obs_climo_ref_startdate, obs_climo_ref_enddate, timestep)
    print(f'observed sst climo: {ds_sst_observed_climo}\n\n')

    # Obtain observed precip climotology
    ds_precip_observed_climo = get_obs_precip_timeseries(obs_climo_ref_startdate, obs_climo_ref_enddate, timestep)
    print(f'observed precip climo: {ds_precip_observed_climo}\n\n')

    # Obtain hybrid prediction
    date_str = startdate.strftime("%m_%d_%Y_%H")
    filepath = hybrid_root + date_str + ".nc"
    print(f'Loading in file at: {filepath}')
    ds_hybrid = xr.open_dataset(filepath)
    ds_hybrid = make_ds_time_dim(ds_hybrid, timestep, startdate)

    # Obtain hybrid nino index
    ds_hybrid_nino = nino_index_monthly_specified_climo_hybrid(ds_sst_observed_climo, "3.4", ds_hybrid)
    ds_hybrid_nino = nino_index_monthly_specified_climo(ds_sst_observed_climo, "3.4", ds_sst_observed_climo) 
    time_index = ds_hybrid_nino['time'] #["Timestep"]
    ds_hybrid_nino = uniform_filter1d(ds_hybrid_nino['sst'].values, size=1) #origin=1
    print(f'ds_hybrid_nino: {ds_hybrid_nino}\n')
    ds_hybrid_nino = xr.DataArray(ds_hybrid_nino, dims="Timestep", coords={"Timestep": time_index.values}, name=date_str)
    ds_hybrid_nino = ds_hybrid_nino.groupby("Timestep.season")
    print(f'\n\nGrouped by season ds_hybrid_nino: {ds_hybrid_nino}\n\n')
    #print(f'ds_hybrid_nino["DJF"]: {ds_hybrid_nino["DJF"]}\n')

    # Obtain hybrid precip anomalies
    ds_hybrid_precip = ds_hybrid["p6hr"]
    print(f'ds_hybrid_precip: {ds_hybrid_precip}\n')
    ds_hybrid_precip_anom = precip_anom_specified_climo_hybrid(ds_precip_observed_climo, ds_precip_observed_climo)['tp'] #ds_hybrid_precip)['tp']
    print(f'ds_hybrid_precip_anom: {ds_hybrid_precip_anom}\n')
    ds_hybrid_precip_anom = ds_hybrid_precip_anom.groupby("Timestep.season")
    print(f'Grouped by season ds_hybrid_precip_anom: {ds_hybrid_precip_anom}\n')

    # Obtain seasonal correlations
    seasonal_corr = {}
    seasons = ds_hybrid_precip_anom.groups
    for season, months in seasons.items():
        seasonal_corr[season] = xr.corr(ds_hybrid_precip_anom[season], ds_hybrid_nino[season], dim="Timestep")
    print(seasonal_corr)

    # Plot results
    lons = seasonal_corr['DJF'].Lon.values
    lats = seasonal_corr['DJF'].Lat.values
    print(f'lons: {lons}\n')

    projection = ccrs.PlateCarree(central_longitude=-179)
    axes_class = (GeoAxes, dict(map_projection=projection))
    plt.rc('font', family='serif')
    plt.rcParams['figure.constrained_layout.use'] = True

    fig = plt.figure(figsize=(6,10))
    axgr = AxesGrid(fig, 111, axes_class=axes_class,
                    nrows_ncols=(2,2),
                    axes_pad=0.7,
                    cbar_location='right',
                    cbar_mode='single',
                    cbar_pad=0.2,
                    cbar_size='3%',
                    label_mode='')  # note the empty label_mode

    for i, season in enumerate(seasons):
        data = seasonal_corr[season].values
        cyclic_data, cyclic_lons = add_cyclic_point(data, coord=lons)
        lons2d, lats2d = np.meshgrid(cyclic_lons,lats)
        ax = axgr[i]
        ax.coastlines()
        ax.set_xticks([-180, -120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree())
        ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_label=False, \
                          linewidth=2, color='gray', alpha=0.5, linestyle='--')
        levels = [-1.0,-0.8,-0.6,-0.4,-0.2,0.0,0.2,0.4,0.6,0.8,1.0] #[0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6]
        temp_colormap = sns.color_palette("Spectral", as_cmap=True)
        temp_colormap = temp_colormap.reversed()
        cmap = temp_colormap
        plot  = ax.contourf(lons2d,lats2d,cyclic_data,transform=ccrs.PlateCarree(),levels=levels,cmap=cmap,extend='both')
        cbar = axgr.cbar_axes[i].colorbar(plot, extend='both')
        cbar.set_ticks(levels)
        cbar.set_label('Correlation',fontsize=16)
        cbar.ax.tick_params(labelsize=16)
        ax.add_feature(cart.feature.LAND, zorder=100, edgecolor='k', facecolor='#808080')
        ax.set_title(season, fontsize=18,fontweight='bold')
        ax.set_extent([-180,180,-90,90], crs=ccrs.PlateCarree())

    plt.show()
    
def get_spatial_corr_nino34_precip(startdate,prediction_length,timestep,ocean_timestep):
    """Obtain the seasonal spatial correlation between nino3.4 index and precipitation."""

    hybrid_root = "/scratch/user/dpp94/Predictions/Hybrid/hybrid_prediction_era6000_20_20_20_sigma0.5_beta_res0.001_beta_model_1.0_prior_0.0_overlap1_vertlevel_1_precip_epsilon0.001_ohtc_multiple_leakage_test_leak_0.1_1_70yr_trial_" #hybrid_prediction_era6000_20_20_20_sigma0.5_beta_res0.001_beta_model_1.0_prior_0.0_overlap1_vertlevel_1_precip_epsilon0.001_multi_gaussian_noise_newest_version_32_processors_root_ssttrial_" 

    lat_slice = slice(None,None) #(-50,50)
    lon_slice = slice(None,None)

    obs_climo_ref_startdate = datetime(1981,1,1,0)
    obs_climo_ref_enddate = datetime(2008,12,31,0)

    # Obtain observed sst climotology
    ds_sst_observed_climo = get_obs_sst_timeseries(obs_climo_ref_startdate, obs_climo_ref_enddate, timestep)
    print(f'observed sst climo: {ds_sst_observed_climo}\n\n')

    # Obtain observed precip climotology
    ds_precip_observed_climo = get_obs_precip_timeseries(obs_climo_ref_startdate, obs_climo_ref_enddate, timestep)
    print(f'observed precip climo: {ds_precip_observed_climo}\n\n')

    # Obtain hybrid prediction
    date_str = startdate.strftime("%m_%d_%Y_%H")
    filepath = hybrid_root + date_str + ".nc"
    print(f'Loading in file at: {filepath}')
    ds_hybrid = xr.open_dataset(filepath)
    ds_hybrid = make_ds_time_dim(ds_hybrid, timestep, startdate)

    # Obtain hybrid nino index
    ds_hybrid_nino = nino_index_monthly_specified_climo_hybrid(ds_sst_observed_climo, "3.4", ds_hybrid) #for observed data corr
    #ds_hybrid_nino = nino_index_monthly_specified_climo(ds_sst_observed_climo, "3.4", ds_sst_observed_climo) #for just ERA5 corr
    time_index = ds_hybrid_nino["Timestep"] #['time'] for just ERA5 corr
    ds_hybrid_nino = uniform_filter1d(ds_hybrid_nino['sst'].values, size=1) #origin=1
    print(f'ds_hybrid_nino: {ds_hybrid_nino}\n')
    ds_hybrid_nino = xr.DataArray(ds_hybrid_nino, dims="Timestep", coords={"Timestep": time_index.values}, name=date_str)

    # Obtain hybrid precip anomalies
    ds_hybrid_precip = ds_hybrid["p6hr"]
    print(f'ds_hybrid_precip: {ds_hybrid_precip}\n')
    ds_hybrid_precip_anom = precip_anom_specified_climo_hybrid(ds_precip_observed_climo, ds_hybrid_precip)['tp'] #ds_precip_observed_climo)['tp'] for just ERA5 corr
    print(f'ds_hybrid_precip_anom: {ds_hybrid_precip_anom}\n')

    # Obtain correlations
    data = xr.corr(ds_hybrid_precip_anom, ds_hybrid_nino, dim="Timestep")
    print(data)

    # Plot results
    lons = data.Lon.values
    lats = data.Lat.values
    print(f'lons: {lons}\n')

    projection = ccrs.PlateCarree(central_longitude=-179)
    axes_class = (GeoAxes, dict(map_projection=projection))
    plt.rc('font', family='serif')
    plt.rcParams['figure.constrained_layout.use'] = True

    fig = plt.figure(figsize=(6,10))
    axgr = AxesGrid(fig, 111, axes_class=axes_class,
                  nrows_ncols=(1,1),
                  axes_pad=0.7,
                  cbar_location='right',
                  cbar_mode='single',
                  cbar_pad=0.2,
                  cbar_size='3%',
                  label_mode='')  # note the empty label_mode
    
    cyclic_data, cyclic_lons = add_cyclic_point(data, coord=lons)
    lons2d, lats2d = np.meshgrid(cyclic_lons,lats)
    ax = axgr[0]
    ax.coastlines()
    ax.set_xticks([-180, -120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree())
    ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_label=False, \
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')
    levels = [-1.0,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1.0] #[0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6]
    temp_colormap = sns.color_palette("Spectral", as_cmap=True)
    temp_colormap = temp_colormap.reversed()
    cmap = temp_colormap
    plot  = ax.contourf(lons2d,lats2d,cyclic_data,transform=ccrs.PlateCarree(),levels=levels,cmap=cmap,extend='both')
    cbar = axgr.cbar_axes[0].colorbar(plot, extend='both')
    cbar.set_ticks(levels)
    cbar.set_label('Correlation',fontsize=16)
    cbar.ax.tick_params(labelsize=16)
    #ax.add_feature(cart.feature.LAND, zorder=100, edgecolor='k', facecolor='#808080')
    ax.set_title('ENSO and Precip Anomaly correlation', fontsize=18,fontweight='bold')
    ax.set_extent([-180,180,-90,90], crs=ccrs.PlateCarree())

    plt.show()

    return data

def get_spatial_corr_nino34_precip_era(startdate,prediction_length,timestep,ocean_timestep):
    """Obtain the seasonal spatial correlation between nino3.4 index and precipitation."""

    hybrid_root = "/scratch/user/dpp94/Predictions/Hybrid/hybrid_prediction_era6000_20_20_20_sigma0.5_beta_res0.001_beta_model_1.0_prior_0.0_overlap1_vertlevel_1_precip_epsilon0.001_ohtc_multiple_leakage_test_leak_0.1_1_70yr_trial_" #hybrid_prediction_era6000_20_20_20_sigma0.5_beta_res0.001_beta_model_1.0_prior_0.0_overlap1_vertlevel_1_precip_epsilon0.001_multi_gaussian_noise_newest_version_32_processors_root_ssttrial_" 

    lat_slice = slice(None,None) #(-50,50)
    lon_slice = slice(None,None)

    obs_climo_ref_startdate = datetime(1981,1,1,0)
    obs_climo_ref_enddate = datetime(2008,12,31,0)

    # Obtain observed sst climotology
    ds_sst_observed_climo = get_obs_sst_timeseries(obs_climo_ref_startdate, obs_climo_ref_enddate, timestep)
    print(f'observed sst climo: {ds_sst_observed_climo}\n\n')

    # Obtain observed precip climotology
    ds_precip_observed_climo = get_obs_precip_timeseries(obs_climo_ref_startdate, obs_climo_ref_enddate, timestep)
    print(f'observed precip climo: {ds_precip_observed_climo}\n\n')

    # Obtain hybrid prediction
    date_str = startdate.strftime("%m_%d_%Y_%H")
    filepath = hybrid_root + date_str + ".nc"
    print(f'Loading in file at: {filepath}')
    ds_hybrid = xr.open_dataset(filepath)
    ds_hybrid = make_ds_time_dim(ds_hybrid, timestep, startdate)

    # Obtain hybrid nino index
    ds_hybrid_nino = nino_index_monthly_specified_climo(ds_sst_observed_climo, "3.4", ds_sst_observed_climo) 
    time_index = ds_hybrid_nino['time'] 
    ds_hybrid_nino = uniform_filter1d(ds_hybrid_nino['sst'].values, size=1) #origin=1
    print(f'ds_hybrid_nino: {ds_hybrid_nino}\n')
    ds_hybrid_nino = xr.DataArray(ds_hybrid_nino, dims="Timestep", coords={"Timestep": time_index.values}, name=date_str)

    # Obtain hybrid precip anomalies
    ds_hybrid_precip = ds_hybrid["p6hr"]
    print(f'ds_hybrid_precip: {ds_hybrid_precip}\n')
    ds_hybrid_precip_anom = precip_anom_specified_climo_hybrid(ds_precip_observed_climo, ds_precip_observed_climo)['tp'] 
    print(f'ds_hybrid_precip_anom: {ds_hybrid_precip_anom}\n')

    # Obtain correlations
    data = xr.corr(ds_hybrid_precip_anom, ds_hybrid_nino, dim="Timestep")
    print(data)

    # Plot results
    lons = data.Lon.values
    lats = data.Lat.values
    print(f'lons: {lons}\n')

    projection = ccrs.PlateCarree(central_longitude=-179)
    axes_class = (GeoAxes, dict(map_projection=projection))
    plt.rc('font', family='serif')
    plt.rcParams['figure.constrained_layout.use'] = True

    fig = plt.figure(figsize=(6,10))
    axgr = AxesGrid(fig, 111, axes_class=axes_class,
                  nrows_ncols=(1,1),
                  axes_pad=0.7,
                  cbar_location='right',
                  cbar_mode='single',
                  cbar_pad=0.2,
                  cbar_size='3%',
                  label_mode='')  # note the empty label_mode

    cyclic_data, cyclic_lons = add_cyclic_point(data, coord=lons)
    lons2d, lats2d = np.meshgrid(cyclic_lons,lats)
    ax = axgr[0]
    ax.coastlines()
    ax.set_xticks([-180, -120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree())
    ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_label=False, \
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')
    levels = [-1.0,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1.0] #[0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6]
    temp_colormap = sns.color_palette("Spectral", as_cmap=True)
    temp_colormap = temp_colormap.reversed()
    cmap = temp_colormap
    plot  = ax.contourf(lons2d,lats2d,cyclic_data,transform=ccrs.PlateCarree(),levels=levels,cmap=cmap,extend='both')
    cbar = axgr.cbar_axes[0].colorbar(plot, extend='both')
    cbar.set_ticks(levels)
    cbar.set_label('Correlation',fontsize=16)
    cbar.ax.tick_params(labelsize=16)
    #ax.add_feature(cart.feature.LAND, zorder=100, edgecolor='k', facecolor='#808080')
    ax.set_title('ENSO and Precip Anomaly correlation', fontsize=18,fontweight='bold')
    ax.set_extent([-180,180,-90,90], crs=ccrs.PlateCarree())

    plt.show()

    return data

def get_spatial_corr_nino34_precip_diff(era_data,hybrid_data):

    data = era_data - hybrid_data

    # Plot results
    lons = data.Lon.values
    lats = data.Lat.values
    print(f'lons: {lons}\n')

    projection = ccrs.PlateCarree(central_longitude=-179)
    axes_class = (GeoAxes, dict(map_projection=projection))
    plt.rc('font', family='serif')
    plt.rcParams['figure.constrained_layout.use'] = True

    fig = plt.figure(figsize=(6,10))
    axgr = AxesGrid(fig, 111, axes_class=axes_class,
                  nrows_ncols=(1,1),
                  axes_pad=0.7,
                  cbar_location='right',
                  cbar_mode='single',
                  cbar_pad=0.2,
                  cbar_size='3%',
                  label_mode='')  # note the empty label_mode

    cyclic_data, cyclic_lons = add_cyclic_point(data, coord=lons)
    lons2d, lats2d = np.meshgrid(cyclic_lons,lats)
    ax = axgr[0]
    ax.coastlines()
    ax.set_xticks([-180, -120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree())
    ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_label=False, \
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')
    levels = [-1.0,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1.0] #[0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6]
    temp_colormap = sns.color_palette("Spectral", as_cmap=True)
    temp_colormap = temp_colormap.reversed()
    cmap = temp_colormap
    plot  = ax.contourf(lons2d,lats2d,cyclic_data,transform=ccrs.PlateCarree(),levels=levels,cmap=cmap,extend='both')
    cbar = axgr.cbar_axes[0].colorbar(plot, extend='both')
    cbar.set_ticks(levels)
    cbar.set_label('Correlation',fontsize=16)
    cbar.ax.tick_params(labelsize=16)
    #ax.add_feature(cart.feature.LAND, zorder=100, edgecolor='k', facecolor='#808080')
    ax.set_title('(ERA5 - Hybrid)', fontsize=18,fontweight='bold')
    ax.set_extent([-180,180,-90,90], crs=ccrs.PlateCarree())

    # Avg absoluter bias
    bias = np.fabs(data).mean()
    print(f'\n\nAvg absolute bias: {bias}\n\n')

    plt.show() 

def get_nino34_ds(startdates,prediction_length,timestep,outlen=9):
    """Return Python lists of observed and era5 ONI for forecasts."""

    hybrid_root = '/scratch/user/dpp94/Predictions/Hybrid/hybrid_prediction_era6000_20_20_20_sigma0.5_beta_res0.001_beta_model_1.0_prior_0.0_overlap1_vertlevel_1_precip_epsilon0.001_ohtc_multiple_leakage_test_oceantimestep_72hr_train1981_2002_oldcal__pred_newcal_trial_'

    obs_climo_ref_startdate = datetime(1981,1,1,0)
    obs_climo_ref_enddate = datetime(2002,12,31,0)

    # Obtain observed nino index
    date, enddate = startdates[0] - timedelta(hours=24*60), startdates[-1] + timedelta(hours=prediction_length)
    ds_observed = get_obs_sst_timeseries(date, enddate, timestep)
    ds_observed_climo = get_obs_sst_timeseries(obs_climo_ref_startdate, obs_climo_ref_enddate, timestep)
    ds_observed = nino_index_monthly_specified_climo(ds_observed, "3.4", ds_observed_climo)
    ds_observed_unsmoothed = ds_observed["sst"]
    time_index = ds_observed["time"]
    ds_observed = uniform_filter1d(ds_observed["sst"].values, size=3, origin=1)
    ds_observed = xr.DataArray(ds_observed, dims="time", coords={"time": time_index.values}, name=date.strftime("%m_%d_%Y_%H"))
    print(f'observed SSTs: {ds_observed}')
    # Note, I believe groupby month sorts the result by calendar month
    # so that, ds_observed_unq[1] is for Jan, etc
 
    # Obtain data for calculating monthly climotology for predictions
    #ds_hybrid_climo = xr.open_dataset(hybrid_climo_ref_path)
    #ds_hybrid_climo = make_ds_time_dim(ds_hybrid_climo, timestep, hybrid_climo_ref_startdate)

    ds_hybrid = []
    ds_obs = []
    for date in startdates:
        # Load in hybrid predictions
        date_str = date.strftime("%m_%d_%Y_%H")
        filepath = hybrid_root + date_str + ".nc"
        print(f'Loading in file at: {filepath}')
        ds_hybrid_temp = xr.open_dataset(filepath)
        ds_hybrid_temp = make_ds_time_dim(ds_hybrid_temp, timestep, date)
        ds_hybrid_temp = nino_index_monthly_specified_climo_hybrid(ds_observed_climo, "3.4", ds_hybrid_temp) #ds_hybrid_climo["SST"])
        time_index = ds_hybrid_temp["Timestep"]
        print(f'ds_hybrid_temp time_index: {time_index}\n')
        print(f'ds_observed time_index: {ds_observed_unsmoothed["time"]}\n')
        date_temp = np.argwhere(ds_observed_unsmoothed["time"].values == time_index[0].values).squeeze() - 1
        print(f'date_temp, date_temp-1: {date_temp}, {date_temp-1}\n')
        print(f'\nds_observed_unsmoothed.isel(): {ds_observed_unsmoothed.isel(time=slice(date_temp-1,date_temp)).values}\n')
        ds_hybrid_temp = np.concatenate((ds_observed_unsmoothed.isel(time=slice(date_temp-1,date_temp+1)).values, ds_hybrid_temp['sst'].values))
        ds_hybrid_temp = uniform_filter1d(ds_hybrid_temp, size=3, origin=1) #ds_hybrid_temp['sst'].values
        print(f'len(ds_hybrid_temp): {len(ds_hybrid_temp)}\n')
        ds_hybrid_temp = ds_hybrid_temp[2:]
        ds_hybrid_temp = xr.DataArray(ds_hybrid_temp, dims="time", coords={"time": time_index.values}, name=date_str)
        ds_hybrid_temp.coords["lead"] = ("time", np.arange(ds_hybrid_temp.size))
        #print(f'\n\nds_hybrid_temp: {ds_hybrid_temp}\n\n')

        # Load in observed SST
        ds_obs_temp = ds_observed.sel(time=time_index.values)
        print(f'ds_obs_temp: {ds_obs_temp}\n')

        ds_hybrid.append(ds_hybrid_temp.isel(time=slice(0,outlen)))
        ds_obs.append(ds_obs_temp.isel(time=slice(0,outlen)))
 
    # Concatenate predictions into single xarray DataArray
    ds_hybrid = xr.concat(ds_hybrid, dim="ens")
    ds_obs = xr.concat(ds_obs, dim='ens')

    return ds_hybrid, ds_obs

def calculate_plot_nino_precip_anom(startdates,ds_hybrid_all,ds_obs_all,ds_hybrid_nino_all,ds_obs_nino_all,var='p6hr',sigtest=True):
    """Calculate avg ONI-precip anom seasonal pcc.
       ds_hybrid_all, ds_obs_all are lists of spatial anom xarray DataArrays heatmaps.
       ds_hybrid_nino_all, ds_obs_nino_all are lists of univariate ONI xarray DataArrays.""" 

    num_dates = len(startdates)
    seasons = ['DJF','MAM','JJA','SON']
    startmonths = [12,3,6,9]
    
    lons = ds_hybrid_all[0].Lon.values
    lats = ds_hybrid_all[0].Lat.values

    nrows, ncols = 4, 2 #3
    fig, axs = plt.subplots(nrows=nrows,ncols=ncols,
                            subplot_kw={'projection': ccrs.PlateCarree(central_longitude=-179)},
                            figsize=(11,8.5), sharex='col', sharey='row')
    axs = axs.flatten()

    # Plot [era5, hybrid, diff] in 4x3
    # Loop over each season. 
    #    1. Calculate oni-precip corr over season (3 timesteps) for each forecast. Average corr across all forecasts. 
    #    OR
    # -> 2. Gather set of oni/precip values for season across all forecasts. Calculate corr between two sets.
    #    OR
    #    3. Resample oni/precip to seasonal for each forecast. Gather set of oni/precip values across all forecasts 
    #       at each start month. Calculate corr between sets.
    #    In each loop do era5, hybrid, and diff. Plot.
   
    for num_season, season in enumerate(seasons):
        samples_hybrid, samples_obs = [], []
        samples_hybrid_nino, samples_obs_nino = [], []
        for i in range(num_dates):
            ds_obs = ds_obs_all[i]
            ds_obs_nino = ds_obs_nino_all[i].rename({'time':'Timestep'})
            ds_hybrid = ds_hybrid_all[i]
            ds_hybrid_nino = ds_hybrid_nino_all[i].rename({'time':'Timestep'})
            ds_obs = ds_obs.where(ds_obs['Timestep'] >= startdates[i], drop=True)
            ds_hybrid = ds_hybrid.where(ds_hybrid['Timestep'] >= startdates[i], drop=True)
            ds_obs_nino = ds_obs_nino.where(ds_obs_nino['Timestep'] >= startdates[i], drop=True)
            ds_hybrid_nino = ds_hybrid_nino.where(ds_hybrid_nino['Timestep'] >= startdates[i], drop=True)
            print(f'ds_obs: {ds_obs}\n')
            print(f'ds_hybrid: {ds_hybrid}\n')
            print(f'ds_obs_nino: {ds_obs_nino}\n')
            print(f'ds_hybrid_nino: {ds_hybrid_nino}\n')
            if (startdates[i].month == startmonths[num_season]):
               print(f'PCC start month: {startdates[i].month}\n')
               samples_obs.append(ds_obs.isel(Timestep=2)) #slice(0,3)).mean(dim='Timestep'))
               samples_obs_nino.append(ds_obs_nino.isel(Timestep=2)) #slice(0,3)).mean(dim='Timestep'))
               samples_hybrid.append(ds_hybrid.isel(Timestep=2)) #slice(0,3)).mean(dim='Timestep'))
               samples_hybrid_nino.append(ds_hybrid_nino.isel(Timestep=2)) #slice(0,3)).mean(dim='Timestep'))
        samples_obs = xr.concat(samples_obs, dim='Timestep')
        samples_obs_nino = xr.concat(samples_obs_nino, dim='Timestep')
        samples_hybrid = xr.concat(samples_hybrid, dim='Timestep')
        samples_hybrid_nino = xr.concat(samples_hybrid_nino, dim='Timestep')
        print(f'samples_obs: {samples_obs}\n')
        print(f'samples_obs_nino: {samples_obs_nino}\n')
        pcc_obs = xr.corr(samples_obs, samples_obs_nino, dim='Timestep')
        pcc_hybrid = xr.corr(samples_hybrid, samples_hybrid_nino, dim='Timestep')
        pcc_diff = pcc_hybrid - pcc_obs
        pcc = [pcc_hybrid, pcc_obs, pcc_diff]
        print(f'pcc_obs: {pcc_obs}\n')
        print(f'pcc_hybrid: {pcc_hybrid}\n')
        if sigtest:
           pvals_obs = pearson_r_eff_p_value(samples_obs, samples_obs_nino, dim='Timestep')
           pvals_hybrid = pearson_r_eff_p_value(samples_hybrid, samples_hybrid_nino, dim='Timestep')
           pvals = [pvals_hybrid, pvals_obs]

        # Plotting routine
        for i, ax_num in enumerate([2*num_season, 2*num_season+1]): #[3*num_season, 3*num_season+1, 3*num_season+2]):
            data = pcc[i]
            data, data_lons = add_cyclic_point(data, coord=lons)
            levels = [-1,-0.8,-0.6,-0.4,-0.2,0.,0.2,0.4,0.6,0.8,1.]
            temp_colormap = sns.color_palette("Spectral", as_cmap=True)
            temp_colormap = temp_colormap.reversed()
            cmap = temp_colormap
            cs = axs[ax_num].pcolormesh(data_lons,lats,data,
                                        transform=ccrs.PlateCarree(),
                                        cmap=cmap,vmin=-1,vmax=1)
            if sigtest and (i<2):
               pvals_mask = pvals[i].where(pvals[i] < 0.05).roll(Lon=int(len(lons)/2)).values.flatten()
               print(f'pvals_mask.flatten(): {pvals_mask}\n')
               X, Y = np.meshgrid(lons,lats)
               colors = ["none" if np.isnan(pvals_mask[j]) else "black" for j in range(pvals_mask.size)]
               axs[ax_num].scatter(X, Y, s=0.25, marker='.', color=colors)
            title_pre = ['Hybrid', 'ERA5', 'Hybrid - ERA5']
            #axs[ax_num].set_title(f'{seasons[num_season]}: {title_pre[i]}')
            axs[ax_num].coastlines()
            if ax_num in [0,2,4,6]: #[0,3,6,9]:
               axs[ax_num].set_yticks([-60,-30,0,30,60], crs=ccrs.PlateCarree())
               lat_formatter = LatitudeFormatter()
               axs[ax_num].yaxis.set_major_formatter(lat_formatter)
            if ax_num in [6,7,]: #9,10,11]:
               axs[ax_num].set_xticks([-90,0,90,180],crs=ccrs.PlateCarree())
               lon_formatter = LongitudeFormatter(zero_direction_label=True)
               axs[ax_num].xaxis.set_major_formatter(lon_formatter)
             
    
    fig.subplots_adjust(bottom=0.2, top=0.9, left=0.38, right=0.88, #0.95,
                        wspace=0.05, hspace=0.15) # [left,right,bottom,top,width-white-space,height-white-space]
    cbar_ax = fig.add_axes([0.9, 0.2, 0.01, 0.6]) # [left,bottom,width,height]
    cbar = fig.colorbar(cs, cax=cbar_ax, orientation='vertical', label='r')
    #plt.show()

    fig.savefig('ONI_Precip_PCC.svg')

def calculate_plot_nino_precip_anom_all_season(startdates,ds_hybrid_all,ds_obs_all,ds_hybrid_nino_all,ds_obs_nino_all,var='p6hr',sigtest=True):
    """Calculate avg ONI-precip anom all-seasonal pcc.
       ds_hybrid_all, ds_obs_all are lists of spatial anom xarray DataArrays heatmaps.
       ds_hybrid_nino_all, ds_obs_nino_all are lists of univariate ONI xarray DataArrays."""

    num_dates = len(startdates)

    lons = ds_hybrid_all[0].Lon.values
    lats = ds_hybrid_all[0].Lat.values

    nrows, ncols = 1, 2 #3
    fig, axs = plt.subplots(nrows=nrows,ncols=ncols,
                            subplot_kw={'projection': ccrs.PlateCarree(central_longitude=-179)},
                            figsize=(11,8.5), sharex='col', sharey='row')
    axs = axs.flatten()

    # Plot [era5, hybrid, diff] in 4x3
    # Loop over each season.
    #    1. Calculate oni-precip corr over season (3 timesteps) for each forecast. Average corr across all forecasts.
    #    OR
    # -> 2. Gather set of oni/precip values for season across all forecasts. Calculate corr between two sets.
    #    OR
    #    3. Resample oni/precip to seasonal for each forecast. Gather set of oni/precip values across all forecasts
    #       at each start month. Calculate corr between sets.
    #    In each loop do era5, hybrid, and diff. Plot.

    samples_hybrid, samples_obs = [], []
    samples_hybrid_nino, samples_obs_nino = [], []
    for i in range(num_dates):
        print(f'startdate: {startdates[i]}\n')
        print(f'ds_obs["Timestep"]: {ds_obs_all[i]["Timestep"]}\n\n')
        ds_obs = ds_obs_all[i]
        ds_obs_nino = ds_obs_nino_all[i].rename({'time':'Timestep'})
        ds_hybrid = ds_hybrid_all[i]
        ds_hybrid_nino = ds_hybrid_nino_all[i].rename({'time':'Timestep'})
        ds_obs = ds_obs.where(ds_obs['Timestep'] >= startdates[i], drop=True)
        ds_hybrid = ds_hybrid.where(ds_hybrid['Timestep'] >= startdates[i], drop=True)
        ds_obs_nino = ds_obs_nino.where(ds_obs_nino['Timestep'] >= startdates[i], drop=True)
        ds_hybrid_nino = ds_hybrid_nino.where(ds_hybrid_nino['Timestep'] >= startdates[i], drop=True)
        #print(f'ds_obs: {ds_obs}\n')
        #print(f'ds_hybrid: {ds_hybrid}\n')
        #print(f'ds_obs_nino: {ds_obs_nino}\n')
        #print(f'ds_hybrid_nino: {ds_hybrid_nino}\n')
        # ... for concurrent ... isel(Timestep=2) for var
        # ... for lagged ...     isel(Timestep=5) for var
        samples_obs.append(ds_obs.isel(Timestep=5)) #slice(0,3)).mean(dim='Timestep'))
        samples_obs_nino.append(ds_obs_nino.isel(Timestep=2)) #slice(0,3)).mean(dim='Timestep'))
        samples_hybrid.append(ds_hybrid.isel(Timestep=5)) #slice(0,3)).mean(dim='Timestep'))
        samples_hybrid_nino.append(ds_hybrid_nino.isel(Timestep=2)) #slice(0,3)).mean(dim='Timestep'))
    samples_obs = xr.concat(samples_obs, dim='ens') #'Timestep')
    samples_obs_nino = xr.concat(samples_obs_nino, dim='ens') #'Timestep')
    samples_hybrid = xr.concat(samples_hybrid, dim='ens') #'Timestep')
    samples_hybrid_nino = xr.concat(samples_hybrid_nino, dim='ens') #'Timestep')
    print(f'samples_obs: {samples_obs}\n')
    print(f'samples_obs_nino: {samples_obs_nino}\n')
    print(f'samples_hybrid: {samples_hybrid}\n')
    pcc_obs = xr.corr(samples_obs, samples_obs_nino, dim='ens') #'Timestep')
    pcc_hybrid = xr.corr(samples_hybrid, samples_hybrid_nino, dim='ens') #'Timestep')
    pcc_diff = pcc_obs - pcc_hybrid
    pcc = [pcc_hybrid, pcc_obs, pcc_diff]
    print(f'pcc_obs: {pcc_obs}\n')
    print(f'pcc_hybrid: {pcc_hybrid}\n')
    if sigtest:
       pvals_obs = pearson_r_eff_p_value(samples_obs, samples_obs_nino, dim='ens') #'Timestep')
       pvals_hybrid = pearson_r_eff_p_value(samples_hybrid, samples_hybrid_nino, dim='ens') #'Timestep')
       pvals = [pvals_hybrid, pvals_obs]

    # Plotting routine
    for i in range(nrows*ncols):
        data = pcc[i]
        data, data_lons = add_cyclic_point(data, coord=lons)
        levels = [-1,-0.8,-0.6,-0.4,-0.2,0.,0.2,0.4,0.6,0.8,1.]
        temp_colormap = sns.color_palette("Spectral", as_cmap=True)
        temp_colormap = temp_colormap.reversed()
        cmap = temp_colormap
        cs = axs[i].pcolormesh(data_lons,lats,data,
                               transform=ccrs.PlateCarree(),
                               cmap=cmap,vmin=-1,vmax=1)
        if sigtest and (i<2):
           pvals_mask = pvals[i].where(pvals[i] < 0.05).roll(Lon=int(len(lons)/2)).values.flatten()
           print(f'pvals_mask.flatten(): {pvals_mask}\n')
           X, Y = np.meshgrid(lons,lats)
           colors = ["none" if np.isnan(pvals_mask[j]) else "black" for j in range(pvals_mask.size)]
           axs[i].scatter(X, Y, s=0.25, marker='.', color=colors)
        title_pre = ['Hybrid', 'ERA5', 'Hybrid - ERA5']
        axs[i].set_title(f'{title_pre[i]}')
        axs[i].coastlines()
        if i == 0:
           axs[i].set_yticks([-60,-30,0,30,60], crs=ccrs.PlateCarree())
           lat_formatter = LatitudeFormatter()
           axs[i].yaxis.set_major_formatter(lat_formatter)
        axs[i].set_xticks([-90,0,90,180],crs=ccrs.PlateCarree())
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        axs[i].xaxis.set_major_formatter(lon_formatter)


    fig.subplots_adjust(bottom=0.2, top=0.9, left=0.35, right=0.85,
                        wspace=0.05, hspace=0.25) # [left,right,bottom,top,width-white-space,height-white-space]
    cbar_ax = fig.add_axes([0.87, 0.2, 0.01, 0.6]) # [left,bottom,width,height]
    cbar = fig.colorbar(cs, cax=cbar_ax, orientation='vertical', label='r')
    #plt.show()

    fig.savefig('ONI_SST_PCC_lagged.svg')

def get_spatial_var_anom_ds_for_oni_corr(startdates,prediction_length,timestep,var,leadtime=12):
    """Return Python lists of xarray DataArrays of anomaly heatmaps for given variable.
       var: 'p6hr','SST','U-wind','logp'.
       leadtime: integer."""

    hybrid_root = '/scratch/user/dpp94/Predictions/Hybrid/hybrid_prediction_era6000_20_20_20_sigma0.5_beta_res0.001_beta_model_1.0_prior_0.0_overlap1_vertlevel_1_precip_epsilon0.001_ohtc_multiple_leakage_test_oceantimestep_72hr_train1981_2002_oldcal__pred_newcal_trial_'

    obs_climo_ref_startdate = datetime(1981,1,1,0)
    obs_climo_ref_enddate = datetime(2002,12,31,0)

    # Obtain observed var climotology
    if var == 'p6hr':
       ds_observed_climo = get_obs_precip_timeseries(obs_climo_ref_startdate,obs_climo_ref_enddate,1)['tp'] * 39.37
       ds_observed_climo = ds_observed_climo.resample(Timestep='6H').sum(dim='Timestep')
    elif var == 'SST':
       ds_observed_climo = get_obs_sst_timeseries(obs_climo_ref_startdate,obs_climo_ref_enddate,timestep)['sst'] #.astype('float32')
       ds_observed_climo = ds_observed_climo.rename({'time':'Timestep','lon':'Lon','lat':'Lat'})
    elif var == 'U-wind':
       ds_observed_climo = get_obs_atmo_timeseries_var(obs_climo_ref_startdate,obs_climo_ref_enddate,timestep,var,sigma_lvl=7)[var]
    elif var == 'logp':
       ds_observed_climo = get_obs_atmo_timeseries_var(obs_climo_ref_startdate,obs_climo_ref_enddate,timestep,var)[var]
       ds_observed_climo = 1000 * np.exp(ds_observed_climo)
    print(f'ds_observed_climo {var}: {ds_observed_climo}\n')  
  
    ds_hybrid_all, ds_observed_all = [], []
    for startdate in startdates:
        # Read in hybrid forecast, calculate anomaly, and retain only up to leadtime
        date_str = startdate.strftime("%m_%d_%Y_%H")
        filepath = hybrid_root + date_str + ".nc"
        print(f'Loading in file at: {filepath}')
        ds_hybrid = xr.open_dataset(filepath)
        ds_hybrid = make_ds_time_dim(ds_hybrid, timestep, startdate)
        ds_hybrid = ds_hybrid[var]
        if var == 'U-wind':
           ds_hybrid = ds_hybrid.sel(Sigma_Level=7)
        elif var == 'logp':
           ds_hybrid = 1000 * np.exp(ds_hybrid)
        elif var == 'SST':
           ds_observed_climo = ds_observed_climo.assign_coords(coords={'Lon':ds_hybrid.coords['Lon'].values,'Lat':ds_hybrid.coords['Lat'].values})
        print(f'ds_hybrid {var}: {ds_hybrid}\n')
        print(f'ds_observed_climo {var}: {ds_observed_climo}\n')
        # --- CHANGE get_anom FUNC BELOW FOR APPROPRIATE SAMPLING --- #
        ds_hybrid_anom = get_anom_specified_climo_hybrid(ds_observed_climo, ds_hybrid)
        ds_hybrid_anom = ds_hybrid_anom.isel(Timestep=slice(0,leadtime,1))
        #print(f'ds_hybrid_anom: {ds_hybrid_anom}\n')

        # Read in observed forecast, calculate anomaly, and retain only up to leadtime
        date_i, date_f = startdate - timedelta(hours=60*24), startdate + timedelta(hours=prediction_length)
        if var == 'p6hr':
           ds_observed = get_obs_precip_timeseries(date_i,date_f,1)['tp'] * 39.37
           ds_observed = ds_observed.resample(Timestep='6H').sum(dim='Timestep')
        elif var == 'SST':
           ds_observed = get_obs_sst_timeseries(date_i,date_f,timestep)['sst'] #.astype('float32')
           ds_observed = ds_observed.rename({'time':'Timestep','lon':'Lon','lat':'Lat'})
           ds_observed = ds_observed.assign_coords(coords={'Lon':ds_hybrid.coords['Lon'].values,'Lat':ds_hybrid.coords['Lat'].values})
        elif var == 'U-wind':
           ds_observed = get_obs_atmo_timeseries_var(date_i,date_f,timestep,var,sigma_lvl=7)[var]
        elif var == 'logp':
           ds_observed = get_obs_atmo_timeseries_var(date_i,date_f,timestep,var)[var]
           ds_observed = 1000 * np.exp(ds_observed_climo)
        # --- CHANGE get_anom FUNC BELOW FOR APPROPRIATE SAMPLING --- #
        ds_observed_anom = get_anom_specified_climo_hybrid(ds_observed_climo, ds_observed)
        #print(f'ds_observed_anom: {ds_observed_anom}\n')
        
        # Prepend ds_hybrid with ds_obs and take rolling of BOTH obs and hybrid
        ds_obs_prepend = ds_observed_anom.isel(Timestep=slice(0,2,1))
        print(f'ds_obs_prepend: {ds_obs_prepend}\n')
        ds_hybrid_anom = xr.concat([ds_obs_prepend, ds_hybrid_anom], dim='Timestep')
        ds_hybrid_anom = ds_hybrid_anom.rolling(Timestep=3, center=False).sum().isel(Timestep=slice(2,None)) #.dropna(dim='Timestep')
        ds_observed_anom = ds_observed_anom.isel(Timestep=slice(0,leadtime,1))
        ds_observed_anom = ds_observed_anom.rolling(Timestep=3, center=False).sum().isel(Timestep=slice(2,None)) #.dropna(dim='Timestep')
        print(f'date_i, startdate: {date_i}, {startdate}\n')
        print(f'ds_hybrid_anom {var}: {ds_hybrid_anom}\n')
        print(f'ds_observed_anom {var}: {ds_observed_anom}\n')

        #print(f'ds_observed_anom.groupby("season")["DJF"]: {ds_observed_anom.groupby("season")["DJF"]}\n')
        #print(f'ds_observed_anom.isel(Timestep=0).season {ds_observed_anom.isel(Timestep=0).season.values == "DJF"}\n')

        ds_hybrid_all.append(ds_hybrid_anom)
        ds_observed_all.append(ds_observed_anom)

    return ds_hybrid_all, ds_observed_all

def get_spatial_var_anom_ds(startdates,prediction_length,timestep,var,resamp='seasonal',leadtime=4):
    """Return Python lists of xarray DataArrays of anomaly heatmaps for given variable.
       var: 'p6hr','SST','U-wind','logp'.
       resamp: 'seasonal','monthly'
       leadtime: integer."""

    hybrid_root = '/scratch/user/dpp94/Predictions/Hybrid/hybrid_prediction_era6000_20_20_20_sigma0.5_beta_res0.001_beta_model_1.0_prior_0.0_overlap1_vertlevel_1_precip_epsilon0.001_ohtc_multiple_leakage_test_oceantimestep_72hr_train1981_2002_oldcal__pred_newcal_trial_'

    obs_climo_ref_startdate = datetime(1981,1,1,0)
    obs_climo_ref_enddate = datetime(2002,12,31,0)

    # Obtain observed var climotology
    if var == 'p6hr':
       ds_observed_climo = get_obs_precip_timeseries(obs_climo_ref_startdate,obs_climo_ref_enddate,1)['tp'] * 39.37
       ds_observed_climo = ds_observed_climo.resample(Timestep='6H').sum(dim='Timestep')  #experimental
    elif var == 'SST':
       ds_observed_climo = get_obs_sst_timeseries(obs_climo_ref_startdate,obs_climo_ref_enddate,timestep)['sst'] #.astype('float32')
       ds_observed_climo = ds_observed_climo.rename({'time':'Timestep','lon':'Lon','lat':'Lat'})
    elif var == 'U-wind':
       ds_observed_climo = get_obs_atmo_timeseries_var(obs_climo_ref_startdate,obs_climo_ref_enddate,timestep,var,sigma_lvl=7)[var]
    elif var == 'logp':
       ds_observed_climo = get_obs_atmo_timeseries_var(obs_climo_ref_startdate,obs_climo_ref_enddate,timestep,var)[var]
       ds_observed_climo = 1000 * np.exp(ds_observed_climo)
    print(f'ds_observed_climo {var}: {ds_observed_climo}\n')  
  
    ds_hybrid_all, ds_observed_all, ds_per_all = [], [], []
    for startdate in startdates:
        # Read in hybrid forecast, calculate anomaly, and retain only up to leadtime
        date_str = startdate.strftime("%m_%d_%Y_%H")
        filepath = hybrid_root + date_str + ".nc"
        print(f'Loading in file at: {filepath}')
        ds_hybrid = xr.open_dataset(filepath)
        ds_hybrid = make_ds_time_dim(ds_hybrid, timestep, startdate)
        ds_hybrid = ds_hybrid[var]
        if var == 'U-wind':
           ds_hybrid = ds_hybrid.sel(Sigma_Level=7)
        elif var == 'logp':
           ds_hybrid = 1000 * np.exp(ds_hybrid)
        elif var == 'SST':
           ds_observed_climo = ds_observed_climo.assign_coords(coords={'Lon':ds_hybrid.coords['Lon'].values,'Lat':ds_hybrid.coords['Lat'].values})
        print(f'ds_hybrid {var}: {ds_hybrid}\n')
        print(f'ds_observed_climo {var}: {ds_observed_climo}\n')
        # --- CHANGE get_anom FUNC BELOW FOR APPROPRIATE SAMPLING --- #
        if resamp == 'monthly':
           ds_hybrid_anom = get_anom_specified_climo_hybrid(ds_observed_climo, ds_hybrid)
        else:
           #ds_hybrid_anom = get_anom_specified_climo_hybrid_weekly(ds_observed_climo, ds_hybrid)
           ds_hybrid_anom = get_anom_specified_climo_hybrid_daily(ds_observed_climo, ds_hybrid)  #its ACCUMULATED not AVERAGED
           ds_hybrid_anom = ds_hybrid_anom.resample(Timestep='7D').sum(dim='Timestep')
        ds_hybrid_anom = ds_hybrid_anom.isel(Timestep=slice(0,leadtime,1))  
        print(f'ds_hybrid_anom {var}: {ds_hybrid_anom}\n')

        # Read in observed forecast, calculate anomaly, and retain only up to leadtime
        date_i, date_f = startdate, startdate + timedelta(hours=prediction_length)
        if var == 'p6hr':
           ds_observed = get_obs_precip_timeseries(date_i,date_f,1)['tp'] * 39.37
           ds_observed = ds_observed.resample(Timestep='6H').sum(dim='Timestep')  #experimental
        elif var == 'SST':
           ds_observed = get_obs_sst_timeseries(date_i,date_f,timestep)['sst'] #.astype('float32')
           ds_observed = ds_observed.rename({'time':'Timestep','lon':'Lon','lat':'Lat'})
           ds_observed = ds_observed.assign_coords(coords={'Lon':ds_hybrid.coords['Lon'].values,'Lat':ds_hybrid.coords['Lat'].values})
        elif var == 'U-wind':
           ds_observed = get_obs_atmo_timeseries_var(date_i,date_f,timestep,var,sigma_lvl=7)[var]
        elif var == 'logp':
           ds_observed = get_obs_atmo_timeseries_var(date_i,date_f,timestep,var)[var]
           ds_observed = 1000 * np.exp(ds_observed_climo)
        print(f'ds_observed {var}: {ds_observed}\n')
        # --- CHANGE get_anom FUNC BELOW FOR APPROPRIATE SAMPLING --- #
        if resamp == 'monthly':
           ds_observed_anom = get_anom_specified_climo_hybrid(ds_observed_climo, ds_observed)
        else:
           #ds_observed_anom = get_anom_specified_climo_hybrid_weekly(ds_observed_climo, ds_observed)
           ds_observed_anom = get_anom_specified_climo_hybrid_daily(ds_observed_climo, ds_observed)
           ds_observed_anom = ds_observed_anom.resample(Timestep='7D').sum(dim='Timestep')
        print(f'ds_observed_anom {var}: {ds_observed_anom}\n')
        ds_observed_anom = ds_observed_anom.isel(Timestep=slice(0,leadtime,1))
        print(f'ds_observed_anom {var}: {ds_observed_anom}\n')

        ds_hybrid_all.append(ds_hybrid_anom)
        ds_observed_all.append(ds_observed_anom)

    return ds_hybrid_all, ds_observed_all

def get_spatial_precip_weekly_anom_ds(startdates,prediction_length,timestep,leadtime=4):
    """Return Python lists of xarray DataArrays of weekly anomaly heatmaps for precip.
       leadtime: integer."""

    hybrid_root = '/scratch/user/dpp94/Predictions/Hybrid/hybrid_prediction_era6000_20_20_20_sigma0.5_beta_res0.001_beta_model_1.0_prior_0.0_overlap1_vertlevel_1_precip_epsilon0.001_ohtc_multiple_leakage_test_oceantimestep_72hr_train1981_2002_oldcal__pred_newcal_trial_'

    obs_climo_ref_startdate = datetime(1981,1,1,0)
    obs_climo_ref_enddate = datetime(2002,12,31,0)

    # Obtain observed var climotology
    ds_observed_climo = get_obs_precip_timeseries(obs_climo_ref_startdate,obs_climo_ref_enddate,timestep)['tp']
    ds_observed_climo = ds_observed_climo.resample(Timestep='6H').sum(dim='Timestep') 
  
    # Get daily climotology
    ds_observed_climo_daily = ds_observed_climo.groupby('Timestep.dayofyear').sum(dim='Timestep') 
  
    ds_hybrid_all, ds_observed_all, ds_per_all = [], [], []
    for startdate in startdates:
        # Prepare climo data
        dayofyear = startdate.timetuple().tm_yday
        ds_observed_climo_daily_rolled = ds_observed_climo_daily.roll(shifts={'dayofyear':dayofyear},
                                                                              roll_coords=True)
 
 
        # Read in hybrid forecast, calculate anomaly, and retain only up to leadtime
        date_str = startdate.strftime("%m_%d_%Y_%H")
        filepath = hybrid_root + date_str + ".nc"
        print(f'Loading in file at: {filepath}')
        ds_hybrid = xr.open_dataset(filepath)
        ds_hybrid = make_ds_time_dim(ds_hybrid, timestep, startdate)
        ds_hybrid = ds_hybrid['p6hr']
        print(f'ds_hybrid {p6hr}: {ds_hybrid}\n')
        print(f'ds_observed_climo {p6hr}: {ds_observed_climo}\n')
        # --- Calculate anomaly signal --- #
        ds_hybrid = ds_hybrid.resample(Timestep='7D').sum(dim='Timestep')
        ds_hybrid_anom = ds_hybrid_anom.isel(Timestep=slice(0,leadtime,1))  
        print(f'ds_hybrid_anom p6hr: {ds_hybrid_anom}\n')

        # Read in observed forecast, calculate anomaly, and retain only up to leadtime
        date_i, date_f = startdate, startdate + timedelta(hours=prediction_length)
        ds_observed = get_obs_precip_timeseries(date_i,date_f,timestep)['tp']
        ds_observed = ds_observed.resample(Timestep='6H').sum(dim='Timestep')  #experimental
        # --- Calculate anomaly signal --- #

        print(f'ds_observed_anom {var}: {ds_observed_anom}\n')
        ds_observed_anom = ds_observed_anom.isel(Timestep=slice(0,leadtime,1))
        print(f'ds_observed_anom {var}: {ds_observed_anom}\n')

        ds_hybrid_all.append(ds_hybrid_anom)
        ds_observed_all.append(ds_observed_anom)

    return ds_hybrid_all, ds_observed_all

def plot_spatial_anom_pcc_weekly(ds_hybrid_all,ds_obs_all,startdates,sigtest=True):
    """Plot spatial heatmaps of weekly pcc and rmse."""

    leadtime = 4
    num_dates = len(startdates)

    lons = ds_hybrid_all[0].Lon.values
    lats = ds_hybrid_all[0].Lat.values
   
    nrows, ncols = 1, 4
    fig, axs = plt.subplots(nrows=nrows,ncols=ncols,
                            subplot_kw={'projection': ccrs.PlateCarree(central_longitude=-179)},
                            figsize=(11,8.5), sharex='col', sharey='row')

    axs = axs.flatten()

    pcc, pvals, pvals_temp = [], [], []
    for lead in range(leadtime):
        # Calculate PCC
        samples_hybrid, samples_obs = [], []
        for i in range(num_dates):
            ds_hybrid = ds_hybrid_all[i]
            ds_obs = ds_obs_all[i]
            samples_hybrid.append(ds_hybrid.isel(Timestep=lead))
            samples_obs.append(ds_obs.isel(Timestep=lead))
        samples_hybrid = xr.concat(samples_hybrid, dim='Timestep')
        samples_obs = xr.concat(samples_obs, dim='Timestep')
        print(f'samples_hybrid: {samples_hybrid}\n')
        pcc.append(xr.corr(samples_hybrid, samples_obs, dim='Timestep'))
        if sigtest:
           pvals_temp = pearson_r_eff_p_value(samples_hybrid, samples_obs, dim='Timestep')
        print(f'pvals_temp: {pvals_temp}\n')
        pvals.append(pvals_temp)
    print(f'pcc[0].shape, pcc[0]: {pcc[0].shape}\n, {pcc[0]}\n')
 
    for lead in range(leadtime):
        print(f'plotting lead time {lead}\n')
        data = pcc[lead]
        data, data_lons = add_cyclic_point(data, coord=lons)
        levels = [-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1]
        temp_colormap = sns.color_palette("Spectral", as_cmap=True)
        temp_colormap = temp_colormap.reversed()
        cmap = temp_colormap
        white_color = (1,1,1,1) #RGBA for white
        colors = [cmap(i) for i in np.linspace(0,1,256)]
        for i in range(len(colors)):
            if -0.25 <= (i / 255 - 0.5) * 2 <= 0.25:
                colors[i] = white_color
        cmap = LinearSegmentedColormap.from_list('custom_spectral', colors)
        #cmap = (mpl.colors.ListedColormap(["#0000ff","#6464ff","#c8c8ff",
        #                                   "#eee68c","#e7dc32","#ff8801",
        #                                   "#ff0100"]).with_extremes(under="#000178",
        #                                                             over="#790000")) 
        cs = axs[lead].pcolormesh(data_lons,lats,data,
                                  transform=ccrs.PlateCarree(),
                                  cmap=cmap,vmin=-1,vmax=1)
        if sigtest:
           pvals_temp = pvals[lead].where(pvals[lead] < 0.05).roll(Lon=int(len(lons)/2)).values.flatten()
           print(f'pvals_temp.flatten(): {pvals_temp}\n')
           # --- Plot --- #
           X, Y = np.meshgrid(lons,lats)
           print(f'X, Y: {X}, {Y}\n')
           colors = ["none" if np.isnan(pvals_temp[i]) else "black" for i in range(pvals_temp.size)]
           delta_x = 360 / lons.size / 2
           delta_y = 180 / lats.size / 2
           axs[lead].scatter(X+0*delta_x, Y+0*delta_x, s=0.25, marker='.', color=colors)
        #axs[lead].set_title(f'Week {lead+1}')
        axs[lead].coastlines()
        if lead == 0:
           axs[lead].set_yticks([-60,-30,0,30,60], crs=ccrs.PlateCarree())
           lat_formatter = LatitudeFormatter()
           axs[lead].yaxis.set_major_formatter(lat_formatter)
        axs[lead].set_xticks([-90,0,90,180], crs=ccrs.PlateCarree())
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        axs[lead].xaxis.set_major_formatter(lon_formatter)
 
    fig.subplots_adjust(bottom=0.2,top=0.9,left=0.05,right=0.95,
                 wspace=0.3,hspace=0.25) #[left,right,bottom,top,width-white-space,height-white-space]
    plt.suptitle('PCC of weekly precipitation anomalies')


    ## Make colorbar
    #cmap = (mpl.colors.ListedColormap(["#0000ff","#6464ff", "#c8c8ff", "#eee68c", "#e7dc32",
    #                                   "#ff8801", "#ff0100"]).with_extremes(under="#000178",
    #                                                                       over="#790000"))
    #bounds = [-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8]
    #norm = mpl.colors.BoundaryNorm(bounds, cmap.N) 
    cbar_ax = fig.add_axes([0.25,0.15,0.5,0.025]) #[left,bottom,width,height]
    fig.colorbar(cs, cax=cbar_ax, orientation='horizontal', label='r')
    #fig.colorbar(
    #    mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
    #    cax=cbar_ax, orientation='horizontal', 
    #    extend='both',
    #    spacing='proportional',
    #    label='r'
    #)
    
    #plt.show()
    plt.tight_layout() 
    fig.savefig('Precip_Weekly_PCC.svg')

def plot_spatial_bias_weekly_djf(ds_hybrid_all, ds_obs_all, startdates):
    """Plot mean precip rates at weekly lead times in DJF season.
       Row 1: Hybrid mean rates
       Row 2: ERA5 mean rates
       Row 3: Abs difference."""

    leadtime = 4
    num_dates = len(startdates)

    lons = ds_hybrid_all[0].Lon.values
    lats = ds_hybrid_all[0].Lat.values

    startmonth = 12   

    nrows, ncols = 1, 4
    fig, axs = plt.subplots(nrows=nrows,ncols=ncols,
                            subplot_kw={'projection': ccrs.PlateCarree(central_longitude=-179)},
                            figsize=(11,8.5), sharex='col', sharey='row')

    axs = axs.flatten()

    hybrid_mean, obs_mean, bias = [], [], []
    for lead in range(leadtime):
        samples_hybrid, samples_obs = [], []
        for i in range(num_dates):
            ds_hybrid = ds_hybrid_all[i] * 25.4  # for converting inches -> mm
            ds_obs = ds_obs_all[i] * 25.4
            if (startdates[i].month == startmonth):
               print(f'Mean using startmonth: {startdates[i].month}\n')
               samples_hybrid.append(ds_hybrid.isel(Timestep=lead))
               samples_obs.append(ds_obs.isel(Timestep=lead))
        samples_hybrid = xr.concat(samples_hybrid, dim='Timestep')
        samples_obs = xr.concat(samples_obs, dim='Timestep')
        print(f'samples_hybrid: {samples_hybrid}\n')
        hybrid_mean.append(samples_hybrid.mean(dim='Timestep'))
        obs_mean.append(samples_obs.mean(dim='Timestep'))
        bias.append(hybrid_mean[-1] - obs_mean[-1])
        print(f'bias[-1]: {bias[-1]}\n')
    print(f'bias[0].shape: {bias[0].shape}\n')

    for lead in range(leadtime):
        print(f'plotting lead time {lead}\n')
        hybrid_data = hybrid_mean[lead]
        obs_data = obs_mean[lead]
        bias_data = bias[lead]
        hybrid_data, hybrid_data_lons = add_cyclic_point(hybrid_data, coord=lons)
        obs_data, obs_data_lons = add_cyclic_point(obs_data, coord=lons)
        bias_data, bias_data_lons = add_cyclic_point(bias_data, coord=lons)
        
        # Discrete colorbar from de Andrade Global Precip .. paper
        colors = ['#0000ff', '#8282ff', '#cccdff', '#ffffff', '#ffcdce', '#ff8281', '#ff0100'] # blue to red
        cmap = (ListedColormap(colors).with_extremes(under='#000178', over='#790002'))
        bounds = [-35,-25,-15,-5,5,15,25,35]
        norm = BoundaryNorm(bounds, cmap.N) #, extend='both')
        
        #cmap = sns.color_palette("vlag", as_cmap=True)
        #cmap = cmap.reversed()
        
        #hybrid_cs = axs[lead, 0].pcolormesh(hybrid_data_lons, lats, hybrid_data,
        #                                    transform=ccrs.PlateCarree(),
        #                                    cmap=cmap, vmin=0, vmax=16)
        #fig.colorbar(hybrid_cs, shrink=0.5, extend='both')
         
        #obs_cs = axs[lead, 1].pcolormesh(obs_data_lons, lats, obs_data,
        #                                 transform=ccrs.PlateCarree(),
        #                                 cmap=cmap, vmin=0, vmax=16)
        #fig.colorbar(obs_cs, shrink=0.5, extend='both')
 
        cs = bias_cs = axs[lead].pcolormesh(bias_data_lons, lats, bias_data,
                                            transform=ccrs.PlateCarree(),
                                            cmap=cmap, norm=norm) #, vmin=-35, vmax=35)
        #fig.colorbar(bias_cs, shrink=0.5, extend='both')

    #axs = axs.reshape((nrows, ncols))
    #for row in range(nrows):
    #    axs[row, 0].set_yticks([-60,-30,0,30,60], crs=ccrs.PlateCarree())
    #    lat_formatter = LatitudeFormatter()
    #    axs[row, 0].yaxis.set_major_formatter(lat_formatter)
    axs[0].set_yticks([-60,-30,0,30,60], crs=ccrs.PlateCarree())
    lat_formatter = LatitudeFormatter()
    axs[0].yaxis.set_major_formatter(lat_formatter)

    #for col in range(ncols):
    #    axs[nrows-1,col].set_xticks([-90,0,90,180], crs=ccrs.PlateCarree())
    #    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    #    axs[nrows-1,col].xaxis.set_major_formatter(lon_formatter)  
   
    for ax in axs.flatten():
        ax.set_xticks([-90,0,90,180], crs=ccrs.PlateCarree())
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        ax.xaxis.set_major_formatter(lon_formatter)
    for ax in axs.flatten():
        ax.coastlines()
 
    #fig.subplots_adjust(bottom=0.2,top=0.9,left=0.05,right=0.95,
    #             wspace=0.3,hspace=0.25) #[left,right,bottom,top,width-white-space,height-white-space]
    #plt.suptitle('Mean Precip Rates')

    cbar_ax = fig.add_axes([0.25,0.15,0.5,0.025])
    #fig.colorbar(cs, cax=cbar_ax, orientation='horizontal', label='mm') #, boundaries=bounds, ticks=np.arange(-35,36,10))
    fig.colorbar(
        mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
        cax=cbar_ax, orientation='horizontal',
        extend='both', extendfrac='auto',
        spacing='uniform', label='mm') 

    #plt.show() 
    plt.tight_layout() 
    fig.savefig('DJF_Weekly_Precip_Errors.svg')   
 

def plot_spatial_anom_rmse_pcc_seasonal(ds_hybrid_all,ds_obs_all,startdates,sigtest=True):
    """Plot spatial heatmaps of seasonal pcc and rmse."""

    leadtime = 1
    num_dates = len(startdates)

    lons = ds_hybrid_all[0].Lon.values
    lats = ds_hybrid_all[0].Lat.values

    print(f'ds_obs_all[0].groupby("season"): {ds_obs_all[0].groupby("season")}\n')
    seasons = ['DJF','MAM','JJA','SON'] #list(ds_obs_all[0].groupby("season").groups.keys())
    startmonths = [12,3,6,9]
    print(f'seasons: {seasons}\n')

    nrows, ncols = 2, 2
    fig, axs = plt.subplots(nrows=nrows,ncols=ncols,
                            subplot_kw={'projection': ccrs.PlateCarree(central_longitude=-179)},
                            figsize=(11,8.5), sharex='col', sharey='row')
    axs = axs.flatten()

    for num_season, season in enumerate(seasons):    
        # Calculate RMSE
        rmse = []
        for lead in range(leadtime):
            err = []
            for i in range(num_dates):
                ds_hybrid = ds_hybrid_all[i]
                ds_obs = ds_obs_all[i]
                if (startdates[i].month == startmonths[num_season]): #(ds_hybrid.isel(Timestep=0).season.values == season):
                   print(f'RMSE using start month: {startdates[i].month}\n')
                   err.append(ds_hybrid.isel(Timestep=lead).to_numpy() - ds_obs.isel(Timestep=lead).to_numpy()) 
            err = np.stack(err, axis=0)
            rmse.append(np.sqrt(np.mean(err**2, axis=0)))
        print(f'rmse[0].shape: {rmse[0].shape}\n')

        for lead, ax_num in enumerate([num_season]): #[2*num_season, 2*num_season+1]):
            data = rmse[lead]
            data, data_lons = add_cyclic_point(data,coord=lons)
            levels = [0,0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5]
            temp_colormap = sns.color_palette("Spectral", as_cmap=True)
            temp_colormap = temp_colormap.reversed()
            cmap = temp_colormap
            cs = axs[ax_num].pcolormesh(data_lons,lats,data,
                                      transform=ccrs.PlateCarree(),
                                      cmap=cmap, vmin=-1, vmax=1)                        
            #-- contourf smooths when rendering (give levels=levels and extend='both' as args, and remove vmin/vmax) --#
            axs[ax_num].set_title(f'{seasons[(num_season+lead)%4]}')
            axs[ax_num].coastlines()
            #axs[ax_num].set_xticks([-180, -120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree())
            #axs[ax_num].set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
            #lon_formatter = LongitudeFormatter(zero_direction_label=True)
            #lat_formatter = LatitudeFormatter()
            #axs[ax_num].xaxis.set_major_formatter(lon_formatter)
            #axs[ax_num].yaxis.set_major_formatter(lat_formatter)
    
    fig.subplots_adjust(bottom=0.2, top=0.9, left=0.05, right=0.95,
                        wspace=0.3, hspace=0.25)  # [left,right,bottom,top,width-white-space,height-white-space]
    cbar_ax = fig.add_axes([0.9, 0.2, 0.05, 0.6])  # [left,bottom,widht,height]
    cbar = fig.colorbar(cs, cax=cbar_ax, orientation='vertical')
    plt.suptitle('RMSE of seasonal precipitation anomalies')

    #fig.delaxes(axs[8])

    plt.show()

    # PCC
    nrows, ncols = 2, 2
    fig, axs = plt.subplots(nrows=nrows,ncols=ncols,
                           subplot_kw={'projection': ccrs.PlateCarree(central_longitude=-179)},
                           figsize=(11,8.5), sharex='col', sharey='row')
    axs = axs.flatten() 

    for num_season, season in enumerate(seasons):
        # Calculate PCC
        pcc = []
        pvals = []
        pvals_temp = []
        for lead in range(leadtime):
            samples_hybrid, samples_obs = [], []
            for i in range(num_dates):
                ds_hybrid = ds_hybrid_all[i]
                ds_obs = ds_obs_all[i]
                if (startdates[i].month == startmonths[num_season]): #(ds_hybrid.isel(Timestep=0).season.values == season):
                   print(f'PCC start month: {startdates[i].month}\n')
                   samples_hybrid.append(ds_hybrid.isel(Timestep=lead))
                   samples_obs.append(ds_obs.isel(Timestep=lead))
            samples_hybrid = xr.concat(samples_hybrid, dim='Timestep') #dim='samples')
            samples_obs = xr.concat(samples_obs, dim='Timestep') #dim='samples')
            print(f'samples_hybrid: {samples_hybrid}\n')
            pcc.append(xr.corr(samples_hybrid, samples_obs, dim='Timestep')) #dim='samples'))
            if sigtest:
               # --- Uncomment next line if confidence interval is calculated using Student's t-distribution --- #
               pvals_temp = pearson_r_eff_p_value(samples_hybrid, samples_obs, dim='Timestep')
               # --- Uncomment until next comment line if confidence interval is calculated using Fisher's method --- #
               #for lat in lats:
               #    pvals_temp_temp = []
               #    for lon in lons:
               #        pr = pearsonr(samples_hybrid.sel(Lat=lat,Lon=lon).values, samples_obs.sel(Lat=lat,Lon=lon).values)
               #        ci = pr.confidence_interval(0.95)
               #        if (ci.low < 0) & (ci.high > 0):
               #           pvals_temp_temp.append(0.)
               #        else:
               #           pvals_temp_temp.append(1.)
               #    pvals_temp.append(pvals_temp_temp)
               #pvals_temp = np.array(pvals_temp) #.reshape(len(lons), len(lats))
               #print(f'shape(pvals_temp), pvals_temp: {pvals_temp.shape}, {pvals_temp}\n')
               #pvals_temp = xr.DataArray(pvals_temp, dims=['Lat','Lon'], \
               #                          coords=dict(Lon=lons, Lat=lats))
               # --- Finished calculating confidence interval --- #
               print(f'pvals_temp: {pvals_temp}\n')
               pvals.append(pvals_temp)
        print(f'pcc[0].shape, pcc[0]: {pcc[0].shape}\n {pcc[0]}\n')

        for lead, ax_num in enumerate([num_season]): #[2*num_season, 2*num_season+1]):
            print(f'plotting lead, ax_num, season: {lead}, {ax_num}, {seasons[num_season]}\n')
            data = pcc[lead]
            data, data_lons = add_cyclic_point(data,coord=lons)
            levels = [-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1.0]
            temp_colormap = sns.color_palette("Spectral", as_cmap=True)
            temp_colormap = temp_colormap.reversed()
            cmap = temp_colormap
            cs = axs[ax_num].pcolormesh(data_lons,lats,data,
                                      transform=ccrs.PlateCarree(),
                                      cmap=cmap,vmin=-1,vmax=1)  
            if sigtest:
               # --- Uncomment next line if confidence interval is calculated using Student's t-distribution --- #
               pvals_temp = pvals[lead].where(pvals[lead] < 0.05).roll(Lon=int(len(lons)/2)).values.flatten()
               ## --- Uncomment next line if confidence interval is calculated using Fishers transformation --- #
               #pvals_temp = pvals[lead].where(pvals[lead] == 1.).roll(Lon=int(len(lons)/2)).values.flatten()
               print(f'pvals_temp.flatten() : {pvals_temp}\n')
               # --- Plot --- #
               X, Y = np.meshgrid(lons,lats)               
               print(f'X, Y: {X}, {Y}\n') 
               colors = ["none" if np.isnan(pvals_temp[i]) else "black" for i in range(pvals_temp.size)]
               delta_x = 360 / lons.size / 2
               delta_y = 180 / lats.size / 2
               axs[ax_num].scatter(X+0*delta_y, Y+0*delta_y, s=0.25, marker='.', color=colors) 
            #-- contourf will smooth when rendering (give levels=levels, extend='both' as args, and remove vmin/vmax) --#
            axs[ax_num].set_title(f'{seasons[(num_season+lead)%4]}')
            print(f'plot title: {seasons[(num_season+lead)%4]}\n')
            axs[ax_num].coastlines()
            #axs[ax_num].set_xticks([-180, -120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree())
            #axs[ax_num].set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
            #lon_formatter = LongitudeFormatter(zero_direction_label=True)
            #lat_formatter = LatitudeFormatter()
            #axs[ax_num].xaxis.set_major_formatter(lon_formatter)
            #axs[ax_num].yaxis.set_major_formatter(lat_formatter)

    fig.subplots_adjust(bottom=0.2, top=0.9, left=0.05, right=0.95,
                        wspace=0.3, hspace=0.25)  # [left,right,bottom,top,width-white-space,height-white-space]
    #cbar_ax = fig.add_axes([0.9, 0.2, 0.02, 0.6])  # [left,bottom,widht,height]
    #cbar = fig.colorbar(cs, cax=cbar_ax, orientation='vertical')
    plt.suptitle('PCC of seasonal precipitation anomalies')

    #fig.delaxes(axs[8])

    plt.show()

def get_spatial_corr_nino34_var_hybrid(startdate,prediction_length,timestep,var):
    """Obtain the seasonal spatial correlation between nino3.4 index and a given variable.
       var: 'p6hr','SST','U-wind','logp'."""

    hybrid_root = "/scratch/user/dpp94/Predictions/Hybrid/hybrid_prediction_era6000_20_20_20_sigma0.5_beta_res0.001_beta_model_1.0_prior_0.0_overlap1_vertlevel_1_precip_epsilon0.001_ohtc_multiple_leakage_test_oceantimestep_72hr_uvwindtemplogponly_70yr_trial_" 

    lat_slice = slice(-50,50)
    lon_slice = slice(None,None)

    obs_climo_ref_startdate = datetime(1981,1,1,0)
    obs_climo_ref_enddate = datetime(2008,12,31,0)

    # Obtain observed sst climotology for ONI
    ds_sst_observed_climo = get_obs_sst_timeseries(obs_climo_ref_startdate,obs_climo_ref_enddate,timestep)
    print(f'observed sst climo: {ds_sst_observed_climo}\n\n')

    # Obtain observed var climotology
    if var == 'p6hr':
       ds_observed_climo = get_obs_precip_timeseries(obs_climo_ref_startdate,obs_climo_ref_enddate,timestep)['tp']
    elif var == 'SST':
       ds_observed_climo = ds_sst_observed_climo['sst'] #.astype('float32')
       ds_observed_climo = ds_observed_climo.rename({'time':'Timestep','lon':'Lon','lat':'Lat'})
    elif var == 'U-wind':
       ds_observed_climo = get_obs_atmo_timeseries_var(obs_climo_ref_startdate,obs_climo_ref_enddate,timestep,var,sigma_lvl=7)[var]
    elif var == 'logp':
       ds_observed_climo = get_obs_atmo_timeseries_var(obs_climo_ref_startdate,obs_climo_ref_enddate,timestep,var)[var]
       ds_observed_climo = 1000 * np.exp(ds_observed_climo)
    print(f'ds_observed_climo {var}: {ds_observed_climo}\n')

    # Obtain hybrid prediction
    date_str = startdate.strftime("%m_%d_%Y_%H")
    filepath = hybrid_root + date_str + ".nc"
    print(f'Loading in file at: {filepath}')
    ds_hybrid = xr.open_dataset(filepath)
    ds_hybrid = make_ds_time_dim(ds_hybrid, timestep, startdate)

    # Obtain hybrid nino index
    ds_hybrid_nino = nino_index_monthly_specified_climo_hybrid(ds_sst_observed_climo, "3.4", ds_hybrid) 
    time_index = ds_hybrid_nino["Timestep"] 
    ds_hybrid_nino = uniform_filter1d(ds_hybrid_nino['sst'].values, size=1) #origin=1
    print(f'ds_hybrid_nino: {ds_hybrid_nino}\n')
    ds_hybrid_nino = xr.DataArray(ds_hybrid_nino, dims="Timestep", coords={"Timestep": time_index.values}, name=date_str)

    # Obtain hybrid var anomalies
    ds_hybrid = ds_hybrid[var]
    if var == 'U-wind':
       ds_hybrid = ds_hybrid.sel(Sigma_Level=7)
    elif var == 'logp':
       ds_hybrid = 1000 * np.exp(ds_hybrid)
    elif var == 'SST':
       ds_observed_climo = ds_observed_climo.assign_coords(coords={'Lon':ds_hybrid.coords['Lon'].values,'Lat':ds_hybrid.coords['Lat'].values})
    print(f'ds_hybrid {var}: {ds_hybrid}\n')
    print(f'ds_observed_climo {var}: {ds_observed_climo}\n')
    ds_hybrid_anom = get_anom_specified_climo_hybrid(ds_observed_climo, ds_hybrid)
    print(f'ds_hybrid_anom {var}: {ds_hybrid_anom}\n')

    # Obtain correlations
    data = xr.corr(ds_hybrid_anom, ds_hybrid_nino, dim="Timestep")
    print(data)

    # Plot results
    lons = data.Lon.values
    lats = data.Lat.values
    print(f'lons: {lons}\n')

    projection = ccrs.PlateCarree(central_longitude=-179)
    axes_class = (GeoAxes, dict(map_projection=projection))
    plt.rc('font', family='serif')
    plt.rcParams['figure.constrained_layout.use'] = True

    fig = plt.figure(figsize=(6,10))
    axgr = AxesGrid(fig, 111, axes_class=axes_class,
                  nrows_ncols=(1,1),
                  axes_pad=0.7,
                  cbar_location='right',
                  cbar_mode='single',
                  cbar_pad=0.2,
                  cbar_size='3%',
                  label_mode='')  # note the empty label_mode
    
    cyclic_data, cyclic_lons = add_cyclic_point(data, coord=lons)
    lons2d, lats2d = np.meshgrid(cyclic_lons,lats)
    ax = axgr[0]
    ax.coastlines()
    ax.set_xticks([-180, -120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree())
    ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_label=False, \
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')
    levels = [-1.0,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1.0] #[0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6]
    temp_colormap = sns.color_palette("Spectral", as_cmap=True)
    temp_colormap = temp_colormap.reversed()
    cmap = temp_colormap
    plot  = ax.contourf(lons2d,lats2d,cyclic_data,transform=ccrs.PlateCarree(),levels=levels,cmap=cmap,extend='both')
    cbar = axgr.cbar_axes[0].colorbar(plot, extend='both')
    cbar.set_ticks(levels)
    cbar.set_label('Correlation',fontsize=16)
    cbar.ax.tick_params(labelsize=16)
    #ax.add_feature(cart.feature.LAND, zorder=100, edgecolor='k', facecolor='#808080')
    if var == 'p6hr':
       title = 'ENSO and Precip Anomaly Correlation - Hybrid'
    elif var == 'SST':
       title = 'ENSO and Global SST Anomaly Correlation - Hybrid'
    elif var == 'U-wind':
       title = 'ENSO and u-wind Anomaly Correlation - Hybrid'
    elif var == 'logp':
       title = 'ENSO and Surface Pressue Anomaly Correlation - Hybrid'
    ax.set_title(title, fontsize=18,fontweight='bold')
    ax.set_extent([-180,180,-90,90], crs=ccrs.PlateCarree())

    plt.show()

    return data

def get_spatial_corr_nino34_var_era(timestep,var):
    """Obtain the seasonal spatial correlation between nino3.4 index and a given variable.
       var: 'tp','sst','U-wind','logp'."""

    lat_slice = slice(-50,50)
    lon_slice = slice(None,None)

    obs_climo_ref_startdate = datetime(1981,1,1,0)
    obs_climo_ref_enddate = datetime(2008,12,31,0)

    # Obtain observed sst climotology for ONI
    ds_sst = get_obs_sst_timeseries(obs_climo_ref_startdate,obs_climo_ref_enddate,timestep)
    ds_sst = ds_sst.rename({'time':'Timestep','lon':'Lon','lat':'Lat'})
    print(f'observed sst climo: {ds_sst}\n\n')

    # Obtain observed var climotology
    if var == 'tp':
       ds_var = get_obs_precip_timeseries(obs_climo_ref_startdate,obs_climo_ref_enddate,timestep)[var]
    elif var == 'sst':
       ds_var = ds_sst[var]
    elif var == 'U-wind':
       ds_var = get_obs_atmo_timeseries_var(obs_climo_ref_startdate,obs_climo_ref_enddate,timestep,var,sigma_lvl=7)[var]
    elif var == 'logp':
       ds_var = get_obs_atmo_timeseries_var(obs_climo_ref_startdate,obs_climo_ref_enddate,timestep,var)[var]
       ds_var = 1000 * np.exp(ds_var)
    print(f'ds_var {var}: {ds_var}\n')

    # Obtain nino index
    date_str = obs_climo_ref_startdate.strftime("%m_%d_%Y_%H")
    ds_sst_nino = nino_index_monthly_specified_climo(ds_sst, "3.4", ds_sst) 
    time_index = ds_sst_nino["Timestep"] 
    ds_sst_nino = uniform_filter1d(ds_sst_nino['sst'].values, size=1) #origin=1
    print(f'ds_sst_nino: {ds_sst_nino}\n')
    ds_sst_nino = xr.DataArray(ds_sst_nino, dims="Timestep", coords={"Timestep": time_index.values}, name=date_str)

    # Obtain hybrid var anomalies
    ds_var_anom = get_anom_specified_climo_hybrid(ds_var, ds_var)
    print(f'ds_var_anom {var}: {ds_var_anom}\n')

    # Obtain correlations
    data = xr.corr(ds_var_anom, ds_sst_nino, dim="Timestep")
    print(data)

    # Plot results
    lons = data.Lon.values
    lats = data.Lat.values
    print(f'lons: {lons}\n')

    projection = ccrs.PlateCarree(central_longitude=-179)
    axes_class = (GeoAxes, dict(map_projection=projection))
    plt.rc('font', family='serif')
    plt.rcParams['figure.constrained_layout.use'] = True

    fig = plt.figure(figsize=(6,10))
    axgr = AxesGrid(fig, 111, axes_class=axes_class,
                  nrows_ncols=(1,1),
                  axes_pad=0.7,
                  cbar_location='right',
                  cbar_mode='single',
                  cbar_pad=0.2,
                  cbar_size='3%',
                  label_mode='')  # note the empty label_mode
    
    cyclic_data, cyclic_lons = add_cyclic_point(data, coord=lons)
    lons2d, lats2d = np.meshgrid(cyclic_lons,lats)
    ax = axgr[0]
    ax.coastlines()
    ax.set_xticks([-180, -120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree())
    ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_label=False, \
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')
    levels = [-1.0,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1.0] #[0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6]
    temp_colormap = sns.color_palette("Spectral", as_cmap=True)
    temp_colormap = temp_colormap.reversed()
    cmap = temp_colormap
    plot  = ax.contourf(lons2d,lats2d,cyclic_data,transform=ccrs.PlateCarree(),levels=levels,cmap=cmap,extend='both')
    cbar = axgr.cbar_axes[0].colorbar(plot, extend='both')
    cbar.set_ticks(levels)
    cbar.set_label('Correlation',fontsize=16)
    cbar.ax.tick_params(labelsize=16)
    #ax.add_feature(cart.feature.LAND, zorder=100, edgecolor='k', facecolor='#808080')
    if var == 'tp':
       title = 'ENSO and Precip Anomaly Correlation - ERA'
    elif var == 'sst':
       title = 'ENSO and Global SST Anomaly Correlation - ERA'
    elif var == 'U-wind':
       title = 'ENSO and u-wind Anomaly Correlation - ERA'
    elif var == 'logp':
       title = 'ENSO and Surface Pressue Anomaly Correlation - ERA'
    ax.set_title(title, fontsize=18,fontweight='bold')
    ax.set_extent([-180,180,-90,90], crs=ccrs.PlateCarree())

    plt.show()

    return data

def get_spatial_corr_nino34_var_diff(era_data,hybrid_data,var):
    """var: 'precip','SST','U-wind','Surface Pressure'."""

    era_data = era_data.assign_coords(coords={'Lon':hybrid_data.coords['Lon'].values,'Lat':hybrid_data.coords['Lat'].values})
    data = era_data - hybrid_data

    # Plot results
    lons = data.Lon.values
    lats = data.Lat.values
    print(f'lons: {lons}\n')

    projection = ccrs.PlateCarree(central_longitude=-179)
    axes_class = (GeoAxes, dict(map_projection=projection))
    plt.rc('font', family='serif')
    plt.rcParams['figure.constrained_layout.use'] = True

    fig = plt.figure(figsize=(6,10))
    axgr = AxesGrid(fig, 111, axes_class=axes_class,
                  nrows_ncols=(1,1),
                  axes_pad=0.7,
                  cbar_location='right',
                  cbar_mode='single',
                  cbar_pad=0.2,
                  cbar_size='3%',
                  label_mode='')  # note the empty label_mode

    cyclic_data, cyclic_lons = add_cyclic_point(data, coord=lons)
    lons2d, lats2d = np.meshgrid(cyclic_lons,lats)
    ax = axgr[0]
    ax.coastlines()
    ax.set_xticks([-180, -120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree())
    ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_label=False, \
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')
    levels = [-1.0,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1.0] #[0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6]
    temp_colormap = sns.color_palette("Spectral", as_cmap=True)
    temp_colormap = temp_colormap.reversed()
    cmap = temp_colormap
    plot  = ax.contourf(lons2d,lats2d,cyclic_data,transform=ccrs.PlateCarree(),levels=levels,cmap=cmap,extend='both')
    cbar = axgr.cbar_axes[0].colorbar(plot, extend='both')
    cbar.set_ticks(levels)
    cbar.set_label('Correlation',fontsize=16)
    cbar.ax.tick_params(labelsize=16)
    #ax.add_feature(cart.feature.LAND, zorder=100, edgecolor='k', facecolor='#808080')
    ax.set_title(f'{var}: (ERA5 - Hybrid)', fontsize=18,fontweight='bold')
    ax.set_extent([-180,180,-90,90], crs=ccrs.PlateCarree())

    # Avg absoluter bias
    bias = np.fabs(data).mean()
    print(f'\n\nAvg absolute bias: {bias}\n\n')

    plt.show()

def get_predictionhorizon_vs_startdate(rmses_climo,rmses):
    """Get prediction horizon (lead time at which rmse equals or exceeds rmse_ref).
    Input:
        rmses_climo: [list] list of lists of rmse of reference forecasts (climatology).
        rmses: [list] list of lists of rmse of forecasts.
    Returns:
        phorizon: [list] prediction horizon (lead time, in days, at which rmse equals of exceeds rmse_climo)."""

    phorizon = []
    for rmse_climo, rmse in zip(rmses_climo,rmses):
        rmse_climo = np.array(rmse_climo)
        rmse = np.array(rmse)
        print(f'rmse_climo: {rmse_climo}\n')
        print(f'rmse: {rmse}\n\n')
        idx = np.argwhere(rmse >= rmse_climo)[0][0]
        phorizon.append(idx)
    return phorizon

def get_rmse_training_error():
    """Calculate RMSE training error.
    Input:
        
    Returns:
        rmse_atmo: [] rmse training error for atmosphere model variables.
        rmse_sst: [] rmse training error for ocean model variables"""

    #error_atmo_path = '/home/dpp94/test-SPEEDY-ML/SPEEDY-ML/src/training_error_6000_20_20_20_sigma0.5_beta_res0.001_beta_model_1.0_prior_0.0_overlap1_vertlevel_1_precip_epsilon0.001_no_ohtc_multiple_leakage_atmo_test_.nc'
    error_sst_path = '/scratch/user/dpp94/training_error_ocean_6000_20_20_20_sigma0.5_beta_res0.001_beta_model_1.0_prior_0.0_overlap1_vertlevel_1_precip_epsilon0.001_ohtc_multiple_leakage_test_oceantimestep_72hr_train1981_2002_newcal_trainerror_.nc'   

    lat_slice = slice(-50,50) #-5,5) #-50,50
    lon_slice = slice(None,None) #360-170,360-120) #None,None)

    #error_atmo = xr.open_dataset(error_atmo_path)["Temperature"]
    error_sst = xr.open_dataset(error_sst_path)["SST"].sel(Lat=lat_slice,Lon=lon_slice)
    error_ohtc = xr.open_dataset(error_sst_path)["sohtc300"].sel(Lat=lat_slice,Lon=lon_slice)
    #error_mld = xr.open_dataset(error_sst_path)["somxl010"]
    error_sst = error_sst.where(error_sst < 10**30)
    error_ohtc = error_ohtc.where(error_ohtc < 10*30)
    #error_mld = error_mld.where(error_mld > 0)
 
    rmse_atmo = []
    rmse_sst = []
    rmse_ohtc = []
    #rmse_mld = []
    #for i in error_atmo["Timestep"]:
        #rmse_atmo.append(np.sqrt((error_atmo.isel(Timestep=i)**2).mean(dim=['Lat','Lon','Sigma_Level'], skipna=True).values))
    for i in error_sst["Timestep"]:
        print(i)
        rmse_sst.append((error_sst.isel(Timestep=i).mean(dim=['Lat','Lon'], skipna=True).values)) #np.sqrt((error_sst.isel(Timestep=i)).mean(dim=['Lat','Lon'], skipna=True).values)) 
        rmse_ohtc.append(error_ohtc.isel(Timestep=i).mean(dim=['Lat','Lon'], skipna=True).values) #np.sqrt((error_ohtc.isel(Timestep=i)).mean(dim=['Lat','Lon'], skipna=True).values)) 
        #rmse_mld.append(np.sqrt((error_mld.isel(Timestep=i)).mean(dim=['Lat','Lon'], skipna=True).values)) 
    
    print(f'avg rmse_sst: {np.mean(rmse_sst)}\n')
    print(f'avg rmse_ohtc: {np.mean(rmse_ohtc)}\n')
    return rmse_atmo, rmse_sst, rmse_ohtc #, rmse_mld

def find_min_rmse(date,timestep,ocean_timestep,window):
    """Find RMSE of first step of prediction w.r.t observed data in a window before and after it.
    Input:

    Returns:
        t: time vector of datetimes correponding to rmse.
        rmse: rmse of first of prediction.
        rmse_per: rmse of persistence."""

    hybrid_root = "/scratch/user/dpp94/Predictions/Hybrid/hybrid_prediction_era6000_20_20_20_sigma0.5_beta_res0.001_beta_model_1.0_prior_0.0_overlap1_vertlevel_1_precip_epsilon0.001_ohtc_multiple_leakage_test_blkdiagres_leak_1_4_trial_"

    lat_slice = slice(-5,5)
    lon_slice = slice(360-170,360-120)
     
    startdate = date - timedelta(hours=window*ocean_timestep)
    enddate = date + timedelta(hours=window*ocean_timestep)
    date_per = date - timedelta(hours=ocean_timestep)
  
    # Load in observed data
    ds_obs = get_obs_sst_timeseries(startdate, enddate, timestep)['sst'].sel(lat=lat_slice,lon=lon_slice) #['sohtc300'] #get_obs_ohtc_timeseries
    ds_obs = ds_obs.where(ds_obs > 272.0) #.mean(dim=['lat','lon'], skipna=True)
    print(f'ds_obs: {ds_obs.sel(time=date).mean(dim=["lat","lon"])}\n') #sst:'time', ohtc:'time_counter'
    ds_obs = ds_obs.rolling(time=int(ocean_timestep/timestep)).mean('time')  #time -> time_counter for ohtc

    # Load in hybrid prediction
    date_str = date.strftime("%m_%d_%Y_%H")
    filepath = hybrid_root + date_str + ".nc"
    ds_hybrid = xr.open_dataset(filepath)
    ds_hybrid = make_ds_time_dim(ds_hybrid, timestep, date)['SST'].sel(Lat=lat_slice,Lon=lon_slice) #['SST'] ['sohtc300']
    ds_hybrid = ds_hybrid.where(ds_hybrid > 272.0) #.mean(dim=['Lat','Lon'], skipna=True)
   
    rmse = []
    rmse_per = []
    time_idx = ds_obs.indexes["time"] #'time_counter'
    for t in time_idx:
        rmse.append(rms(ds_hybrid.sel(Timestep=date).values,ds_obs.sel(time=t).values))  #time -> time_counter for ohtc
        rmse_per.append(rms(ds_obs.sel(time=date_per).values,ds_obs.sel(time=t).values))

    return time_idx, rmse, rmse_per

def rmse_unnoisysync_training():

    pred_root = "/home/dpp94/test-SPEEDY-ML/SPEEDY-ML/src/vanilla_sync_prediction_ocean_6000_20_20_20_sigma0.5_beta_res0.001_beta_model_1.0_prior_0.0_overlap1_vertlevel_1_precip_epsilon0.001_ohtc_test_.nc"
    truth_root = "/home/dpp94/test-SPEEDY-ML/SPEEDY-ML/src/vanilla_sync_error_ocean_6000_20_20_20_sigma0.5_beta_res0.001_beta_model_1.0_prior_0.0_overlap1_vertlevel_1_precip_epsilon0.001_ohtc_multiple_leakage_test_3_.nc"

    ds_pred = xr.open_dataset(pred_root)["sohtc300"] #["SST"]
    ds_pred = ds_pred.where(ds_pred > 272.0)
    ds_truth = xr.open_dataset(truth_root)["sohtc300"] #["SST"]
    ds_truth = ds_truth.where(ds_truth > 272.0)
   
    rmse = []
    for t in ds_pred["Timestep"]:
        print(t)
        rmse.append(rms(ds_pred.sel(Timestep=t).values,ds_truth.sel(Timestep=t).values))
        #rmse.append(np.sqrt((ds_sync.sel(Timestep=t)**2).mean(dim=['Lat','Lon'], skipna=True).values))

    return rmse

def get_tlag_correlation_oni_ohtc(timestep, detrend_bool=True):
    """Calculate time-lag correlation between oceanic nino index (oni) and the oceanic heat content (ohtc) anomalies."""

    lat_slice = slice(-5,5)
    lon_slice = slice(360-170,360-120)

    startdate = datetime(1981,1,1,0)
    enddate = datetime(2010,1,1,0)

    # Get ohtc anomalies
    ds_ohtc = get_obs_ohtc_timeseries(startdate, enddate, timestep)['sohtc300']
    ds_ohtc = ds_ohtc.sel(lat=lat_slice, lon=lon_slice)
    ds_ohtc_copy = ds_ohtc.resample(time_counter="1MS").mean(dim="time_counter")
    ds_ohtc_copy = ds_ohtc_copy.groupby("time_counter.month")
    monthly_climo_ohtc = ds_ohtc_copy.mean("time_counter")
    ds_ohtc_anom = (ds_ohtc_copy - monthly_climo_ohtc).mean(dim=["lat","lon"])
    print(f'ds_ohtc_anom : {ds_ohtc_anom}\n')
    time_index_ohtc = ds_ohtc_anom["time_counter"] 
    ds_ohtc_anom = uniform_filter1d(ds_ohtc_anom.values, size=3, origin=1)

    # Get nino index 
    ds_sst = get_obs_sst_timeseries(startdate, enddate, timestep)
    ds_sst = nino_index_monthly_specified_climo(ds_sst, "3.4", ds_sst)
    time_index = ds_sst["time"]
    ds_sst = uniform_filter1d(ds_sst["sst"].values, size=3, origin=1)

    if detrend_bool:
       ds_ohtc_anom = detrend(ds_ohtc_anom, type='linear')
       ds_sst = detrend(ds_sst, type='linear')

    ds_ohtc_anom = xr.DataArray(ds_ohtc_anom, dims="time", coords={"time": time_index_ohtc.values}) #, name=date.strftime("%m_%d_%Y_%H"))
    print(f'observed ohtc anom: {ds_ohtc_anom}\n')
    ds_sst = xr.DataArray(ds_sst, dims="time", coords={"time": time_index.values}) #, name=date.strftime("%m_%d_%Y_%H"))
    print(f'observed SSTs anom: {ds_sst}')

    ds_sst_standard = (ds_sst - ds_sst.mean("time"))/ds_sst.std("time")
    ds_ohtc_anom_standard = (ds_ohtc_anom - ds_ohtc_anom.mean("time"))/ds_ohtc_anom.std("time")
    corr = correlate(ds_sst_standard.values, ds_ohtc_anom_standard.values, mode="full") / ds_sst_standard.values.size
    print(f'ds_sst_standard.values.size: {ds_sst_standard.values.size}\n')
    corr_lags = correlation_lags(ds_sst.size, ds_ohtc_anom.size, mode="full")
    print(f'lag: {corr_lags[np.argmax(corr)]}\n')

    return ds_sst_standard, ds_ohtc_anom_standard, corr, corr_lags

def get_tlag_correlation_oni_mld(timestep, detrend_bool=True):
    """Calculate time-lag correlation between oceanic nino index (oni) and the mixed layer depth anomalies."""

    lat_slice = slice(-50,50)
    lon_slice = slice(None,None) #360-170,360-120)

    startdate = datetime(1981,1,1,0)
    enddate = datetime(2010,1,1,0)

    # Get ohtc anomalies
    ds_mld = get_obs_mld_timeseries(startdate, enddate, timestep)['somxl010']
    ds_mld = ds_mld.sel(lat=lat_slice, lon=lon_slice)
    ds_mld_copy = ds_mld.resample(time_counter="1MS").mean(dim="time_counter")
    ds_mld_copy = ds_mld_copy.groupby("time_counter.month")
    monthly_climo_mld = ds_mld_copy.mean("time_counter")
    ds_mld_anom = (ds_mld_copy - monthly_climo_mld).mean(dim=["lat","lon"])
    print(f'ds_mld_anom : {ds_mld_anom}\n')
    time_index_mld = ds_mld_anom["time_counter"]
    ds_mld_anom = uniform_filter1d(ds_mld_anom.values, size=3, origin=1)

    # Get nino index
    ds_sst = get_obs_sst_timeseries(startdate, enddate, timestep)
    ds_sst = nino_index_monthly_specified_climo(ds_sst, "3.4", ds_sst)
    time_index = ds_sst["time"]
    ds_sst = uniform_filter1d(ds_sst["sst"].values, size=3, origin=1)

    if detrend_bool:
       ds_mld_anom = detrend(ds_mld_anom, type='linear')
       ds_sst = detrend(ds_sst, type='linear')

    ds_mld_anom = xr.DataArray(ds_mld_anom, dims="time", coords={"time": time_index_mld.values}) #, name=date.strftime("%m_%d_%Y_%H"))
    print(f'observed mld anom: {ds_mld_anom}\n')
    ds_sst = xr.DataArray(ds_sst, dims="time", coords={"time": time_index.values}) #, name=date.strftime("%m_%d_%Y_%H"))
    print(f'observed SSTs anom: {ds_sst}')

    ds_sst_standard = (ds_sst - ds_sst.mean("time"))/ds_sst.std("time")
    ds_mld_anom_standard = (ds_mld_anom - ds_mld_anom.mean("time"))/ds_mld_anom.std("time")
    corr = correlate(ds_sst_standard.values, ds_mld_anom_standard.values, mode="full") / ds_sst_standard.values.size
    print(f'ds_sst_standard.values.size: {ds_sst_standard.values.size}\n')
    corr_lags = correlation_lags(ds_sst.size, ds_mld_anom.size, mode="full")
    print(f'lag: {corr_lags[np.argmax(corr)]}\n')

    return ds_sst_standard, ds_mld_anom_standard, corr, corr_lags

def plot_trained_wout():

    path = "/scratch/user/dpp94/ML_SPEEDY_WEIGHTS/worker_0582_ocean_6000_20_20_20_sigma0.5_beta_res0.001_beta_model_1.0_prior_0.0_overlap1_vertlevel_1_precip_epsilon0.001_ohtc_multiple_leakage_test_blkdiagres_.nc"

    ds = xr.open_dataset(path)["wout"]
    print(f'ds: {ds}\n')
    print(f'ds.values: {ds.values}\n')
    #print(f'win_y: {win["win_y"]}\n')
    fig, ax = plt.subplots()
    ax.set_xlabel('outputs')
    ax.set_ylabel('reservoir nodes')
    plt.imshow(ds.values, aspect='auto')
    plt.colorbar()
    plt.show()

def plot_trained_wout_avg():

    root = "/scratch/user/dpp94/ML_SPEEDY_WEIGHTS/"
    workers = np.arange(1,1152)
    ds = []
    for worker in workers:
        filepath = root + "worker_" + str(worker).zfill(4) + "_ocean_6000_20_20_20_sigma0.5_beta_res0.001_beta_model_1.0_prior_0.0_overlap1_vertlevel_1_precip_epsilon0.001_ohtc_multiple_leakage_test_oceantimestep_72hr_train1959_2003_.nc" 
        try:
           temp = xr.open_dataset(filepath)["wout"].values
           if(np.sum(np.isnan(temp)) > 0):
             print(f'worker: {worker}\n')
           if temp.shape[0] == 6016:
              ds.append(np.abs(temp))
        except:
           pass

    print(f'len(ds): {len(ds)}\n')
    print(f'ds[1]: {ds[1]}\n')
    ds = np.stack(ds, axis=0)
    print(f'ds.shape: {ds.shape}\n')
    print(f'ds: {ds}\n')
    ds = np.mean(ds, axis=0)

    fig, ax = plt.subplots()
    ax.set_xlabel('outputs')
    ax.set_ylabel('reservoir nodes')
    plt.imshow(ds, aspect='auto')
    plt.colorbar()
    plt.clim(0,0.09)
    plt.show()      

def create_avg_Wout_and_save():
 
    root = "/scratch/user/dpp94/ML_SPEEDY_WEIGHTS/"
    filepath_end = "_ocean_6000_20_20_20_sigma0.5_beta_res0.001_beta_model_1.0_prior_0.0_overlap1_vertlevel_1_precip_epsilon0.001_ohtc_multiple_leakage_test_oceantimestep_72hr_train1959_2002_.nc" 
    workers = np.arange(1,1152)
    ds = []
    for worker in workers:
        filepath = root + "worker_" + str(worker).zfill(4) +  filepath_end
        try: 
          temp = xr.open_dataset(filepath)["wout"].values
          #print(f'temp: {temp}\n')
          if(np.sum(np.isnan(temp)) > 0):
            print(f'worker: {worker}\n')
          if temp.shape[0] == 6016:
            ds.append(temp)
        except:
          pass

    ds = np.stack(ds, axis=0)
    wout = np.mean(ds, axis=0)
   
    wout = xr.DataArray(data=wout, dims=['woutx','wouty'], name='wout')
    print(f'avg wout: {wout}\n') 
    wout.to_netcdf(path=root+"avg_wout"+filepath_end)


# ------------------------------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------- #

startdates = pd.date_range(start='1/16/2003', end='12/23/2018', freq='30D') #'12/23/2018' 
prediction_length = 8760*2  
timestep = 6

font = {'size'  : 14} #16
mpl.rc('font', **font)

# ----- UNCOMMENT FOR: Pearson-r skill and avg Nino3.4 RMSE ----- #
## Get Hybride PCC and RMSE
##C, C_per, C_std, C_per_std = get_pearson_corr(startdates,prediction_length,timestep,lead_max=12)
##rmse, rmse_per, rmse_climo, rmse_std, rmse_per_std, rmse_climo_std, bias, var = get_nino_index_rmse_new(startdates,prediction_length,timestep,lead_max=12)
##C_std_low = np.array([a[0] for a in C_std])
##C_std_high = np.array([a[1] for a in C_std])
##C_per_std_low = np.array([a[0] for a in C_per_std])
##C_per_std_high = np.array([a[1] for a in C_per_std])
##print(f'\n\nPearson-r: {C}\n\n')
##print(f'\n\nPearson-r_per: {C_per}\n\n')
##print(f'\n\nrmse: {rmse}\n\n')
##print(f'\n\nrmse_per: {rmse_per}\n\n')
##print(f'\n\nrmse_climo: {rmse_climo}\n\n')
##lead = np.arange(1,len(rmse)+1)

## Get NMME PCC and RMSE
##nmme_startdate = '2003-01-01'
##nmme_enddate = '2014-09-01'

##models = ['CanCM4i',
##          'CanSIPS-IC3',
##          'CanSIPSv2',
##          'CMC2-CanCM4',
##          'GEM-NEMO',
##          'NASA-GEOSS2S']

##nmme_accs, nmme_rmses, nmme_biases, nmme_vars = get_avg_nmme_skill(models,startdates,nmme_startdate, nmme_enddate)

## Plot PCC
##fig, ax = plt.subplots()
##p1 = ax.plot(lead, C, '-b', label='Hybrid', linewidth=2)
##ax.fill_between(lead, C_std_low, C_std_high, color='b', alpha=0.1)
##p2 = ax.plot(lead, C_per, '--b', label='Persistence', linewidth=1)
##ax.fill_between(lead, C_per_std_low, C_per_std_high, color='b', alpha=0.1)
##for model in models:
##    acc = nmme_accs[model]
##    x = np.arange(1, len(acc)+1)
##    ax.plot(x, acc, label=model, linewidth=1, alpha=0.75)
##ax.hlines(0.5, 0, 13, colors='k', linestyles='--')
##ax.legend(fontsize=12, framealpha=0.5)
##ax.set_xlabel('Lead time (months)')
##ax.set_ylabel('PCC')
##ax.set_ylim([0, 1])
##ax.set_xlim([1, 12])
##ax.grid()
##plt.tight_layout()
##fig.savefig('ENSO_ACC_with_NMME.svg') #pdf', format='pdf')

## Plot RMSE
##fig, ax = plt.subplots()
##p3 = ax.plot(lead, rmse, '-b', label='Hybrid', linewidth=2)
##ax.fill_between(lead, rmse-rmse_std/2, rmse+rmse_std/2, color='b', alpha=0.1)
##p4 = ax.plot(lead, rmse_per, '--b', label='Persistence', linewidth=1) 
##ax.fill_between(lead, rmse_per-rmse_per_std/2, rmse_per+rmse_per_std/2, color='b', alpha=0.1)
##p5 = ax.plot(lead, rmse_climo, '--k', linewidth=1)
##ax.fill_between(lead, rmse_climo-rmse_climo_std/2, rmse_climo+rmse_climo_std/2, color='k', alpha=0.1)
##for model in models:
##    rmse = nmme_rmses[model]
##    x = np.arange(1, len(rmse)+1)
##    ax.plot(x, rmse, label=model, linewidth=1, alpha=0.75)
##ax.legend(fontsize=12, framealpha=0.5) #, ncols=4, framealpha=1)
##ax.set_xlabel('Lead time (months)')
##ax.set_ylabel('RMSE')
##ax.set_ylim([0, 3])
##ax.set_xlim([1, 12])
##ax.grid()
##plt.tight_layout()
##fig.savefig('ENSO_RMSE_with_NMME.svg') #pdf', format='pdf')

# Plot bias of errors
##fig, ax = plt.subplots()
##lead = np.arange(1, len(bias)+1)
##ax.plot(lead, bias, '-b', label='Hybrid', linewidth=2)
##for model in models:
##    nmme_bias = nmme_biases[model]
##    x = np.arange(1, len(nmme_bias)+1)
##    ax.plot(x, nmme_bias, label=model, linewidth=1, alpha=0.75)
##ax.legend(fontsize=12, framealpha=0.5)
##ax.set_xlabel('Lead time (months)')
##ax.set_ylabel('Bias (K)')
##ax.set_ylim([-2.5, 1.5])
##ax.set_xlim([1, 12])
##ax.grid()
##plt.tight_layout()
##fig.savefig('ENSO_BIAS_with_NMME.svg') #pdf', format='pdf')

# Plot variances of errors 
##fig, ax = plt.subplots()
##lead = np.arange(1, len(var)+1)
##ax.plot(lead, var, '-b', label='Hybrid', linewidth=2)
##for model in models:
##    nmme_var = nmme_vars[model]
##    x = np.arange(1, len(nmme_var)+1)
##    ax.plot(x, nmme_var, label=model, linewidth=1, alpha=0.75)
##ax.legend(fontsize=12, framealpha=0.5)
##ax.set_xlabel('Lead time (months)')
##ax.set_ylabel('Standard Deviation (K)')
##ax.set_ylim([0, 1.5])
##ax.set_xlim([1, 12])
##ax.grid()
##plt.tight_layout()
##fig.savefig('ENSO_STD_with_NMME.svg') #pdf', format='pdf')

# ----- UNCOMMENT FOR: Nino skill contour plot for lead time vs start month ----- #
##ds_hybrid, ds_observed, ds_per = get_predicted_nino34_ens(startdates,prediction_length,timestep)
##pcorr = get_ninoskill_lead_stmon(ds_hybrid, ds_observed)
##pcorr_per = get_ninoskill_lead_stmon(ds_per, ds_observed)
##print(f'pcorr: {pcorr}\n')
##plot_ninoskill_contour(pcorr) #, pcorr_per)

# ----- UNCOMMENT FOR: Plot spatial correlation between nino3.4 index and global sst/precip ----- #
# -- MAKE SURE RESAMPLING IN ANOMALIES IS MONTHLY -- #
##ds_hybrid, ds_obs = get_spatial_var_anom_ds_for_oni_corr(startdates,prediction_length,6,'SST') #'SST','p6hr'
##ds_hybrid_nino, ds_obs_nino = get_nino34_ds(startdates,prediction_length,6)
##calculate_plot_nino_precip_anom(startdates,ds_hybrid,ds_obs,ds_hybrid_nino,ds_obs_nino)
##calculate_plot_nino_precip_anom_all_season(startdates,ds_hybrid,ds_obs,ds_hybrid_nino,ds_obs_nino)

# ----- UNCOMMENT FOR: Plot weekly precip anom PCC and magnitude heatmaps ----- #
# -- MAKE SURE RESAMPLING IN ANOMALIES IS SEASONAL -- #
##ds_hybrid, ds_obs = get_spatial_var_anom_ds(startdates,prediction_length,6,'p6hr')
##plot_spatial_anom_pcc_weekly(ds_hybrid, ds_obs, startdates, sigtest=False)

# Plot forecast errors vs lead time 
# -- MAKE SURE TO REMOVE ANOM CALCULATION IN 'get_spatial_var_anom_ds()' above
##plot_spatial_bias_weekly_djf(ds_hybrid, ds_obs, startdates)

# ----- UNCOMMENT FOR: Plot MJO-related precipitation vs lead time ----- #
# -- MAKE SURE LATS/LONS ARE CORRECT IN BELOW FUNCTIONS -- #
C, C_per, C_std, C_per_std, C_all, C_obs_all, C_climo_all = get_pearson_corr_mjo_precip(startdates,prediction_length,6,lead_max=20)
C_std_low = np.array([a[0] for a in C_std])
C_std_high = np.array([a[1] for a in C_std])
C_per_std_low = np.array([a[0] for a in C_per_std])
C_per_std_high = np.array([a[1] for a in C_per_std])
print(f'\n\nPearson-r: {C}\n\n')
print(f'\n\nPearson-r_per: {C_per}\n\n')
lead = np.arange(5,len(C)+5) #* 5

fig, ax = plt.subplots(figsize=(9,5))
p1 = ax.plot(lead, C, '-b')
ax.fill_between(lead, C_std_low, C_std_high, color='b', alpha=0.1)
p2 = ax.plot(lead, C_per, '--b')
ax.fill_between(lead, C_per_std_low, C_per_std_high, color='b', alpha=0.1)
plt.axhline(y=0.5, color='k', linestyle='--')
ax.set_xlabel('Lead time (days)', fontsize=16)
ax.set_ylabel('PCC',fontsize=16)
ax.grid(True)
ax.set_ylim([0.25, 1])
ax.legend([p1[0],p2[0]],['PCC - Hybrid','PCC - Persistence'],fontsize=16)
ax.tick_params(axis='both', labelsize=16)
plt.tight_layout()
#plt.show()
fig.savefig('WP_Precip_PCC.svg')

leadtimes = [0,3,6,9]
fig, axs = plt.subplots(nrows=2,ncols=2,
                       sharex=True,sharey=True)
axs = axs.flatten()
for i, lead in enumerate(leadtimes):
    data_x, data_y = C_all[lead], C_obs_all[lead]
    axs[i].scatter(data_x, data_y)
    axs[i].grid(True)
    axs[i].set_ylim([-1,1.5])
    axs[i].set_xlim([-0.75,1.25])

    linfit = linregress(data_x, data_y)
    x = np.linspace(-3, 3, 1000)
    #y = linfit.slope * x + linfit.intercept
    y = x

    axs[i].plot(x,y,'-k')
  
    print(f'lead time: {5+i*3}\n') 
    #print(f'r-value: {linfit.rvalue}\n') 

    axs[i].tick_params(axis='both', labelsize=16)
    axs[i].tick_params(axis='both', labelsize=16)

axs[0].set_ylabel('ERA5',fontsize=16)
axs[2].set_ylabel('ERA5',fontsize=16)
axs[2].set_xlabel('Hybrid',fontsize=16)
axs[3].set_xlabel('Hybrid',fontsize=16)

plt.tight_layout()
#plt.show()
fig.savefig('WP_Precip_Scatter.svg')
