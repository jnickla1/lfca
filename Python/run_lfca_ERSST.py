#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 17:58:52 2018

@author: Zhaoyi.Shen
"""

import sys
import os
cur_path = os.path.dirname(os.path.realpath(__file__))
#sys.path.append('/home/z1s/py/lib/')
from signal_processing import lfca
import numpy as np
import scipy as sp
from scipy import io
from matplotlib import pyplot as plt


filename = cur_path+'/../ERSST_1900_2016.mat'
mat = io.loadmat(filename)

lat_axis = mat['LAT_AXIS']
lon_axis = mat['LON_AXIS']
sst = mat['SST']
nlon = sst.shape[0]
nlat = sst.shape[1]
ntime = sst.shape[2]
time = np.arange(1900,2016.99,1/12.)

cutoff = 120
truncation = 30
#%%
mean_seasonal_cycle = np.zeros((nlon,nlat,12))
sst_anomalies0 = np.zeros((nlon,nlat,ntime))
for i in range(12):
    mean_seasonal_cycle[...,i] = np.nanmean(sst[...,i:ntime:12],-1)
    sst_anomalies0[...,i:ntime:12] = sst[...,i:ntime:12] - mean_seasonal_cycle[...,i][...,np.newaxis]
#%%

dir_path = os.path.dirname(os.path.realpath(__file__))
load_file = dir_path+"/../ERSST_firstLFCA_s1940.npy"
monthly_full = np.full((117-40, ntime ), np.nan)
for i in range(40,117):
    print(i)
    thisrun_row = i-40
    thisrun_ntime= i*12
    sst_anomalies = sst_anomalies0[...,0:thisrun_ntime] #if we restrict to only 10 years - get el nino as major signal, so start with 40
    s = sst_anomalies.shape
    y, x = np.meshgrid(lat_axis,lon_axis)
    area = np.cos(y*np.pi/180.)
    area[np.where(np.isnan(np.mean(sst_anomalies,-1)))] = 0
    #%%
    domain = np.ones(area.shape)
    domain[np.where(x<100)] = 0
    domain[np.where((x<103) & (y<5))] = 0
    domain[np.where((x<105) & (y<2))] = 0
    domain[np.where((x<111) & (y<-6))] = 0
    domain[np.where((x<114) & (y<-7))] = 0
    domain[np.where((x<127) & (y<-8))] = 0
    domain[np.where((x<147) & (y<-18))] = 0      
    domain[np.where(y>70)] = 0
    domain[np.where((y>65) & ((x<175) | (x>200)))] = 0
    domain[np.where(y<-45)] = 0
    domain[np.where((x>260) & (y>17))] = 0
    domain[np.where((x>270) & (y<=17) & (y>14))] = 0
    domain[np.where((x>276) & (y<=14) & (y>9))] = 0
    domain[np.where((x>290) & (y<=9))] = 0
    #%%
    order = 'C'
    x = np.transpose(np.reshape(sst_anomalies,(s[0]*s[1],s[2]),order=order))
    area_weights = np.transpose(np.reshape(area,(s[0]*s[1],1),order=order))
    domain = np.transpose(np.reshape(domain,(s[0]*s[1],1),order=order))
    icol_ret = np.where((area_weights!=0) & (domain!=0))
    icol_disc = np.where((area_weights==0) | (domain==0))
    x = x[:,icol_ret[1]]
    area_weights = area_weights[:,icol_ret[1]]
    normvec = np.transpose(area_weights)/np.sum(area_weights)
    scale = np.sqrt(normvec)
    #%%
    lfcs, lfps, weights, r, pvar, pcs, eofs, ntr, pvar_slow, pvar_lfc, r_eofs, pvar_slow_eofs = \
    lfca(x, cutoff, truncation, scale)
    #%%
    nins = np.size(icol_disc[1])
    nrows = lfps.shape[0]
    lfps_aug = np.zeros((nrows,lfps.shape[1]+nins))
    lfps_aug[:] = np.nan
    lfps_aug[:,icol_ret[1]] = lfps
    nrows = eofs.shape[0]
    eofs_aug = np.zeros((nrows,eofs.shape[1]+nins))
    eofs_aug[:] = np.nan
    eofs_aug[:,icol_ret[1]] = eofs
    #%%
    s1 = np.size(lon_axis)
    s2 = np.size(lat_axis)
    imode = 0
   # pattern = np.reshape(lfps_aug[imode,...],(s1,s2),order=order)
   # pattern[np.where(np.abs(pattern)>1.e5)] = np.nan
    monthly_full[thisrun_row, 0:thisrun_ntime] = lfcs[:,imode]

    
#plt.figure()
#plt.contourf(np.squeeze(lon_axis),np.squeeze(lat_axis),np.transpose(pattern),\
#             np.arange(-1,1.1,0.1),cmap=plt.cm.RdYlBu_r)
#plt.figure()
#plt.plot(lfcs[:,i])

#We aren't saving the area-weighted pattern average, just not worth it because we are fitting in the next time step anyway.
np.save(load_file, monthly_full)


#John Nicklas additions
if False:
    domain_sq = np.reshape(domain,(s1,s2),order=order)
    domain_sq2 = domain_sq* (~np.isnan(pattern))
    ar_pat = np.nansum(pattern*area)
    ar_domain = np.nansum(domain_sq2*area)
    area_av_pattern = ar_pat/ar_domain
    from PIL import Image
    import netCDF4
    data_path = '/Users/JohnMatthew/Downloads/Thorne_15_codefigurestats/Common_Data'
    filein = netCDF4.Dataset(data_path+"/HadCRUT.5.0.2.0.analysis.anomalies.ensemble_mean.nc",'r')
    tas0 = np.array(filein.variables['tas_mean'])
    tas= tas0.copy()
    tas[tas0<-100]=np.nan
    domain_sq2_image = Image.fromarray(domain_sq2.astype(np.uint8) * 255)
    target_shape = (np.shape(tas)[1], np.shape(tas)[2])
    resized_image = domain_sq2_image.resize(target_shape, resample=Image.BILINEAR)
    resampled_mask = np.fliplr(np.array(resized_image)) / 255
    lat5 = np.array(filein.variables['latitude'])
    lon5 = np.array(filein.variables['longitude'])
    plt.contourf(np.squeeze(lon5)+180,np.squeeze(lat5),resampled_mask.T,\
                 np.arange(0,1.5,0.1),cmap=plt.cm.RdYlBu_r)
    cos_lat5_weights = np.cos(np.radians(lat5))  # Compute cosine weights for each latitude
    # Apply the weights to the pattern
    weighted_pattern = pattern * cos_lat5_weights
    tot_weight = np.nansum((~np.isnan(tas))*weighted_pattern.T,axis=(1,2))
    tas_wrtT  = np.nansum(tas*weighted_pattern.T,axis=(1,2))/tot_weight
    
    tas_1875_1925 = np.mean(tas_wrtT[(1875-1850)*12 : (1925-1850)*12])
    tas_1900_1912 = np.mean(tas_wrtT[(1900-1850)*12 : (1912-1850)*12])
    tas_2006_2016 = np.mean(tas_wrtT[(2010-1850)*12 : (2022-1850)*12])
    print(tas_2006_2016-tas_1900_1912)
    lfcs_2006_2016 = np.mean(lfcs[(2006-1900)*12 :(2016-1900)*12,0])
    lfcs_1900_1912 = np.mean(lfcs[:(1912-1900)*12,0])
    print(lfcs_2006_2016 - lfcs_1900_1912)
    import pandas as pd
    data = pd.read_csv(data_path+"/HadCRUT.5.0.2.0.analysis.summary_series.global.monthly.csv")
    temps_obs_mothly = data.loc[:,"Anomaly (deg C)"].to_numpy()
    temps_cropped = temps_obs_mothly[(1900-1850)*12 : (2017-1850)*12]
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(lfcs[:,i], temps_cropped)
    #slope 0.3114061701545104
    predicted_temps = np.real(slope) * lfcs[:,i] + intercept
    residuals = temps_cropped - predicted_temps
    residual_std_dev = np.std(residuals)
    #residual_std_dev 0.16949657978893795
