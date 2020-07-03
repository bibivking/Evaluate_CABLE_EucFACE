#!/usr/bin/env python

"""
Draw the optimisation of hydraulic conductivity 
"""

__author__  = "MU Mengyuan"
__email__   = 'mengyuan.mu815@gmail.com'

import os
import sys
import glob
import numpy as np
import pandas as pd
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib import ticker
import scipy.stats as stats
from matplotlib import cm
import datetime as dt
import netCDF4 as nc
from sklearn.metrics import mean_squared_error

def plot_2d(x, xaxis, rmse, r):

    # _____________ Make plot _____________
    fig = plt.figure(figsize=(7.2,4.5))
    fig.subplots_adjust(hspace=0.3)
    fig.subplots_adjust(wspace=0.2)

    plt.rcParams['text.usetex']     = False
    plt.rcParams['font.family']     = "sans-serif"
    plt.rcParams['font.serif']      = "Helvetica"
    plt.rcParams['axes.linewidth']  = 1.5
    plt.rcParams['axes.labelsize']  = 14
    plt.rcParams['font.size']       = 14
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 14

    almost_black = '#262626'
    # change the tick colors also to the almost black
    plt.rcParams['ytick.color'] = almost_black
    plt.rcParams['xtick.color'] = almost_black

    # change the text colors also to the almost black
    plt.rcParams['text.color']  = almost_black

    # Change the default axis colors from black to a slightly lighter black,
    # and a little thinner (0.5 instead of 1)
    plt.rcParams['axes.edgecolor']  = almost_black
    plt.rcParams['axes.labelcolor'] = almost_black

    # set the box type of sequence number
    props = dict(boxstyle="round", facecolor='white', alpha=0.0, ec='white')
    # choose colormap
    colors = cm.Set2(np.arange(0,4))

    ax1 = fig.add_subplot(111)

    ax1.plot(x, rmse[1,:], c=colors[0], lw=1.5, ls="-", label="$θ_{all}$", alpha=1.)
    ax1.plot(x, rmse[0,:], c=colors[1], lw=1.5, ls="-", label="$θ_{top}$", alpha=1.)

    ax2  = ax1.twinx()
    ax2.plot(x, rmse[2,:], c=colors[2], lw=1.5, ls="-", label="$E_{tr}$", alpha=1.)
    ax2.plot(x, rmse[3,:], c=colors[3], lw=1.5, ls="-", label="$E_{s}$", alpha=1.)

    ax1.set(xticks=x, xticklabels=xaxis)
    ax1.axis('tight')
    ax1.set_ylim(0.,0.2)
    ax1.axvline(x=1 , ls="--")
    ax1.set_ylabel('RMSE of $θ_{all}$, $θ_{top}$ (m$^{3}$ m$^{-3}$)')
    ax1.legend(loc='upper center', frameon=False)

    ax2.set(xticks=x, xticklabels=xaxis)
    ax2.axis('tight')
    ax2.set_ylim(0.,1.)
    ax2.set_ylabel('RMSE of $E_{tr}$, $E_{s}$ (mm d$^{-1}$)')
    ax2.legend(loc='upper right', frameon=False)

    fig.savefig("./plots/EucFACE_hyds-Opt" , bbox_inches='tight', pad_inches=0.1)

def calc_2d(ref_vars, ring, layer):

    case_name  = "30cm-deep_Ksat_Opt"

    var_values = ["-4","-3","-2","-1","0","1","2","3","4"]
    x          = [-4,-3,-2,-1,0,1,2,3,4]
    xaxis      = ["×0.0001","×0.001","×0.01","×0.1","×1","×10","×100","×1000","×10000"]

    rmse = np.zeros([4,len(var_values)])

    for i in np.arange(len(ref_vars)):

        for j in np.arange(len(var_values)):
            output_file = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run_opt_31uni_hyds-30cm-deep/Opt/outputs/met_LAI-08_vrt_swilt-watr-ssat_31uni_hyds^%s-%s_teuc_sres_watr/EucFACE_amb_out.nc"\
                            % (var_values[j],var_values[j])
            cable_var, obs_var = get_var_value(ref_vars[i], output_file, layer, ring)

            #=========== rmse ============
            rmse[i,j] = np.sqrt(np.mean((obs_var - cable_var)**2))

    np.savetxt("./csv/EucFACE_RMSE_hyds_Opt.csv", rmse, delimiter=",")

    plot_2d(x, xaxis, rmse, r)

def get_var_value(ref_var, output_file, layer, ring):

    if ref_var == 'swc_25':
        cable_var = read_cable_swc_25cm(output_file, layer)
        obs_var   = read_obs_swc_tdr(ring)
    elif ref_var == 'swc_all':
        cable_var = read_cable_swc_all(output_file, layer)
        obs_var   = read_obs_swc_neo(ring)
    elif ref_var == 'trans':
        cable_var = read_cable_var(output_file, 'TVeg')
        obs_var   = read_obs_trans(ring)
    elif ref_var == 'esoil':
        cable_var = read_cable_var(output_file, 'ESoil')
        obs_var   = read_obs_esoil(ring)

    return get_same_dates(cable_var, obs_var)

def get_same_dates(cable_var, obs_var):
    print("carry on get_same_dates")
    cable_var = cable_var['cable'].loc[cable_var.index.isin(obs_var.index)]
    obs_var   = obs_var['obs'].loc[obs_var.index.isin(cable_var.index)]
    mask      = np.any([np.isnan(cable_var), np.isnan(obs_var)],axis=0)

    cable_var = cable_var[mask == False]
    obs_var   = obs_var[mask == False]
    print(cable_var, obs_var)

    return cable_var, obs_var

def read_cable_swc_25cm(output_file, layer):

    """
    read the average swc in top 25cm from CABLE output
    """
    print("carry on read_cable_swc_25cm")

    cable = nc.Dataset(output_file, 'r')
    Time  = nc.num2date(cable.variables['time'][:],cable.variables['time'].units)
    SoilMoist = pd.DataFrame(cable.variables['SoilMoist'][:,0,0,0], columns=['cable'])

    if layer == "6":
        SoilMoist['cable'] = (  cable.variables['SoilMoist'][:,0,0,0]*0.022 \
                                 + cable.variables['SoilMoist'][:,1,0,0]*0.058 \
                                 + cable.variables['SoilMoist'][:,2,0,0]*0.154 \
                                 + cable.variables['SoilMoist'][:,3,0,0]*(0.25-0.022-0.058-0.154) )/0.25
    elif layer == "31uni":
        SoilMoist['cable'] = ( cable.variables['SoilMoist'][:,0,0,0]*0.15 \
                                 + cable.variables['SoilMoist'][:,1,0,0]*0.10 )/0.25

    SoilMoist['Date'] = Time
    SoilMoist = SoilMoist.set_index('Date')
    SoilMoist = SoilMoist.resample("D").agg('mean')
    SoilMoist.index = SoilMoist.index - pd.datetime(2011,12,31)
    SoilMoist.index = SoilMoist.index.days
    SoilMoist = SoilMoist.sort_values(by=['Date'])

    return SoilMoist

def read_cable_swc_all(output_file, layer):

    """
    read swc from CABLE output and calculate the average swc of the whole soil columns
    """

    print("carry on read_cable_swc_all")

    cable = nc.Dataset(output_file, 'r')
    Time  = nc.num2date(cable.variables['time'][:],cable.variables['time'].units)
    SoilMoist = pd.DataFrame(cable.variables['SoilMoist'][:,0,0,0], columns=['cable'])

    SoilMoist['cable'][:] = 0.

    if layer == "6":
        zse       = [ 0.022, 0.058, 0.154, 0.409, 1.085, 2.872 ]
    elif layer == "31uni":
        zse       = [ 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, \
                      0.015, 0.015, 0.015, 0.015, 0.015, 0.015, \
                      0.015, 0.015, 0.015, 0.015, 0.015, 0.015, \
                      0.015, 0.015, 0.015, 0.015, 0.015, 0.015, \
                      0.015, 0.015, 0.015, 0.015, 0.015, 0.015, \
                      0.015 ]

    for i in np.arange(len(zse)):
        SoilMoist['cable'][:] =  SoilMoist['cable'][:] + cable.variables['SoilMoist'][:,i,0,0]*zse[i]

    SoilMoist['cable'][:] = SoilMoist['cable'][:]/sum(zse)

    SoilMoist['Date'] = Time
    SoilMoist = SoilMoist.set_index('Date')
    SoilMoist = SoilMoist.resample("D").agg('mean')
    SoilMoist.index = SoilMoist.index - pd.datetime(2011,12,31)
    SoilMoist.index = SoilMoist.index.days
    SoilMoist = SoilMoist.sort_values(by=['Date'])

    return SoilMoist

def read_cable_var(output_file, var_name):

    """
    read transpiration or soil evaporation from CABLE output
    """

    print("carry on read_cable_var")
    cable = nc.Dataset(output_file, 'r')
    Time  = nc.num2date(cable.variables['time'][:],cable.variables['time'].units)

    var = pd.DataFrame(cable.variables[var_name][:,0,0]*1800., columns=['cable'])

    var['Date'] = Time
    var = var.set_index('Date')
    var = var.resample("D").agg('sum')
    var.index = var.index - pd.datetime(2011,12,31)
    var.index = var.index.days
    var = var.sort_values(by=['Date'])
    print(var)

    return var

def read_obs_swc_tdr(ring):

    """
    read the 25 cm swc from tdr observation
    """

    fobs = "/srv/ccrc/data25/z5218916/data/Eucface_data/swc_average_above_the_depth/swc_tdr.csv"
    tdr = pd.read_csv(fobs, usecols = ['Ring','Date','swc.tdr'])
    tdr['Date'] = pd.to_datetime(tdr['Date'],format="%Y-%m-%d",infer_datetime_format=False)
    tdr['Date'] = tdr['Date'] - pd.datetime(2011,12,31)
    tdr['Date'] = tdr['Date'].dt.days
    tdr = tdr.sort_values(by=['Date'])
    # divide neo into groups
    if ring == 'amb':
        subset = tdr[(tdr['Ring'].isin(['R2','R3','R6'])) & (tdr.Date > 366)]
    elif ring == 'ele':
        subset = tdr[(tdr['Ring'].isin(['R1','R4','R5'])) & (tdr.Date > 366)]
    else:
        subset = tdr[(tdr['Ring'].isin([ring]))  & (tdr.Date > 366)]

    subset = subset.groupby(by=["Date"]).mean()/100.

    subset['swc.tdr'] = subset['swc.tdr'].clip(lower=0.)
    subset['swc.tdr'] = subset['swc.tdr'].replace(0., float('nan'))
    subset = subset.rename({'swc.tdr': 'obs'}, axis='columns')
    print(subset)

    return subset

def read_obs_swc_neo(ring):

    """
    read the neo swc observation and calculate the soil columns average
    """

    fobs = "/srv/ccrc/data25/z5218916/data/Eucface_data/swc_at_depth/FACE_P0018_RA_NEUTRON_20120430-20190510_L1.csv"
    neo = pd.read_csv(fobs, usecols = ['Ring','Depth','Date','VWC'])

    neo['Date'] = pd.to_datetime(neo['Date'],format="%d/%m/%y",infer_datetime_format=False)
    neo['Date'] = neo['Date'] - pd.datetime(2011,12,31)
    neo['Date'] = neo['Date'].dt.days
    neo = neo.sort_values(by=['Date','Depth'])

    if ring == 'amb':
        subset = neo[(neo['Ring'].isin(['R2','R3','R6'])) & (neo.Date > 366)]
    elif ring == 'ele':
        subset = neo[(neo['Ring'].isin(['R1','R4','R5'])) & (neo.Date > 366)]
    else:
        subset = neo[(neo['Ring'].isin(['Ring'])) & (neo.Date > 366)]
    print("------", subset)
    subset = subset.groupby(by=["Depth","Date"]).mean()
    subset[:] = subset[:]/100.
    subset['VWC'] = subset['VWC'].clip(lower=0.)
    subset['VWC'] = subset['VWC'].replace(0., float('nan'))

    zse_obs = [0.375, 0.25, 0.25, 0.25, 0.25, 0.375,\
               0.5, 0.5, 0.5, 0.5, 0.5, 0.35 ]
    layer_cm = [25, 50, 75, 100, 125, 150, 200, 250,\
                300, 350, 400, 450]

    neo_obs = subset.loc[25]

    neo_obs['VWC'][:] = 0.
    for i in np.arange(len(zse_obs)):
        print("i = ", i )
        print(subset.loc[layer_cm[i]]['VWC'])
        neo_obs['VWC'][:] = neo_obs['VWC'][:] + subset.loc[layer_cm[i]]['VWC']*zse_obs[i]
    neo_obs['VWC'][:] = neo_obs['VWC'][:]/4.6

    neo_obs = neo_obs.rename({'VWC' : 'obs'}, axis='columns')
    print(neo_obs)

    return neo_obs

def read_obs_trans(ring):

    """
    read transpiration from observation, in G 2016
    """

    print("carry on read_obs_trans")

    fobs = "/srv/ccrc/data25/z5218916/data/Eucface_data/FACE_PACKAGE_HYDROMET_GIMENO_20120430-20141115/data/Gimeno_wb_EucFACE_sapflow.csv"
    est_trans = pd.read_csv(fobs, usecols = ['Ring','Date','volRing'])
    est_trans['Date'] = pd.to_datetime(est_trans['Date'],format="%d/%m/%Y",infer_datetime_format=False)
    est_trans['Date'] = est_trans['Date'] - pd.datetime(2011,12,31)
    est_trans['Date'] = est_trans['Date'].dt.days
    est_trans = est_trans.sort_values(by=['Date'])

    # divide neo into groups
    if ring == 'amb':
       subs = est_trans[(est_trans['Ring'].isin(['R2','R3','R6'])) & (est_trans.Date > 366)]
    elif ring == 'ele':
       subs = est_trans[(est_trans['Ring'].isin(['R1','R4','R5'])) & (est_trans.Date > 366)]
    else:
       subs = est_trans[(est_trans['Ring'].isin([ring]))  & (est_trans.Date > 366)]

    subs = subs.groupby(by=["Date"]).mean()
    subs['volRing']   = subs['volRing'].clip(lower=0.)
    subs['volRing']   = subs['volRing'].replace(0., float('nan'))

    subs = subs.rename({'volRing' : 'obs'}, axis='columns')

    print(subs)

    return subs

def read_obs_esoil(ring):

    """
    read soil evaporation from observation, in G 2016
    """

    print("carry on read_obs_esoil")

    fobs = "/srv/ccrc/data25/z5218916/data/Eucface_data/FACE_PACKAGE_HYDROMET_GIMENO_20120430-20141115/data/Gimeno_wb_EucFACE_underET.csv"
    est_esoil = pd.read_csv(fobs, usecols = ['Ring','Date','wuTP'])
    est_esoil['Date'] = pd.to_datetime(est_esoil['Date'],format="%d/%m/%Y",infer_datetime_format=False)
    est_esoil['Date'] = est_esoil['Date'] - pd.datetime(2011,12,31)
    est_esoil['Date'] = est_esoil['Date'].dt.days
    est_esoil = est_esoil.sort_values(by=['Date'])
    # divide neo into groups
    if ring == 'amb':
       subs = est_esoil[(est_esoil['Ring'].isin(['R2','R3','R6'])) & (est_esoil.Date > 366)]
    elif ring == 'ele':
       subs = est_esoil[(est_esoil['Ring'].isin(['R1','R4','R5'])) & (est_esoil.Date > 366)]
    else:
       subs = est_esoil[(est_esoil['Ring'].isin([ring]))  & (est_esoil.Date > 366)]

    subs = subs.groupby(by=["Date"]).mean()
    subs['wuTP']   = subs['wuTP'].clip(lower=0.)
    subs['wuTP']   = subs['wuTP'].replace(0., float('nan'))

    subs = subs.rename({'wuTP' : 'obs'}, axis='columns')

    return subs

if __name__ == "__main__":

    layer    = "31uni"
    ring     = "amb"
    ref_vars = ['swc_25','swc_all','trans','esoil']

    calc_2d(ref_vars,ring,layer)
