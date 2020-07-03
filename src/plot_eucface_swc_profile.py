#!/usr/bin/env python

"""
Draw Etr Es & θ

Include functions:

    plot_profile_tdr_ET_error_rain
    plot_profile_ET_error_rain

"""

__author__ = "MU Mengyuan"
__email__  = "mu.mengyuan815@gmail.com"

import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib import ticker
import datetime as dt
import netCDF4 as nc
from scipy.interpolate import griddata
from plot_eucface_get_var import *
import matplotlib.font_manager
from matplotlib import rc

def plot_profile_tdr_ET_error_rain(CTL, fpath, case_name, ring, contour, layer):

    """
    plot simulation status and fluxes
    """

    # ========================= ET FLUX  ============================

    # ===== CABLE =====
    cable = nc.Dataset("%s/EucFACE_amb_out.nc" % fpath, 'r')
    Time  = nc.num2date(cable.variables['time'][:],cable.variables['time'].units,
            calendar="standard")

    Rainf = pd.DataFrame(cable.variables['Rainf'][:,0,0],columns=['Rainf'])
    Rainf = Rainf*1800.
    Rainf['dates'] = Time

    Rainf = Rainf.set_index('dates')
    Rainf = Rainf.resample("D").agg('sum')
    Rainf.index = Rainf.index - pd.datetime(2011,12,31)
    Rainf.index = Rainf.index.days

    TVeg = pd.DataFrame(cable.variables['TVeg'][:,0,0],columns=['TVeg'])
    TVeg = TVeg*1800.
    TVeg['dates'] = Time
    TVeg = TVeg.set_index('dates')
    TVeg = TVeg.resample("D").agg('sum')
    TVeg.index = TVeg.index - pd.datetime(2011,12,31)
    TVeg.index = TVeg.index.days

    ESoil = pd.DataFrame(cable.variables['ESoil'][:,0,0],columns=['ESoil'])
    ESoil = ESoil*1800.
    ESoil['dates'] = Time
    ESoil = ESoil.set_index('dates')
    ESoil = ESoil.resample("D").agg('sum')
    ESoil.index = ESoil.index - pd.datetime(2011,12,31)
    ESoil.index = ESoil.index.days

    T  = np.zeros([3,len(TVeg)])
    Es = np.zeros([3,len(ESoil)])

    T[0,:]  =  read_cable_var("%s/EucFACE_R2_out.nc" % fpath, 'TVeg')['cable'].values
    Es[0,:] =  read_cable_var("%s/EucFACE_R2_out.nc" % fpath, 'ESoil')['cable'].values
    T[1,:]  =  read_cable_var("%s/EucFACE_R3_out.nc" % fpath, 'TVeg')['cable'].values
    Es[1,:] =  read_cable_var("%s/EucFACE_R3_out.nc" % fpath, 'ESoil')['cable'].values
    T[2,:]  =  read_cable_var("%s/EucFACE_R6_out.nc" % fpath, 'TVeg')['cable'].values
    Es[2,:] =  read_cable_var("%s/EucFACE_R6_out.nc" % fpath, 'ESoil')['cable'].values

    TVeg['min']  = T.min(axis=0)
    TVeg['max']  = T.max(axis=0)
    ESoil['min'] = Es.min(axis=0)
    ESoil['max'] = Es.max(axis=0)

    # ===== Obs   =====
    subs_Esoil = read_obs_esoil(ring)
    subs_Trans = read_obs_trans(ring)

    Es_R2 = read_obs_esoil('R2')['obs']
    Es_R3 = read_obs_esoil('R3')['obs']
    Es_R6 = read_obs_esoil('R6')['obs']

    T_R2 = read_obs_trans('R2')['obs']
    T_R3 = read_obs_trans('R3')['obs']
    T_R6 = read_obs_trans('R6')['obs']

    T_error = np.zeros([3,len(TVeg)])
    Es_error = np.zeros([3,len(TVeg)])

    for date in TVeg.index:
        if np.any(Es_R2.index == date):
            Es_error[0,date-367] = Es_R2[Es_R2.index == date].values
        else:
            Es_error[0,date-367] = float('NaN')
        if np.any(Es_R3.index == date):
            Es_error[1,date-367] = Es_R3[Es_R3.index == date].values
        else:
            Es_error[1,date-367] = float('NaN')
        if np.any(Es_R6.index == date):
            Es_error[2,date-367] = Es_R6[Es_R6.index == date].values
        else:
            Es_error[2,date-367] = float('NaN')

        if np.any(T_R2.index == date):
            T_error[0,date-367] = T_R2[T_R2.index == date].values
        else:
            T_error[0,date-367] = float('NaN')
        if np.any(T_R3.index == date):
            T_error[1,date-367] = T_R3[T_R3.index == date].values
        else:
            T_error[1,date-367] = float('NaN')
        if np.any(T_R6.index == date):
            T_error[2,date-367] = T_R6[T_R6.index == date].values
        else:
            T_error[2,date-367] = float('NaN')

    ESoil['obs_min'] = Es_error.min(axis=0)
    ESoil['obs_max'] = Es_error.max(axis=0)
    TVeg['obs_min']  = T_error.min(axis=0)
    TVeg['obs_max']  = T_error.max(axis=0)

    # ========================= SM IN TOP 25cm ==========================
    SoilMoist_25cm = pd.DataFrame(cable.variables['SoilMoist'][:,0,0,0], columns=['SoilMoist'])

    if layer == "6":
        SoilMoist_25cm['SoilMoist'] = ( cable.variables['SoilMoist'][:,0,0,0]*0.022 \
                                      + cable.variables['SoilMoist'][:,1,0,0]*0.058 \
                                      + cable.variables['SoilMoist'][:,2,0,0]*0.154 \
                                      + cable.variables['SoilMoist'][:,3,0,0]*(0.25-0.022-0.058-0.154) )/0.25
    elif layer == "31uni":
        SoilMoist_25cm['SoilMoist'] = ( cable.variables['SoilMoist'][:,0,0,0]*0.15 \
                                      + cable.variables['SoilMoist'][:,1,0,0]*0.10 )/0.25

    SoilMoist_25cm['dates'] = Time
    SoilMoist_25cm = SoilMoist_25cm.set_index('dates')
    SoilMoist_25cm = SoilMoist_25cm.resample("D").agg('mean')
    SoilMoist_25cm.index = SoilMoist_25cm.index - pd.datetime(2011,12,31)
    SoilMoist_25cm.index = SoilMoist_25cm.index.days
    SoilMoist_25cm = SoilMoist_25cm.sort_values(by=['dates'])

    # Soil hydraulic param
    swilt = np.zeros(len(TVeg))
    sfc = np.zeros(len(TVeg))
    ssat = np.zeros(len(TVeg))

    if layer == "6":
        swilt[:] = ( cable.variables['swilt'][0]*0.022 + cable.variables['swilt'][1]*0.058 \
                   + cable.variables['swilt'][2]*0.154 + cable.variables['swilt'][3]*(0.25-0.022-0.058-0.154) )/0.25
        sfc[:] = ( cable.variables['sfc'][0]*0.022   + cable.variables['sfc'][1]*0.058 \
                   + cable.variables['sfc'][2]*0.154 + cable.variables['sfc'][3]*(0.25-0.022-0.058-0.154) )/0.25
        ssat[:] = ( cable.variables['ssat'][0]*0.022 + cable.variables['ssat'][1]*0.058 \
                   + cable.variables['ssat'][2]*0.154+ cable.variables['ssat'][3]*(0.25-0.022-0.058-0.154) )/0.25
    elif layer == "31uni":
        swilt[:] =(cable.variables['swilt'][0]*0.15 + cable.variables['swilt'][1]*0.10 )/0.25
        sfc[:] =(cable.variables['sfc'][0]*0.15 + cable.variables['sfc'][1]*0.10 )/0.25
        ssat[:] =(cable.variables['ssat'][0]*0.15 + cable.variables['ssat'][1]*0.10 )/0.25

    # ========================= SM PROFILE ==========================
    ### 1. Read data
    # ==== SM Obs ====
    neo = read_obs_swc_neo(ring)
    tdr = read_obs_swc_tdr(ring)

    # ===== CABLE =====
    SoilMoist = read_cable_SM_one_clmn("%s/EucFACE_amb_out.nc" % fpath, layer)

    ### 2. interpolate SM
    # === Obs SoilMoist ===
    x     = np.concatenate((neo[(25)].index.values,               \
                            neo.index.get_level_values(1).values, \
                            neo[(450)].index.values ))
    y     = np.concatenate(([0]*len(neo[(25)]),                  \
                            neo.index.get_level_values(0).values, \
                            [460]*len(neo[(25)])    ))
    value =  np.concatenate((neo[(25)].values, neo.values, neo[(450)].values))

    X     = neo[(25)].index.values[20:]
    Y     = np.arange(0,461,1)

    grid_X, grid_Y = np.meshgrid(X,Y)

    # === CABLE SoilMoist ===
    if contour:
        grid_data = griddata((x, y) , value, (grid_X, grid_Y), method='cubic')
    else:
        grid_data = griddata((x, y) , value, (grid_X, grid_Y), method='linear') # 'linear' 'nearest'

    ntimes      = len(np.unique(SoilMoist['dates']))
    dates       = np.unique(SoilMoist['dates'].values)

    x_cable     = np.concatenate(( dates, SoilMoist['dates'].values,dates)) # Time
    y_cable     = np.concatenate(([0]*ntimes,SoilMoist['Depth'].values,[460]*ntimes))# Depth
    value_cable = np.concatenate(( SoilMoist.iloc[:ntimes,2].values, \
                                   SoilMoist.iloc[:,2].values,         \
                                   SoilMoist.iloc[-(ntimes):,2].values ))
    value_cable = value_cable*100.

    # add the 12 depths to 0
    grid_X_cable, grid_Y_cable = np.meshgrid(X,Y)

    if contour:
        grid_cable = griddata((x_cable, y_cable) , value_cable, (grid_X_cable, grid_Y_cable),\
                 method='cubic')
    else:
        grid_cable = griddata((x_cable, y_cable) , value_cable, (grid_X_cable, grid_Y_cable),\
                 method='linear')
    difference = grid_cable -grid_data


    # ======================= PLOTTING  ==========================

    if case_name == CTL:
        fig = plt.figure(figsize=[9,17.5])
    else:
        fig = plt.figure(figsize=[9,14])

    fig.subplots_adjust(hspace=0.15)
    fig.subplots_adjust(wspace=0.05)

    plt.rcParams['text.usetex']     = False
    plt.rcParams['font.family']     = "sans-serif"
    plt.rcParams['font.serif']      = "Helvetica"
    plt.rcParams['axes.linewidth']  = 1.5
    plt.rcParams['axes.labelsize']  = 14
    plt.rcParams['font.size']       = 14
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['xtick.labelsize'] = 14
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

    props = dict(boxstyle="round", facecolor='white', alpha=0.0, ec='white')

    if case_name == CTL:
        ax1 = fig.add_subplot(511)
        ax2 = fig.add_subplot(512)
        ax5 = fig.add_subplot(513)
        ax3 = fig.add_subplot(514)
        ax4 = fig.add_subplot(515)
    else:
        ax1 = fig.add_subplot(411)
        ax2 = fig.add_subplot(412)
        ax3 = fig.add_subplot(413)
        ax4 = fig.add_subplot(414)

    x = TVeg.index

    # set x-axis values
    cleaner_dates1 = ["2013","2014","2015","2016","2017","2018","2019"]
    xtickslocs1    = [367,732,1097,1462,1828,2193,2558]
    # set color
    cmap = plt.cm.viridis_r

    ax1.fill_between(x, TVeg['min'].rolling(window=3).mean(),
        TVeg['max'].rolling(window=3).mean(), color="green", alpha=0.2)
    ax1.fill_between(x, ESoil['min'].rolling(window=3).mean(),
        ESoil['max'].rolling(window=3).mean(), color="orange", alpha=0.2)

    ax1.plot(x, TVeg['TVeg'].rolling(window=3).mean(),
        c="green", lw=1.0, ls="-", label="$E_{tr}$ (CABLE)")
    ax1.plot(x, ESoil['ESoil'].rolling(window=3).mean(),
        c="orange", lw=1.0, ls="-", label="$E_{s}$ (CABLE)")

    ax1.fill_between(x, TVeg['obs_min'].rolling(window=3).mean(),
        TVeg['obs_max'].rolling(window=3).mean(), color="blue", alpha=0.2)
    ax1.fill_between(x, ESoil['obs_min'].rolling(window=3).mean(),
        ESoil['obs_max'].rolling(window=3).mean(), color="red", alpha=0.2)
    ax1.scatter(subs_Trans.index, subs_Trans['obs'].rolling(window=3).mean(),
        marker='o', c='',edgecolors='blue', s = 2., label="$E_{tr}$ (Obs)")
    ax1.scatter(subs_Esoil.index, subs_Esoil['obs'].rolling(window=3).mean(),
        marker='o', c='',edgecolors='red', s = 2., label="$E_{s}$ (Obs)")
    ax1.text(0.02, 0.95, '(a)', transform=ax1.transAxes, fontsize=14, verticalalignment='top', bbox=props)

    if case_name == CTL:

        # this order of the setting can affect plot x & y axis
        plt.setp(ax1.get_xticklabels(), visible=True)
        ax1.set(xticks=xtickslocs1, xticklabels=cleaner_dates1)
        ax1.set_ylabel("$E_{tr}$, $E_{s}$ (mm d$^{-1}$)")
        ax1.axis('tight')
        ax1.set_ylim(0.,7.)
        ax1.set_xlim(367,1097)
        ax1.legend(loc='upper right', ncol=2, labelspacing=0.2, columnspacing=0.2, frameon=False)

        ax6  = ax1.twinx()
        ax6.set_ylabel('$P$ (mm d$^{-1}$)')
        ax6.bar(x, -Rainf['Rainf'],  1., color='gray', alpha = 0.5, label='Rainfall')
        ax6.set_ylim(-220.,0)
        ax6.set_xlim(367,1097)
        y_ticks      = [-200,-150,-100,-50,0.]
        y_ticklabels = ['200','150','100','50','0']
        ax6.set_yticks(y_ticks)
        ax6.set_yticklabels(y_ticklabels)
        ax6.get_xaxis().set_visible(False)
    else:
        # this order of the setting can affect plot x & y axis
        plt.setp(ax1.get_xticklabels(), visible=True)
        ax1.set(xticks=xtickslocs1, xticklabels=cleaner_dates1) ####
        ax1.set_ylabel("$E_{tr}$, $E_{s}$ (mm d$^{-1}$)")
        ax1.axis('tight')
        ax1.set_ylim(0.,4.)
        ax1.set_xlim(367,1097)#2923)
        ax1.legend(loc='upper right', ncol=2,labelspacing=0.2, columnspacing=0.2, frameon=False)

    ax2.plot(tdr.index, tdr.values,    c="orange", lw=1.0, ls="-", label="$θ$ (Obs)")
    ax2.plot(x, SoilMoist_25cm.values, c="green", lw=1.0, ls="-", label="$θ$ (CABLE)")
    ax2.plot(x, swilt,                 c="black", lw=1.0, ls="-", label="$θ_{w}$")
    ax2.plot(x, sfc,                   c="black", lw=1.0, ls="-.", label="$θ_{fc}$")
    ax2.plot(x, ssat,                  c="black", lw=1.0, ls=":", label="$θ_{sat}$")
    ax2.text(0.02, 0.95, '(b)', transform=ax2.transAxes, fontsize=14, verticalalignment='top', bbox=props)

    plt.setp(ax2.get_xticklabels(), visible=True)
    ax2.set(xticks=xtickslocs1, xticklabels=cleaner_dates1)
    ax2.set_ylabel("$θ$ in 0.25m (m$^{3}$ m$^{-3}$)")
    ax2.axis('tight')
    ax2.set_ylim(0,0.5)
    ax2.set_xlim(367,2922)
    ax2.legend(loc='upper right', ncol=2, labelspacing=0.2, columnspacing=0.2, frameon=False)

    cleaner_dates  = ["2013","2014","2015","2016","2017","2018","2019"]
    xtickslocs     = [1,19,37,52,66,74,86]
    yticks         = [360,260,160,60]
    yticklabels    = ["100","200","300","400"]

    if case_name == CTL:

        if contour:
            levels = np.arange(0.,0.52,0.02)
            img = ax5.contourf(grid_data/100., cmap=cmap, origin="upper", levels=levels)
            Y_labels = np.flipud(Y)
        else:
            img = ax5.imshow(grid_data/100., cmap=cmap, vmin=0, vmax=0.52, origin="upper", interpolation='nearest')
            Y_labels = Y

        cbar = fig.colorbar(img, ax = ax5, orientation="vertical", pad=0.02, shrink=.6)
        cbar.set_label('$θ$ Obs (m$^{3}$ m$^{-3}$)')
        tick_locator = ticker.MaxNLocator(nbins=5)
        cbar.locator = tick_locator
        cbar.update_ticks()

        ax5.text(0.02, 0.95, '(c)', transform=ax5.transAxes, fontsize=14, verticalalignment='top', bbox=props)

        # every second tick
        ax5.set_yticks(yticks)
        ax5.set_yticklabels(yticklabels)
        plt.setp(ax5.get_xticklabels(), visible=False)

        ax5.set(xticks=xtickslocs, xticklabels=cleaner_dates)
        ax5.set_ylabel("Depth (cm)")
        ax5.axis('tight')

    if contour:
        levels = np.arange(0.,0.52,0.02)
        img2 = ax3.contourf(grid_cable/100., cmap=cmap, origin="upper", levels=levels,interpolation='nearest')
        Y_labels2 = np.flipud(Y)
    else:
        img2 = ax3.imshow(grid_cable/100., cmap=cmap, vmin=0., vmax=0.52, origin="upper", interpolation='nearest')
        Y_labels2 = Y

    cbar2 = fig.colorbar(img2, ax = ax3,  orientation="vertical", pad=0.02, shrink=.6)
    cbar2.set_label('$θ$ CABLE (m$^{3}$ m$^{-3}$)')
    tick_locator2 = ticker.MaxNLocator(nbins=5)
    cbar2.locator = tick_locator2
    cbar2.update_ticks()
    if case_name == CTL:
        ax3.text(0.02, 0.95, '(d)', transform=ax3.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    else:
        ax3.text(0.02, 0.95, '(c)', transform=ax3.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    # every second tick
    ax3.set_yticks(yticks)
    ax3.set_yticklabels(yticklabels)
    plt.setp(ax3.get_xticklabels(), visible=False)

    ax3.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax3.set_ylabel("Depth (cm)")
    ax3.axis('tight')

    cmap = plt.cm.BrBG

    if contour:
        levels = np.arange(-0.30,0.30,0.02)
        img3 = ax4.contourf(difference/100., cmap=cmap, origin="upper", levels=levels)
        Y_labels3 = np.flipud(Y)
    else:
        img3 = ax4.imshow(difference/100., cmap=cmap, vmin=-0.30, vmax=0.30, origin="upper", interpolation='nearest')
        Y_labels3 = Y

    cbar3 = fig.colorbar(img3, ax = ax4, orientation="vertical", pad=0.02, shrink=.6)
    cbar3.set_label('$θ$ (CABLE − Obs) (m$^{3}$ m$^{-3}$)')
    tick_locator3 = ticker.MaxNLocator(nbins=6)
    cbar3.locator = tick_locator3
    cbar3.update_ticks()

    if case_name == CTL:
        ax4.text(0.02, 0.95, '(e)', transform=ax4.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    else:
        ax4.text(0.02, 0.95, '(d)', transform=ax4.transAxes, fontsize=14, verticalalignment='top', bbox=props)

    # every second tick
    ax4.set_yticks(yticks)
    ax4.set_yticklabels(yticklabels)

    ax4.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax4.set_ylabel("Depth (cm)")
    ax4.axis('tight')

    if contour == True:
        fig.savefig("./plots/EucFACE_SW_obsved_dates_ET_contour_error_rain_%s_%s.png" % (case_name, ring), bbox_inches='tight', pad_inches=0.1)
    else:
        fig.savefig("./plots/EucFACE_SW_obsved_dates_ET_error_rain_%s_%s.png" % (case_name, ring), bbox_inches='tight', pad_inches=0.1)


def plot_profile_ET_error_rain(fpath, case_name, ring, contour, layer):

    """
    plot simulation status and fluxes
    """

    # ========================= ET FLUX  ============================

    # ===== CABLE =====
    cable = nc.Dataset("%s/EucFACE_amb_out.nc" % fpath, 'r')
    Time  = nc.num2date(cable.variables['time'][:],units=cable.variables['time'].units,
            calendar="standard")
    print(Time)
    Rainf = pd.DataFrame(cable.variables['Rainf'][:,0,0],columns=['Rainf'])
    Rainf = Rainf*1800.
    Rainf['dates'] = Time
    Rainf = Rainf.set_index('dates')
    Rainf = Rainf.resample("D").agg('sum')
    Rainf.index = Rainf.index - pd.datetime(2011,12,31)
    Rainf.index = Rainf.index.days

    TVeg = pd.DataFrame(cable.variables['TVeg'][:,0,0],columns=['TVeg'])
    TVeg = TVeg*1800.
    TVeg['dates'] = Time
    TVeg = TVeg.set_index('dates')
    TVeg = TVeg.resample("D").agg('sum')
    TVeg.index = TVeg.index - pd.datetime(2011,12,31)
    TVeg.index = TVeg.index.days

    ESoil = pd.DataFrame(cable.variables['ESoil'][:,0,0],columns=['ESoil'])
    ESoil = ESoil*1800.
    ESoil['dates'] = Time
    ESoil = ESoil.set_index('dates')
    ESoil = ESoil.resample("D").agg('sum')
    ESoil.index = ESoil.index - pd.datetime(2011,12,31)
    ESoil.index = ESoil.index.days

    T  = np.zeros([3,len(TVeg)])
    Es = np.zeros([3,len(ESoil)])

    T[0,:]  =  read_cable_var("%s/EucFACE_R2_out.nc" % fpath, 'TVeg')['cable'].values
    Es[0,:] =  read_cable_var("%s/EucFACE_R2_out.nc" % fpath, 'ESoil')['cable'].values
    T[1,:]  =  read_cable_var("%s/EucFACE_R3_out.nc" % fpath, 'TVeg')['cable'].values
    Es[1,:] =  read_cable_var("%s/EucFACE_R3_out.nc" % fpath, 'ESoil')['cable'].values
    T[2,:]  =  read_cable_var("%s/EucFACE_R6_out.nc" % fpath, 'TVeg')['cable'].values
    Es[2,:] =  read_cable_var("%s/EucFACE_R6_out.nc" % fpath, 'ESoil')['cable'].values

    TVeg['min']  = T.min(axis=0)
    TVeg['max']  = T.max(axis=0)
    ESoil['min'] = Es.min(axis=0)
    ESoil['max'] = Es.max(axis=0)

    # ===== Obs   =====
    subs_Esoil = read_obs_esoil(ring)
    subs_Trans = read_obs_trans(ring)

    Es_R2 = read_obs_esoil('R2')['obs']
    Es_R3 = read_obs_esoil('R3')['obs']
    Es_R6 = read_obs_esoil('R6')['obs']

    T_R2 = read_obs_trans('R2')['obs']
    T_R3 = read_obs_trans('R3')['obs']
    T_R6 = read_obs_trans('R6')['obs']

    T_error = np.zeros([3,len(TVeg)])
    Es_error = np.zeros([3,len(TVeg)])

    for date in TVeg.index:
        if np.any(Es_R2.index == date):
            Es_error[0,date-367] = Es_R2[Es_R2.index == date].values
        else:
            Es_error[0,date-367] = float('NaN')
        if np.any(Es_R3.index == date):
            Es_error[1,date-367] = Es_R3[Es_R3.index == date].values
        else:
            Es_error[1,date-367] = float('NaN')
        if np.any(Es_R6.index == date):
            Es_error[2,date-367] = Es_R6[Es_R6.index == date].values
        else:
            Es_error[2,date-367] = float('NaN')

        if np.any(T_R2.index == date):
            T_error[0,date-367] = T_R2[T_R2.index == date].values
        else:
            T_error[0,date-367] = float('NaN')
        if np.any(T_R3.index == date):
            T_error[1,date-367] = T_R3[T_R3.index == date].values
        else:
            T_error[1,date-367] = float('NaN')
        if np.any(T_R6.index == date):
            T_error[2,date-367] = T_R6[T_R6.index == date].values
        else:
            T_error[2,date-367] = float('NaN')

    ESoil['obs_min'] = Es_error.min(axis=0)
    ESoil['obs_max'] = Es_error.max(axis=0)
    TVeg['obs_min']  = T_error.min(axis=0)
    TVeg['obs_max']  = T_error.max(axis=0)

    # ========================= SM IN TOP 25cm ==========================
    SoilMoist_25cm = pd.DataFrame(cable.variables['SoilMoist'][:,0,0,0], columns=['SoilMoist'])

    if layer == "6":
        SoilMoist_25cm['SoilMoist'] = ( cable.variables['SoilMoist'][:,0,0,0]*0.022 \
                                      + cable.variables['SoilMoist'][:,1,0,0]*0.058 \
                                      + cable.variables['SoilMoist'][:,2,0,0]*0.154 \
                                      + cable.variables['SoilMoist'][:,3,0,0]*(0.25-0.022-0.058-0.154) )/0.25
    elif layer == "31uni":
        SoilMoist_25cm['SoilMoist'] = ( cable.variables['SoilMoist'][:,0,0,0]*0.15 \
                                      + cable.variables['SoilMoist'][:,1,0,0]*0.10 )/0.25

    SoilMoist_25cm['dates'] = Time
    SoilMoist_25cm = SoilMoist_25cm.set_index('dates')
    SoilMoist_25cm = SoilMoist_25cm.resample("D").agg('mean')
    SoilMoist_25cm.index = SoilMoist_25cm.index - pd.datetime(2011,12,31)
    SoilMoist_25cm.index = SoilMoist_25cm.index.days
    SoilMoist_25cm = SoilMoist_25cm.sort_values(by=['dates'])

    # Soil hydraulic param
    swilt = np.zeros(len(TVeg))
    sfc = np.zeros(len(TVeg))
    ssat = np.zeros(len(TVeg))

    if layer == "6":
        swilt[:] = ( cable.variables['swilt'][0]*0.022 + cable.variables['swilt'][1]*0.058 \
                   + cable.variables['swilt'][2]*0.154 + cable.variables['swilt'][3]*(0.25-0.022-0.058-0.154) )/0.25
        sfc[:] = ( cable.variables['sfc'][0]*0.022   + cable.variables['sfc'][1]*0.058 \
                   + cable.variables['sfc'][2]*0.154 + cable.variables['sfc'][3]*(0.25-0.022-0.058-0.154) )/0.25
        ssat[:] = ( cable.variables['ssat'][0]*0.022 + cable.variables['ssat'][1]*0.058 \
                   + cable.variables['ssat'][2]*0.154+ cable.variables['ssat'][3]*(0.25-0.022-0.058-0.154) )/0.25
    elif layer == "31uni":
        swilt[:] =(cable.variables['swilt'][0]*0.15 + cable.variables['swilt'][1]*0.10 )/0.25
        sfc[:] =(cable.variables['sfc'][0]*0.15 + cable.variables['sfc'][1]*0.10 )/0.25
        ssat[:] =(cable.variables['ssat'][0]*0.15 + cable.variables['ssat'][1]*0.10 )/0.25

    # ========================= SM PROFILE ==========================
    ### 1. Read data
    # ==== SM Obs ====
    neo = read_obs_swc_neo(ring)
    tdr = read_obs_swc_tdr(ring)

    # ===== CABLE =====
    SoilMoist = read_cable_SM_one_clmn("%s/EucFACE_amb_out.nc" % fpath, layer)

    ### 2. interpolate SM
    # === Obs SoilMoist ===
    x     = np.concatenate((neo[(25)].index.values,               \
                            neo.index.get_level_values(1).values, \
                            neo[(450)].index.values ))
    y     = np.concatenate(([0]*len(neo[(25)]),                  \
                            neo.index.get_level_values(0).values, \
                            [460]*len(neo[(25)])    ))
    value =  np.concatenate((neo[(25)].values, neo.values, neo[(450)].values))

    X     = neo[(25)].index.values[20:]
    Y     = np.arange(0,461,1)

    grid_X, grid_Y = np.meshgrid(X,Y)

    # === CABLE SoilMoist ===
    if contour:
        grid_data = griddata((x, y) , value, (grid_X, grid_Y), method='cubic')
    else:
        grid_data = griddata((x, y) , value, (grid_X, grid_Y), method='linear')

    ntimes      = len(np.unique(SoilMoist['dates']))
    dates       = np.unique(SoilMoist['dates'].values)


    x_cable     = np.concatenate(( dates, SoilMoist['dates'].values,dates)) # Time
    y_cable     = np.concatenate(([0]*ntimes,SoilMoist['Depth'].values,[460]*ntimes))# Depth
    value_cable = np.concatenate(( SoilMoist.iloc[:ntimes,2].values, \
                                   SoilMoist.iloc[:,2].values,         \
                                   SoilMoist.iloc[-(ntimes):,2].values ))
    value_cable = value_cable*100.

    # add the 12 depths to 0
    grid_X_cable, grid_Y_cable = np.meshgrid(X,Y)

    if contour:
        grid_cable = griddata((x_cable, y_cable) , value_cable, (grid_X_cable, grid_Y_cable),\
                 method='cubic')
    else:
        grid_cable = griddata((x_cable, y_cable) , value_cable, (grid_X_cable, grid_Y_cable),\
                 method='linear')
    difference = grid_cable -grid_data


    # ======================= PLOTTING  ==========================
    fig = plt.figure(figsize=[9,14])

    fig.subplots_adjust(hspace=0.15)
    fig.subplots_adjust(wspace=0.05)

    plt.rcParams['text.usetex']     = False
    plt.rcParams['font.family']     = "sans-serif"
    plt.rcParams['font.serif']      = "Helvetica"
    plt.rcParams['axes.linewidth']  = 1.5
    plt.rcParams['axes.labelsize']  = 14
    plt.rcParams['font.size']       = 14
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['xtick.labelsize'] = 14
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

    props = dict(boxstyle="round", facecolor='white', alpha=0.0, ec='white')

    ax1 = fig.add_subplot(411)
    ax2 = fig.add_subplot(412)
    ax3 = fig.add_subplot(413)
    ax4 = fig.add_subplot(414)

    x = TVeg.index

    # set x-axis values
    cleaner_dates1 = ["2013","2014","2015","2016","2017","2018","2019"]
    xtickslocs1    = [367,732,1097,1462,1828,2193,2558]
    # set color
    cmap = plt.cm.viridis_r

    ax1.fill_between(x, ESoil['min'].rolling(window=3).mean(),
        ESoil['max'].rolling(window=3).mean(), color="orange", alpha=0.2)
    ax1.plot(x, ESoil['ESoil'].rolling(window=3).mean(),
        c="orange", lw=1.0, ls="-", label="$E_{s}$ (CABLE)")
    ax1.fill_between(x, ESoil['obs_min'].rolling(window=3).mean(),
        ESoil['obs_max'].rolling(window=3).mean(), color="red", alpha=0.2)
    ax1.scatter(subs_Esoil.index, subs_Esoil['obs'].rolling(window=3).mean(),
        marker='o', c='',edgecolors='red', s = 2., label="$E_{s}$ (Obs)")
    ax1.text(0.02, 0.95, '(a)', transform=ax1.transAxes, fontsize=14, verticalalignment='top', bbox=props)

    ax2.fill_between(x, TVeg['min'].rolling(window=3).mean(),
        TVeg['max'].rolling(window=3).mean(), color="green", alpha=0.2)
    ax2.plot(x, TVeg['TVeg'].rolling(window=3).mean(),
        c="green", lw=1.0, ls="-", label="$E_{tr}$ (CABLE)")
    ax2.fill_between(x, TVeg['obs_min'].rolling(window=3).mean(),
        TVeg['obs_max'].rolling(window=3).mean(), color="blue", alpha=0.2)
    ax2.scatter(subs_Trans.index, subs_Trans['obs'].rolling(window=3).mean(),
        marker='o', c='',edgecolors='blue', s = 2., label="$E_{tr}$ (Obs)")
    ax2.text(0.02, 0.95, '(b)', transform=ax2.transAxes, fontsize=14, verticalalignment='top', bbox=props)

    # this order of the setting can affect plot x & y axis
    plt.setp(ax1.get_xticklabels(), visible=True)
    ax1.set(xticks=xtickslocs1, xticklabels=cleaner_dates1) ####
    ax1.set_ylabel("$E_{s}$ (mm d$^{-1}$)")
    ax1.axis('tight')
    ax1.set_ylim(0.,4.)
    ax1.set_xlim(367,1097)
    ax1.legend(loc='upper right', ncol=2,labelspacing=0.2, columnspacing=0.2, frameon=False)

    # this order of the setting can affect plot x & y axis
    plt.setp(ax2.get_xticklabels(), visible=True)
    ax2.set(xticks=xtickslocs1, xticklabels=cleaner_dates1) ####
    ax2.set_ylabel("$E_{tr}$ (mm d$^{-1}$)")
    ax2.axis('tight')
    ax2.set_ylim(0.,4.)
    ax2.set_xlim(367,1097)
    ax2.legend(loc='upper right', ncol=2,labelspacing=0.2, columnspacing=0.2, frameon=False)

    cleaner_dates  = ["2013","2014","2015","2016","2017","2018","2019"]
    xtickslocs     = [1,19,37,52,66,74,86]
    yticks         = [360,260,160,60]
    yticklabels    = ["100","200","300","400"]

    if contour:
        levels = np.arange(0.,0.52,0.02)
        img2 = ax3.contourf(grid_cable/100., cmap=cmap, origin="upper", levels=levels,interpolation='nearest')
        Y_labels2 = np.flipud(Y)
    else:
        img2 = ax3.imshow(grid_cable/100., cmap=cmap, vmin=0., vmax=0.52, origin="upper", interpolation='nearest')
        Y_labels2 = Y

    cbar2 = fig.colorbar(img2, ax = ax3,  orientation="vertical", pad=0.02, shrink=.6)
    cbar2.set_label('$θ$ CABLE (m$^{3}$ m$^{-3}$)')
    tick_locator2 = ticker.MaxNLocator(nbins=5)
    cbar2.locator = tick_locator2
    cbar2.update_ticks()

    ax3.text(0.02, 0.95, '(c)', transform=ax3.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    # every second tick
    ax3.set_yticks(yticks)
    ax3.set_yticklabels(yticklabels)
    plt.setp(ax3.get_xticklabels(), visible=False)

    ax3.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax3.set_ylabel("Depth (cm)")
    ax3.axis('tight')

    cmap = plt.cm.BrBG

    if contour:
        levels = np.arange(-0.30,0.30,0.02)
        img3 = ax4.contourf(difference/100., cmap=cmap, origin="upper", levels=levels)
        Y_labels3 = np.flipud(Y)
    else:
        img3 = ax4.imshow(difference/100., cmap=cmap, vmin=-0.30, vmax=0.30, origin="upper", interpolation='nearest')
        Y_labels3 = Y

    cbar3 = fig.colorbar(img3, ax = ax4, orientation="vertical", pad=0.02, shrink=.6)
    cbar3.set_label('$θ$ (CABLE − Obs) (m$^{3}$ m$^{-3}$)')
    tick_locator3 = ticker.MaxNLocator(nbins=6)
    cbar3.locator = tick_locator3
    cbar3.update_ticks()

    ax4.text(0.02, 0.95, '(d)', transform=ax4.transAxes, fontsize=14, verticalalignment='top', bbox=props)

    # every second tick
    ax4.set_yticks(yticks)
    ax4.set_yticklabels(yticklabels)

    ax4.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax4.set_ylabel("Depth (cm)")
    ax4.axis('tight')

    if contour == True:
        fig.savefig("./plots/EucFACE_SW_obsved_dates_ET_contour_error_rain_no-tdr_%s_%s.png" % (case_name, ring), bbox_inches='tight', pad_inches=0.1)
    else:
        fig.savefig("./plots/EucFACE_SW_obsved_dates_ET_error_rain_no-tdr_%s_%s.png" % (case_name, ring), bbox_inches='tight', pad_inches=0.1)
