#!/usr/bin/env python

"""
Pick out and draw heatwave events

Include functions :
    find_Heatwave
    find_Heatwave_hourly
    plot_single_HW_event
    plot_EF_SM_HW

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
import datetime as dt
import netCDF4 as nc
import seaborn as sns
import scipy.stats as stats
from matplotlib import cm
from matplotlib import ticker
from scipy.interpolate import griddata
from sklearn.metrics import mean_squared_error
from plot_eucface_get_var import *

def find_Heatwave(fcable, ring, layer):

    cable = nc.Dataset(fcable, 'r')
    Time  = nc.num2date(cable.variables['time'][:],cable.variables['time'].units)

    # Air temperature
    Tair = pd.DataFrame(cable.variables['Tair'][:,0,0]-273.15,columns=['Tair'])
    Tair['dates'] = Time
    Tair = Tair.set_index('dates')
    Tair = Tair.resample("D").agg('max')

    # Precipitation
    Rainf = pd.DataFrame(cable.variables['Rainf'][:,0,0],columns=['Rainf'])
    Rainf = Rainf*1800.
    Rainf['dates'] = Time
    Rainf = Rainf.set_index('dates')
    Rainf = Rainf.resample("D").agg('sum')

    Qle = read_cable_var(fcable, "Qle")
    Qh  = read_cable_var(fcable, "Qh")
    QleQh= read_cable_var(fcable, "Qle") + read_cable_var(fcable, "Qh")

    EF = pd.DataFrame(Qle['cable'].values/QleQh['cable'].values, columns=['EF'])
    SM = read_SM_top_mid_bot(fcable, ring, layer)

    # exclude rainday and the after two days of rain
    day = np.zeros((len(Tair)), dtype=bool)

    for i in np.arange(0,len(Tair)):
        if (Tair.values[i] >= 35.):
            day[i]   = True

    # calculate heatwave event
    HW = [] # create empty list

    i = 0
    while i < len(Tair)-2:
        HW_event = []
        if (np.all([day[i:i+3]])):
            # consistent 3 days > 35 degree
            for j in np.arange(i-2,i+3):

                event = ( Tair.index[j], Tair['Tair'].values[j], Rainf['Rainf'].values[j],
                          Qle['cable'].values[j], Qh['cable'].values[j],
                          EF['EF'].values[j], SM['SM_top'].values[j], SM['SM_mid'].values[j],
                          SM['SM_bot'].values[j], SM['SM_all'].values[j], SM['SM_15m'].values[j])
                HW_event.append(event)
            i = i + 3

            while day[i]:
                # consistent more days > 35 degree
                event = ( Tair.index[i], Tair['Tair'].values[i], Rainf['Rainf'].values[i],
                          Qle['cable'].values[i], Qh['cable'].values[i],
                          EF['EF'].values[i], SM['SM_top'].values[i], SM['SM_mid'].values[i],
                          SM['SM_bot'].values[i], SM['SM_all'].values[i], SM['SM_15m'].values[j] )
                HW_event.append(event)
                i += 1

            # post 2 days
            event = ( Tair.index[i], Tair['Tair'].values[i], Rainf['Rainf'].values[i],
                      Qle['cable'].values[i], Qh['cable'].values[i],
                      EF['EF'].values[i], SM['SM_top'].values[i], SM['SM_mid'].values[i],
                      SM['SM_bot'].values[i], SM['SM_all'].values[i], SM['SM_15m'].values[j] )
            HW_event.append(event)

            event = ( Tair.index[i+1], Tair['Tair'].values[i+1], Rainf['Rainf'].values[i+1],
                      Qle['cable'].values[i+1], Qh['cable'].values[i+1],
                      EF['EF'].values[i+1], SM['SM_top'].values[i+1], SM['SM_mid'].values[i+1],
                      SM['SM_bot'].values[i+1], SM['SM_all'].values[i+1], SM['SM_15m'].values[j] )
            HW_event.append(event)

            HW.append(HW_event)
        else:
            i += 1

    return HW

def find_Heatwave_hourly(fcable, ring, layer):

    cable = nc.Dataset(fcable, 'r')
    Time  = nc.num2date(cable.variables['time'][:],cable.variables['time'].units)

    # Air temperature
    Tair = pd.DataFrame(cable.variables['Tair'][:,0,0]-273.15,columns=['Tair'])
    Tair['dates'] = Time
    Tair = Tair.set_index('dates')

    Tair_daily = Tair.resample("D").agg('max')

    # Precipitation
    Rainf = pd.DataFrame(cable.variables['Rainf'][:,0,0]*1800.,columns=['Rainf'])
    Rainf['dates'] = Time
    Rainf = Rainf.set_index('dates')

    Qle          = pd.DataFrame(cable.variables['Qle'][:,0,0],columns=['cable'])
    Qle['dates'] = Time
    Qle          = Qle.set_index('dates')

    Qh           = pd.DataFrame(cable.variables['Qh'][:,0,0],columns=['cable'])
    Qh['dates']  = Time
    Qh           = Qh.set_index('dates')

    QleQh         = Qle + Qh

    EF          = pd.DataFrame(Qle['cable'].values/QleQh['cable'].values, columns=['EF'])
    EF['dates'] = Time
    EF          = EF.set_index('dates')
    EF['EF']    = np.where(np.all([EF.index.hour >= 9., EF.index.hour <= 16.,
                  EF['EF'].values <= 5. ],axis=0 ), EF['EF'].values, float('nan'))
    SM = read_SM_top_mid_bot_hourly(fcable, ring, layer)

    # exclude rainday and the after two days of rain
    day = np.zeros((len(Tair_daily)), dtype=bool)

    for i in np.arange(0,len(Tair_daily)):
        if (Tair_daily.values[i] >= 35.):
            day[i]   = True

    # calculate heatwave event
    HW = [] # create empty list

    i = 0

    while i < len(Tair_daily)-1:
        HW_event = []

        if (np.all([day[i:i+3]])):

            day_start = Tair_daily.index[i-1]
            i = i + 3

            while day[i]:

                i += 1

            else:

                if i+1 < len(Tair_daily.index):
                    day_end = Tair_daily.index[i+1]
                else:
                    day_end = Tair_daily.index[i]

                Tair_event  = Tair[np.all([Tair.index >= day_start,  Tair.index < day_end],axis=0)]
                Rainf_event = Rainf[np.all([Tair.index >= day_start, Tair.index < day_end],axis=0)]
                Qle_event   = Qle[np.all([Tair.index >= day_start,   Tair.index < day_end],axis=0)]
                Qh_event    = Qh[np.all([Tair.index >= day_start,    Tair.index < day_end],axis=0)]
                EF_event    = EF[np.all([Tair.index >= day_start,    Tair.index < day_end],axis=0)]
                SM_event    = SM[np.all([Tair.index >= day_start,    Tair.index < day_end],axis=0)]

                for hour_num in np.arange(len(Tair_event)):
                    hour_in_event = ( Tair_event.index[hour_num],
                                      Tair_event['Tair'].values[hour_num],
                                      Rainf_event['Rainf'].values[hour_num],
                                      Qle_event['cable'].values[hour_num],
                                      Qh_event['cable'].values[hour_num],
                                      EF_event['EF'].values[hour_num],
                                      SM_event['SM_top'].values[hour_num],
                                      SM_event['SM_mid'].values[hour_num],
                                      SM_event['SM_bot'].values[hour_num],
                                      SM_event['SM_all'].values[hour_num],
                                      SM_event['SM_15m'].values[hour_num] )
                    HW_event.append(hour_in_event)

            HW.append(HW_event)
        else:
            i += 1

    return HW


def plot_single_HW_event(time_scale, case_labels, i, date, Tair, Rainf, Qle, Qh, EF, SM_top, SM_mid, SM_bot, SM_all, SM_15m):

    # ======================= Plot setting ============================
    if time_scale == "daily":
        fig = plt.figure(figsize=[11,17.5])
    elif time_scale == "hourly":
        fig = plt.figure(figsize=[13,17.5])

    fig.subplots_adjust(hspace=0.0)
    fig.subplots_adjust(wspace=0.1)

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

    # set the box type of sequence number
    props = dict(boxstyle="round", facecolor='white', alpha=0.0, ec='white')

    # choose colormap
    colors = cm.tab20(np.arange(0,len(case_labels)))
    ls     = ['-','--','-','--','-','--','-','--','-','--']

    # only plot event number 6
    if i == 6:

        ax1  = fig.add_subplot(411)
        ax2  = fig.add_subplot(412,sharex=ax1)
        ax3  = fig.add_subplot(413,sharex=ax2)
        ax4  = fig.add_subplot(414,sharex=ax3)

        x    = date

        if time_scale == "daily":
            width  = 0.6
        elif time_scale == "hourly":
            width  = 1/48

        ax1.plot(x, Tair,   c="black", lw=1.5, ls="-", label="Air Temperature")
        ax5  = ax1.twinx()

        for case_num in np.arange(len(case_labels)):
            print(case_num)
            ax2.plot(x, EF[case_num, :],  c=colors[case_num], lw=1.5, ls=ls[case_num], label=case_labels[case_num])
            ax3.plot(x, Qle[case_num, :], c=colors[case_num], lw=1.5, ls=ls[case_num], label=case_labels[case_num])
            ax4.plot(x, Qh[case_num, :],  c=colors[case_num], lw=1.5, ls=ls[case_num], label=case_labels[case_num])
            ax5.plot(x, SM_15m[case_num, :], c=colors[case_num], lw=1.5, ls=ls[case_num], label=case_labels[case_num])
            print("***********************")
            print("preceding soil moisture of %s is %f" % (case_labels[case_num], SM_15m[case_num, 0]))
            print("mean sensible heat flux of %s is %f" % (case_labels[case_num], Qh[case_num, :].mean()))
            print("***********************")
        for i in np.arange(48,97):
            print("%f hour: Tair, Qh and Qle are %f %f %f" %((i-47)/2, Tair[i], Qh[6,i], Qle[6,i]))

        if time_scale == "daily":
            ax1.set_ylabel('Max Air Temperature (°C)')
            ax1.set_ylim(20, 45)
        elif time_scale == "hourly":
            ax1.set_ylabel('$Tair$ (°C)')
            ax1.set_ylim(-9.9, 45)

        plt.setp(ax1.get_xticklabels(), visible=False)
        ax1.set_xlim(date[0],date[-47])
        ax1.axhline(y=35.,c=almost_black, ls="--")
        ax1.get_xaxis().set_visible(False)
        ax1.text(0.02, 0.95, '(a)', transform=ax1.transAxes, fontsize=14, verticalalignment='top', bbox=props)

        ax5.set_ylabel("$θ$$_{1.5m}$ (m$^{3}$ m$^{-3}$)")
        ax5.axis('tight')
        ax5.get_xaxis().set_visible(False)
        ax5.set_xlim(date[0],date[-47])
        if time_scale == "daily":
            ax5.set_ylim(0.18,0.32)
            plt.suptitle('Heatwave in %s ~ %s ' % (str(date[2]), str(date[-3])))
        elif time_scale == "hourly":
            ax5.set_ylim(0.12,0.41)

        plt.setp(ax2.get_xticklabels(), visible=False)
        ax2.set_ylabel("$EF$")
        ax2.axis('tight')
        ax2.set_xlim(date[0],date[-47])
        if time_scale == "daily":
            ax2.set_ylim(0.,1.8)
        elif time_scale == "hourly":
            ax2.set_ylim(0,1.1)

        ax2.text(0.02, 0.95, '(b)', transform=ax2.transAxes, fontsize=14, verticalalignment='top', bbox=props)

        plt.setp(ax3.get_xticklabels(), visible=False)
        ax3.set_ylabel('$Q_{E}$ (W m$^{-2}$)')
        ax3.axis('tight')
        ax3.set_xlim(date[0],date[-47])
        if time_scale == "daily":
            ax3.set_ylim(-50.,220)
        elif time_scale == "hourly":
            ax3.set_ylim(-40.,450.)

        ax3.text(0.02, 0.95, '(c)', transform=ax3.transAxes, fontsize=14, verticalalignment='top', bbox=props)
        ax3.legend( loc='best', ncol=2, frameon=False)

        plt.setp(ax4.get_xticklabels(), visible=True)
        ax4.set_ylabel('$Q_{H}$ (W m$^{-2}$)')
        ax4.axis('tight')
        ax4.set_xlim(date[0],date[-47])

        if time_scale == "daily":
            ax4.set_ylim(-50.,220)
        elif time_scale == "hourly":
            ax4.set_ylim(-40.,450.)

        ax4.text(0.02, 0.95, '(d)', transform=ax4.transAxes, fontsize=14, verticalalignment='top', bbox=props)

        fig.savefig("./plots/EucFACE_Heatwave_%s" % str(i) , bbox_inches='tight', pad_inches=0.02)

def plot_EF_SM_HW(fcables, case_labels, layers, ring, time_scale):

    # =========== Calc HW events ==========
    # save all cases and all heatwave events
    # struction : 1st-D  2st-D  3st-D  4st-D
    #             case   event  day    variables

    HW_all   = []
    case_sum = len(fcables)

    for case_num in np.arange(case_sum):
        if time_scale == "daily":
            HW = find_Heatwave(fcables[case_num], ring, layers[case_num])
        elif time_scale == "hourly":
            HW = find_Heatwave_hourly(fcables[case_num], ring, layers[case_num])
        HW_all.append(HW)

    # ============ Read vars ==============
    event_sum = len(HW_all[0])

    for event_num in np.arange(event_sum):

        day_sum = len(HW_all[0][event_num])
        if time_scale == "daily":
            date   = np.zeros(day_sum, dtype='datetime64[D]')
        elif time_scale == "hourly":
            date   = np.zeros(day_sum, dtype='datetime64[ns]')
        Tair   = np.zeros(day_sum)
        Rainf  = np.zeros(day_sum)
        Qle    = np.zeros([case_sum,day_sum])
        Qh     = np.zeros([case_sum,day_sum])
        EF     = np.zeros([case_sum,day_sum])
        SM_top = np.zeros([case_sum,day_sum])
        SM_mid = np.zeros([case_sum,day_sum])
        SM_bot = np.zeros([case_sum,day_sum])
        SM_all = np.zeros([case_sum,day_sum])
        SM_15m = np.zeros([case_sum,day_sum])

        # loop days in one event
        for day_num in np.arange(day_sum):
            date[day_num]      = HW_all[0][event_num][day_num][0].to_datetime64()
            Tair[day_num]      = HW_all[0][event_num][day_num][1]
            Rainf[day_num]     = HW_all[0][event_num][day_num][2]

            for case_num in np.arange(case_sum):

                Qle[case_num,day_num]     =  HW_all[case_num][event_num][day_num][3]
                Qh[case_num,day_num]      =  HW_all[case_num][event_num][day_num][4]
                EF[case_num,day_num]      =  HW_all[case_num][event_num][day_num][5]
                SM_top[case_num,day_num]  =  HW_all[case_num][event_num][day_num][6]
                SM_mid[case_num,day_num]  =  HW_all[case_num][event_num][day_num][7]
                SM_bot[case_num,day_num]  =  HW_all[case_num][event_num][day_num][8]
                SM_all[case_num,day_num]  =  HW_all[case_num][event_num][day_num][9]
                SM_15m[case_num,day_num]  =  HW_all[case_num][event_num][day_num][10]

        plot_single_HW_event(time_scale, case_labels, event_num, date, Tair, Rainf, Qle, Qh, EF, SM_top, SM_mid, SM_bot, SM_all, SM_15m)
