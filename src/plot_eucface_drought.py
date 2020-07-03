#!/usr/bin/env python

"""
Draw drought plot and beta plot

Include functions :

    plot_fwsoil_boxplot_SM
    plot_Rain_Fwsoil_Trans_Esoil_SH_SM

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
import scipy.stats as stats
import seaborn as sns
from matplotlib import cm
from matplotlib import ticker
from scipy.interpolate import griddata
from sklearn.metrics import mean_squared_error
from plot_eucface_get_var import *

def plot_Rain_Fwsoil_Trans_Esoil_SH_SM( fcables, case_labels, layers, ring):

    # ======================= Plot setting ============================
    fig = plt.figure(figsize=[13,17.5])
    fig.subplots_adjust(hspace=0.1)
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

    ax1  = fig.add_subplot(511)
    ax2  = fig.add_subplot(512)
    ax3  = fig.add_subplot(513)
    ax4  = fig.add_subplot(514)
    ax5  = fig.add_subplot(515)

    cleaner_dates = ["Oct 2017","Jan 2018","Apr 2018", "Jul 2018", "Oct 2018"]
    xtickslocs    = [      2101,      2193,      2283,       2374,       2466]
    day_start = 2101 # 2017-10-1
    day_end   = 2467 # 2018-10-1

    day_start_smooth = day_start - 30
    case_sum = len(fcables)

    # read obs soil moisture at top 1.5 m
    subs_neo   = read_obs_neo_top_mid_bot(ring)

    for case_num in np.arange(case_sum):

        Rain  = read_cable_var(fcables[case_num], "Rainf")
        fw    = read_cable_var(fcables[case_num], "Fwsoil")
        Trans = read_cable_var(fcables[case_num], "TVeg")
        Esoil = read_cable_var(fcables[case_num], "ESoil")
        Qle   = read_cable_var(fcables[case_num], "Qle")
        Qh    = read_cable_var(fcables[case_num], "Qh")

        sm = read_SM_top_mid_bot(fcables[case_num], ring, layers[case_num])

        x        = fw.index[fw.index >= day_start]
        x_smooth = fw.index[fw.index >= day_start_smooth]

        if case_num == 0:
            print(subs_neo[subs_neo.index >= day_start].index)
            print(subs_neo["SM_15m"][subs_neo.index >= day_start])
            ax1.scatter(subs_neo[subs_neo.index >= day_start].index, subs_neo["SM_15m"][subs_neo.index >= day_start],
                        marker='o', c='',edgecolors='blue', s = 6., label="Obs")

        ax1.plot(x_smooth, sm['SM_15m'][Qle.index >= day_start_smooth].rolling(window=30).mean(),
                c=colors[case_num], lw=1.5, ls="-", label=case_labels[case_num], alpha=1.)
        ax2.plot(x_smooth, Trans['cable'][Trans.index >= day_start_smooth].rolling(window=30).sum(),
                c=colors[case_num], lw=1.5, ls="-", label=case_labels[case_num], alpha=1.)
        ax3.plot(x_smooth, fw['cable'][fw.index >= day_start_smooth].rolling(window=30).mean(),
                c=colors[case_num], lw=1.5, ls="-", label=case_labels[case_num])
        ax4.plot(x_smooth, Esoil['cable'][Esoil.index >= day_start_smooth].rolling(window=30).sum(),
                c=colors[case_num], lw=1.5, ls="-", label=case_labels[case_num], alpha=1.)
        ax5.plot(x_smooth, Qh['cable'][Qle.index >= day_start_smooth].rolling(window=30).mean(),
                c=colors[case_num], lw=1.5, ls="-", label=case_labels[case_num], alpha=1.)

    ax6  = ax1.twinx()
    ax6.set_ylabel('$P$ (mm d$^{-1}$)')
    ax6.bar(x_smooth, -Rain['cable'][Rain.index >= day_start_smooth].values,  1.,
            color='gray', alpha = 0.5, label='Rainfall')
    ax6.set_ylim(-60.,0)
    y_ticks      = [-60,-50,-40,-30,-20,-10,0.]
    y_ticklabels = ['60','50','40','30','20','10','0']
    ax6.set_yticks(y_ticks)
    ax6.set_yticklabels(y_ticklabels)
    ax6.get_xaxis().set_visible(False)

    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax1.axis('tight')
    ax1.set_ylim(0.05,0.35)
    ax1.set_xlim(day_start,day_end)
    ax1.set_ylabel('$θ$$_{1.5m}$ (m$^{3}$ m$^{-3}$)')
    ax1.text(0.02, 0.95, '(a)', transform=ax1.transAxes, fontsize=14, verticalalignment='top', bbox=props)

    plt.setp(ax2.get_xticklabels(), visible=False)
    ax2.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax2.set_ylabel("$E_{tr}$ (mm mon$^{-1}$)")
    ax2.axis('tight')
    ax2.legend(numpoints=1, ncol=3, loc='best', frameon=False)
    ax2.set_ylim(0.,70.)
    ax2.set_xlim(day_start,day_end)
    ax2.text(0.02, 0.95, '(b)', transform=ax2.transAxes, fontsize=14, verticalalignment='top', bbox=props)

    plt.setp(ax3.get_xticklabels(), visible=False)
    ax3.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax3.set_ylabel("$β$")
    ax3.axis('tight')
    ax3.set_ylim(0.,1.18)
    ax3.set_xlim(day_start,day_end)
    ax3.text(0.02, 0.95, '(c)', transform=ax3.transAxes, fontsize=14, verticalalignment='top', bbox=props)

    plt.setp(ax4.get_xticklabels(), visible=False)
    ax4.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax4.set_ylabel("$E_{s}$ (mm mon$^{-1}$)")
    ax4.axis('tight')
    ax4.set_ylim(0.,50.)
    ax4.set_xlim(day_start,day_end)
    ax4.text(0.02, 0.95, '(d)', transform=ax4.transAxes, fontsize=14, verticalalignment='top', bbox=props)

    plt.setp(ax5.get_xticklabels(), visible=True)
    ax5.set(xticks=xtickslocs, xticklabels=cleaner_dates)

    ax5.set_ylabel("$Q_{H}$ (W m$^{-2}$)")
    ax5.axis('tight')
    ax5.set_ylim(-20.,100.)
    ax5.set_xlim(day_start,day_end)
    ax5.text(0.02, 0.95, '(e)', transform=ax5.transAxes, fontsize=14, verticalalignment='top', bbox=props)

    fig.savefig("./plots/EucFACE_Rain_Fwsoil_Trans_SH_SM" , bbox_inches='tight', pad_inches=0.1)

def plot_fwsoil_boxplot_SM( fcables, case_labels, layers, ring):

    """
    (a) box-whisker of fwsoil
    (b) fwsoil vs SM
    """

    # ======================= Plot setting ============================
    fig = plt.figure(figsize=[10,11])
    fig.subplots_adjust(hspace=0.20)
    fig.subplots_adjust(wspace=0.12)

    plt.rcParams['text.usetex']     = False
    plt.rcParams['font.family']     = "sans-serif"
    plt.rcParams['font.serif']      = "Helvetica"
    plt.rcParams['axes.linewidth']  = 1.5
    plt.rcParams['axes.labelsize']  = 14
    plt.rcParams['font.size']       = 14
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams["legend.markerscale"] = 3.0

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

    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    # ========================= box-whisker of fwsoil============================
    day_start_drought = 2101 # 2017-10-1
    day_end_drought   = 2466 # 2018-10-1

    day_start_all     = 367  # first day of 2013
    day_end           = 2923 # first day of 2020

    day_drought  = day_end_drought - day_start_drought + 1
    day_all      = day_end - day_start_all + 1
    case_sum     = len(fcables)
    fw           = pd.DataFrame(np.zeros((day_drought+day_all)*case_sum),columns=['fwsoil'])
    fw['year']   = [''] * ((day_drought+day_all)*case_sum)
    fw['exp']    = [''] * ((day_drought+day_all)*case_sum)

    s = 0

    for case_num in np.arange(case_sum):

        cable = nc.Dataset(fcables[case_num], 'r')
        Time  = nc.num2date(cable.variables['time'][:],cable.variables['time'].units)

        Fwsoil          = pd.DataFrame(cable.variables['Fwsoil'][:,0,0],columns=['fwsoil'])
        Fwsoil['dates'] = Time
        Fwsoil          = Fwsoil.set_index('dates')
        Fwsoil          = Fwsoil.resample("D").agg('mean')
        Fwsoil.index    = Fwsoil.index - pd.datetime(2011,12,31)
        Fwsoil.index    = Fwsoil.index.days

        e  = s+day_drought

        fw['fwsoil'].iloc[s:e] = Fwsoil[np.all([Fwsoil.index >= day_start_drought,
                                 Fwsoil.index <=day_end_drought],axis=0)]['fwsoil'].values
        fw['year'].iloc[s:e]   = ['drought'] * day_drought
        fw['exp'].iloc[s:e]    = [ case_labels[case_num]] * day_drought
        s  = e
        e  = s+day_all
        fw['fwsoil'].iloc[s:e] = Fwsoil[np.all([Fwsoil.index >= day_start_all,
                                 Fwsoil.index <=day_end],axis=0)]['fwsoil'].values
        fw['year'].iloc[s:e]   = ['all'] * day_all
        fw['exp'].iloc[s:e]    = [ case_labels[case_num]] * day_all
        s  =  e

    sns.boxplot(x="exp", y="fwsoil", hue="year", data=fw, palette="BrBG",
                order=case_labels,  width=0.7, hue_order=['drought','all'],
                ax=ax1, showfliers=False, color=almost_black)

    ax1.set_ylabel("$β$")
    ax1.set_xlabel("")
    ax1.axis('tight')
    ax1.set_ylim(0.,1.1)
    ax1.axhline(y=np.mean(fw[np.all([fw.year=='drought',fw.exp=='Ctl'],axis=0)]['fwsoil'].values),
                c=almost_black, ls="--")
    ax1.legend(loc='best', frameon=False)
    ax1.text(0.02, 0.95, '(a)', transform=ax1.transAxes, fontsize=14, verticalalignment='top', bbox=props)

    print("***********************")
    # case_labels = ["Ctl", "Sres", "Watr", "Hi-Res-1", "Hi-Res-2", "Opt",  "β-hvrd",  "β-exp" ]
    print("median of Ctl is %f" % np.median(fw[np.all([fw.year=='all',fw.exp=='Ctl'],axis=0)]['fwsoil'].values))
    print("median of Sres is %f" % np.median(fw[np.all([fw.year=='all',fw.exp=='Sres'],axis=0)]['fwsoil'].values))
    print("median of Watr is %f" % np.median(fw[np.all([fw.year=='all',fw.exp=='Watr'],axis=0)]['fwsoil'].values))
    print("median of Hi-Res-1 is %f" % np.median(fw[np.all([fw.year=='all',fw.exp=='Hi-Res-1'],axis=0)]['fwsoil'].values))
    print("median of Hi-Res-2 is %f" % np.median(fw[np.all([fw.year=='all',fw.exp=='Hi-Res-2'],axis=0)]['fwsoil'].values))
    print("median of Opt is %f" % np.median(fw[np.all([fw.year=='all',fw.exp=='Opt'],axis=0)]['fwsoil'].values))
    print("median of β-hvrd is %f" % np.median(fw[np.all([fw.year=='all',fw.exp=='β-hvrd'],axis=0)]['fwsoil'].values))
    print("median of β-exp is %f" % np.median(fw[np.all([fw.year=='all',fw.exp=='β-exp'],axis=0)]['fwsoil'].values))
    print("***********************")

    colors = cm.tab20(np.arange(0,len(case_labels)))
    # ============================= boxplot ===================================
    for case_num in np.arange(len(fcables)):
        SM  = read_cable_SM(fcables[case_num], layers[case_num])
        fw  = read_cable_var(fcables[case_num], "Fwsoil")

        # theta_1.5m : using root zone soil moisture
        if layers[case_num] == "6":
            sm =(  SM.iloc[:,0]*0.022 + SM.iloc[:,1]*0.058 \
                 + SM.iloc[:,2]*0.154 + SM.iloc[:,3]*0.409 \
                 + SM.iloc[:,4]*(1.5-0.022-0.058-0.154-0.409) )/1.5
        elif layers[case_num] == "31uni":
            sm = SM.iloc[:,0:10].mean(axis = 1)

        ax2.scatter(sm, fw,  s=1., marker='o', alpha=0.45, c=colors[case_num],label=case_labels[case_num])

    ax2.set_xlim(0.08,0.405)
    ax2.set_ylim(0.0,1.05)
    ax2.set_ylabel("$β$")
    ax2.set_xlabel("$θ$$_{1.5m}$ (m$^{3}$ m$^{-3}$)")
    ax2.legend(numpoints=1, loc='best', frameon=False)
    ax2.text(0.02, 0.95, '(b)', transform=ax2.transAxes, fontsize=14, verticalalignment='top', bbox=props)

    fig.savefig("./plots/EucFACE_Fwsoil_boxplot_SM" , bbox_inches='tight', pad_inches=0.1)
