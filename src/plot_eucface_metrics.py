#!/usr/bin/env python

"""
Calculate metrics and annual values

Include functions :
    calc_metrics
    annual_value

"""

__author__ = "MU Mengyuan"
__email__  = "mu.mengyuan815@gmail.com"

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib import ticker
import datetime as dt
import netCDF4 as nc
from scipy.interpolate import griddata
from scipy.interpolate import interp1d
import scipy.stats as stats
from sklearn.metrics import mean_squared_error
from plot_eucface_get_var import *

def calc_metrics(fcables, case_labels, layers, vars, ring):

    '''
    Computes metrics of r, RMSE, MBE, P5, P95
    '''

    metris = pd.DataFrame(columns=['CASE', 'VAR','r','RMSE', 'MBE', 'P5', 'P95'])
    case_sum = len(case_labels)
    j = 0

    for var in vars:
        for i in np.arange(case_sum):
            # ================= read data and unify dates =====================
            subs_cable = read_ET_SM_top_mid_bot(fcables[i], ring, layers[i])

            if var == 'Esoil':
                subs_Esoil = read_obs_esoil(ring)
                # observation must within simulated dates
                Esoil_obs   = subs_Esoil['obs'].loc[subs_Esoil.index.isin(subs_cable.index)]
                # simulation must at observed dates
                Esoil_cable = subs_cable["ESoil"].loc[subs_cable.index.isin(subs_Esoil.index)]
                # excluding nan dates
                cable    = Esoil_cable[np.isnan(Esoil_obs) == False]
                obs      = Esoil_obs[np.isnan(Esoil_obs) == False]

            elif var == 'Trans':
                subs_Trans = read_obs_trans(ring)
                Trans_obs   = subs_Trans['obs'].loc[subs_Trans.index.isin(subs_cable.index)]
                Trans_cable = subs_cable["TVeg"].loc[subs_cable.index.isin(subs_Trans.index)]
                cable    = Trans_cable[np.isnan(Trans_obs) == False]
                obs      = Trans_obs[np.isnan(Trans_obs) == False]
            elif var == 'VWC':
                subs_neo   = read_obs_neo_top_mid_bot(ring)
                SM_all_obs  = subs_neo["SM_all"].loc[subs_neo.index.isin(subs_cable.index)]
                SM_all_cable= subs_cable["SM_all"].loc[subs_cable.index.isin(subs_neo.index)]
                cable   = SM_all_cable[np.isnan(SM_all_obs) == False]
                obs     = SM_all_obs[np.isnan(SM_all_obs) == False]
            elif var == 'SM_25cm':
                subs_tdr   = read_obs_swc_tdr(ring)
                SM_25cm_obs  = subs_tdr["obs"].loc[subs_tdr.index.isin(subs_cable.index)]
                SM_25cm_cable= subs_cable["SM_25cm"].loc[subs_cable.index.isin(subs_tdr.index)]
                cable   = SM_25cm_cable[np.isnan(SM_25cm_obs) == False]
                obs     = SM_25cm_obs[np.isnan(SM_25cm_obs) == False]
            elif var == 'SM_15m':
                subs_neo   = read_obs_neo_top_mid_bot(ring)
                SM_15m_obs  = subs_neo["SM_15m"].loc[subs_neo.index.isin(subs_cable.index)]
                SM_15m_cable= subs_cable["SM_15m"].loc[subs_cable.index.isin(subs_neo.index)]
                cable   = SM_15m_cable[np.isnan(SM_15m_obs) == False]
                obs     = SM_15m_obs[np.isnan(SM_15m_obs) == False]
            elif var == 'SM_bot':
                subs_neo   = read_obs_neo_top_mid_bot(ring)
                SM_bot_obs  = subs_neo["SM_bot"].loc[subs_neo.index.isin(subs_cable.index)]
                SM_bot_cable= subs_cable["SM_bot"].loc[subs_cable.index.isin(subs_neo.index)]
                cable   = SM_bot_cable[np.isnan(SM_bot_obs) == False]
                obs     = SM_bot_obs[np.isnan(SM_bot_obs) == False]

            # ============ metrics ============
            r    = stats.pearsonr(obs, cable)[0]
            RMSE = np.sqrt(mean_squared_error(obs, cable))
            MBE  = np.mean(cable - obs)
            p5   = np.percentile(cable, 5) - np.percentile(obs, 5)
            p95  = np.percentile(cable, 95) - np.percentile(obs, 95)

            metris.loc[j] = [ case_labels[i], var, r, RMSE, MBE, p5, p95 ]
            j += 1

    #np.savetxt("./csv/Esoil_at_observed_dates.csv" , data, delimiter=",")

    print(metris)

def annual_values(fcables, case_labels, layers, ring):

    """
    calculate annual water budget items, energy flux and soil status
    """

    # units transform
    step_2_sec = 30.*60.
    umol_2_gC  = 12.0107 * 1.0E-6

    case_sum = len(case_labels)

    annual = pd.DataFrame(columns=['CASE', 'P', 'ET', 'Etr', 'Es', 'Ec','R','D','GPP','Qe','Qh','VWC'])

    j = 0

    for case_num in np.arange(case_sum):
        print(layers)
        if layers[case_num] == "6":
            zse = [ 0.022, 0.058, 0.154, 0.409, 1.085, 2.872 ]
        elif layers[case_num] == "31uni":
            zse = [ 0.15,  0.15,  0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15,  \
                    0.15,  0.15,  0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15,  \
                    0.15,  0.15,  0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15,  \
                    0.15 ]


        cable = nc.Dataset(fcables[case_num], 'r')

        df        = pd.DataFrame(cable.variables['Rainf'][:,0,0]*step_2_sec, columns=['P'])
                      # 'Rainfall+snowfall'
        df['ET']  = cable.variables['Evap'][:,0,0]*step_2_sec   # 'Total evaporation'
        df['Etr'] = cable.variables['TVeg'][:,0,0]*step_2_sec   # 'Vegetation transpiration'
        df['Es']  = cable.variables['ESoil'][:,0,0]*step_2_sec  # 'evaporation from soil'
        df['Ec']  = cable.variables['ECanop'][:,0,0]*step_2_sec # 'Wet canopy evaporation'
        df['R']   = cable.variables['Qs'][:,0,0]*step_2_sec  + cable.variables['Qsb'][:,0,0]*step_2_sec
                    # 'Surface runoff'+ 'Subsurface runoff'
        df['D']   = cable.variables['Qrecharge'][:,0,0]*step_2_sec
        df['GPP'] = cable.variables['GPP'][:,0,0]*step_2_sec*umol_2_gC

        status    = pd.DataFrame(cable.variables['Qle'][:,0,0] , columns=['Qe'])
                    # 'Surface latent heat flux'
        status['Qh'] = cable.variables['Qh'][:,0,0]    # 'Surface sensible heat flux'

        if layers[case_num] == "6":

            status['VWC'] = cable.variables['SoilMoist'][:,0,0,0]*zse[0]
            for i in np.arange(1,6):
                status['VWC'] = status['VWC'] + cable.variables['SoilMoist'][:,i,0,0]*zse[i]
            status['VWC'] = status['VWC']/sum(zse)

        elif layers[case_num] == "31uni":

            status['VWC']     = cable.variables['SoilMoist'][:,30,0,0]*0.1
            for i in np.arange(0,30):
                status['VWC'] = status['VWC'] + cable.variables['SoilMoist'][:,i,0,0]*zse[i]
            status['VWC']     = status['VWC']/4.6


        df['dates']     = nc.num2date(cable.variables['time'][:], cable.variables['time'].units)
        df              = df.set_index('dates')
        df              = df.resample("Y").agg('sum')

        status['dates']   = nc.num2date(cable.variables['time'][:], cable.variables['time'].units)
        status            = status.set_index('dates')
        status            = status.resample("Y").agg('mean')

        # multi-year average
        df     = df.iloc[:,:].mean(axis=0)
        status = status.iloc[:,:].mean(axis=0)
        print(df)
        print(status)
        print([ df.values[0:8], status.values[0:3] ])
        annual.loc[j] = [ case_labels[case_num], df['P'],
                          df['ET'], df['Etr'],
                          df['Es'], df['Ec'],
                          df['R'], df['D'],
                          df['GPP'], status['Qe'],
                          status['Qh'], status['VWC'] ]
        j += 1

        df = None
        status = None
        cable = None

    print(annual)
