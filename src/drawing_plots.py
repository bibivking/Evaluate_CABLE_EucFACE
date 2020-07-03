#!/usr/bin/env python
'''
Plot Figure 2-10, S1-S3, S5 and make Table 2,3 S2
'''

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
import scipy.stats as stats
from sklearn.metrics import mean_squared_error
from plot_eucface_metrics import *
from plot_eucface_swc_profile import *
from plot_eucface_drought import *
from plot_eucface_heatwave import *

if __name__ == "__main__":

    ring  = 'amb'

    contour = True

    # "Out of box CABLE"
    case_name1 = "met_LAI-08_6convex07"
    # "Ctl"
    case_name2 = "met_LAI-08_6_teuc"
    # "Sres"
    case_name3 = "met_LAI-08_6_teuc_sres"
    # "Watr"
    case_name4 = "met_LAI-08_6_teuc_sres_watr"
    # "Hi-Res-1"
    case_name5 = "met_LAI-08_31uni_teuc_sres_watr"
    # "Hi-Res-2"
    case_name6 = "met_LAI-08_vrt_31uni_teuc_sres_watr"
    # "Opt"
    case_name7 = "met_LAI-08_vrt_swilt-watr-ssat_hyds10_31uni_teuc_sres_watr"
    # "β-hvrd"
    case_name8 = "met_LAI-08_vrt_swilt-watr-ssat_hyds10_31uni_teuc_sres_watr_beta-hvrd"
    # "β-exp"
    case_name9 = "met_LAI-08_vrt_swilt-watr-ssat_hyds10_31uni_teuc_sres_watr_beta-exp"
    # "Opt-sub"
    case_name10 = "met_LAI-08_vrt_swilt-watr-ssat_31uni_teuc_sres_watr"

    pyth = "/srv/ccrc/data25/z5218916/cable/EucFACE"

    case_1 = "%s/EucFACE_run/outputs/%s" % (pyth, case_name1)
    fcbl_1 ="%s/EucFACE_%s_out.nc" % (case_1, ring)

    case_2 = "%s/EucFACE_run/outputs/%s" % (pyth, case_name2)
    fcbl_2 ="%s/EucFACE_%s_out.nc" % (case_2, ring)

    case_3 = "%s/EucFACE_run/outputs/%s" % (pyth, case_name3)
    fcbl_3 ="%s/EucFACE_%s_out.nc" % (case_3, ring)

    case_4 = "%s/EucFACE_run/outputs/%s" % (pyth, case_name4)
    fcbl_4 ="%s/EucFACE_%s_out.nc" % (case_4, ring)

    case_5 = "%s/EucFACE_run/outputs/%s" % (pyth, case_name5)
    fcbl_5 ="%s/EucFACE_%s_out.nc" % (case_5, ring)

    case_6 = "%s/EucFACE_run/outputs/%s" % (pyth, case_name6)
    fcbl_6 ="%s/EucFACE_%s_out.nc" % (case_6, ring)

    case_7 = "%s/EucFACE_run/outputs/%s" % (pyth, case_name7)
    fcbl_7 ="%s/EucFACE_%s_out.nc" % (case_7, ring)

    case_8 = "%s/EucFACE_run/outputs/%s" % (pyth, case_name8)
    fcbl_8 = "%s/EucFACE_%s_out.nc" % (case_8, ring)

    case_9 = "%s/EucFACE_run/outputs/%s" % (pyth, case_name9)
    fcbl_9 = "%s/EucFACE_%s_out.nc" % (case_9, ring)

    case_10 = "%s/EucFACE_run/outputs/%s" % (pyth, case_name10)
    fcbl_10 = "%s/EucFACE_%s_out.nc" % (case_10, ring)

    # simulations needed
    fcables     = [fcbl_2,   fcbl_3,    fcbl_4,     fcbl_5,      fcbl_6,
                   fcbl_7,   fcbl_8,   fcbl_9 ]
    case_labels = ["Ctl",    "Sres",    "Watr",  "Hi-Res-1", "Hi-Res-2",
                   "Opt",  "β-hvrd",  "β-exp" ]
    layers      = [   "6",      "6",       "6",     "31uni",    "31uni",
                   "31uni", "31uni",  "31uni" ]

    time_scale  = "hourly"
    vars        = ['Esoil', 'Trans', 'VWC', 'SM_25cm', 'SM_15m', 'SM_bot']
    CTL         = case_name2

    '''
    Profile
    '''

    fpath1 = "%s/EucFACE_run/outputs/%s" % (pyth, case_name1)
    fpath2 = "%s/EucFACE_run/outputs/%s" % (pyth, case_name2)
    fpath3 = "%s/EucFACE_run/outputs/%s" % (pyth, case_name3)
    fpath4 = "%s/EucFACE_run/outputs/%s" % (pyth, case_name4)
    fpath5 = "%s/EucFACE_run/outputs/%s" % (pyth, case_name5)
    fpath6 = "%s/EucFACE_run/outputs/%s" % (pyth, case_name6)
    fpath7 = "%s/EucFACE_run/outputs/%s" % (pyth, case_name7)
    fpath8 = "%s/EucFACE_run/outputs/%s" % (pyth, case_name8)
    fpath9 = "%s/EucFACE_run/outputs/%s" % (pyth, case_name9)
    fpath10 = "%s/EucFACE_run/outputs/%s" % (pyth, case_name10)

    # Figure S1 - "Out of box CABLE"
    plot_profile_tdr_ET_error_rain(CTL, fpath1, case_name1, ring, contour, '6')

    # Figure 2 -  "Ctl"
    plot_profile_tdr_ET_error_rain(CTL, fpath2, case_name2, ring, contour, '6')

    # Figure 3 - "Sres"
    plot_profile_ET_error_rain(fpath3, case_name3, ring, contour, '6')

    # Figure 4 - "Watr"
    plot_profile_tdr_ET_error_rain(CTL, fpath4, case_name4, ring, contour, '6')

    # Figure S2 - "Hi-Res-1"
    plot_profile_tdr_ET_error_rain(CTL, fpath5, case_name5, ring, contour, '31uni')

    # Figure 5 - "Hi-Res-2"
    plot_profile_tdr_ET_error_rain(CTL, fpath6, case_name6, ring, contour, '31uni')

    # Figure S5 - "Opt"
    plot_profile_tdr_ET_error_rain(CTL, fpath7, case_name7, ring, contour, '31uni')

    # Figure 6 - "β-hvrd"
    plot_profile_tdr_ET_error_rain(CTL, fpath8, case_name8, ring, contour, '31uni')

    # Figure 7 - "β-exp"
    plot_profile_tdr_ET_error_rain(CTL, fpath9, case_name9, ring, contour, '31uni')

    # Figure S3 - "Opt-sub"
    plot_profile_tdr_ET_error_rain(CTL, fpath10, case_name10, ring, contour, '31uni')

    '''
    Drought plot - Figure 8
    '''
    plot_Rain_Fwsoil_Trans_Esoil_SH_SM( fcables, case_labels, layers, ring)

    '''
    Beta plot - Figure 9
    '''
    plot_fwsoil_boxplot_SM( fcables, case_labels, layers, ring)

    '''
    Heatwave plots - Figure 10
    '''
    plot_EF_SM_HW(fcables, case_labels, layers, ring, time_scale)

    '''
    metrics - Table 2 S2
    '''
    calc_metrics(fcables, case_labels, layers, vars, ring)

    '''
    annual values - Table 3
    '''
    annual_values(fcables, case_labels, layers, ring)
