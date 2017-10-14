# -*- coding: utf-8 -*-
"""
Copyright (c) Nievas, Cecilia I., 2016-2017 

This Python code contains tools used by cin_MDCS.py, which in turn implements
the procedure to calculate Multidirectional Conditional Spectra, as described
in the following publication:

Nievas, Cecilia I. & Sullivan, Timothy J. (2017) "A Multidirectional 
Conditional Spectrum". Earthquake Engineering & Structural Dynamics, UNDER
REVISION

This code has been downloaded from: https://github.com/CINievas/MDCS

This code is free software, and can be redistributed and/or modified under the
terms of the GNU General Public License, as published by the Free Software
Foundation, either Version 3 of the License, or (at your option) any later
version. Please take a minute of your time to read the disclaimer below, and
the license in the accompanying TXT file.

DISCLAIMER
==========
This code is distributed in the hope that it will be useful to the scientific
and engineering communities. This code is distributed AS IS, WITHOUT ANY
WARRANTY, without even the implied warranty of fitness for a particular
purpose. Nievas, Cecilia I. assumes NO LIABILITY for the use of this code.

DEPENDENCIES
============
This Python code uses:
* Python 2.7
* Numpy (1.9.2 or later) 
* h5py (2.5.0 or later)
* The file eta_distribution_empirical_RESORCE_Gaussian_Kernel.hdf5 provided
in the repository.

WHAT THIS FILE IN PARTICULAR DOES
=================================
This Python code reads the Kernel Density Estimations (KDE) that provide the
probability density of eta(T,theta).
For details regarding the derivation of these KDEs, please refer to:
Nievas, Cecilia I. & Sullivan, Timothy J. (2017) "A Multidirectional 
Conditional Spectrum". Earthquake Engineering & Structural Dynamics, UNDER
REVISION
These KDEs are a representation of eta(T,xi,theta) based on an analysis
carried out using the RESORCE database (Akkar et al., 2014), and not
necessarily a universally applicable model.

"""

import h5py
import numpy as np
import os

model_file_name= os.path.join(os.curdir, 'cin_eta_distribution_empirical_RESORCE_Gaussian_Kernel.hdf5')


bins_rjb= [(1.0,4.99),(5.0,9.99),(10.0,14.99),(15.0,19.9),(20.0,24.9),
           (25.0,49.9), (50.0,74.9),(75.0,99.9),(100.0,199.9),(200.0,499.9)]
           
bins_saRD100= [(0.0,10.0),(10.0,100.0),(100.0,250.0),(250.0,1000.0)]      

allperiods = np.array([0.20,0.40,0.60,0.80,1.00,1.50,2.00,2.50,3.00,4.00]) 

min_n_recs= 20

def takeClosest(num, collection):
   return min(collection,key=lambda x:abs(x-num))

def get_eta_edges():
    eta_distrib_fle = h5py.File(model_file_name)
    eta_edges= eta_distrib_fle['Edges Eta'][:]
    eta_edges_midp= eta_distrib_fle['Edges Eta Midpoints'][:]
    eta_distrib_fle.close()
    return eta_edges, eta_edges_midp

def get_defined_periods():
    eta_distrib_fle = h5py.File(model_file_name)
    period_list= np.array([], dtype='float32')
    for key_name in eta_distrib_fle.keys():
        if ((key_name!='Edges Eta') and (key_name!='Edges Eta Midpoints')):
            period_list= np.concatenate((period_list, np.array([float(key_name.split()[0])]))) 
    eta_distrib_fle.close()
    return period_list

def define_rjb_bin(rjb):
    """
    This function assumes that 1.0<=rjb<=499.9    
    """
    for rjb_bin_case in range(0,len(bins_rjb)):
        if ((rjb>=bins_rjb[rjb_bin_case][0]) and (rjb<=bins_rjb[rjb_bin_case][1])):
            rjb_bin= rjb_bin_case
            break
    return rjb_bin

def define_SaRotD100_bin(SaRotD100_val):
    """
    This function assumes that 0.0<=SaRotD100_val<1000.0.
    If SaRotD100_val>1000.0, it assigns the last bin (250.0,1000.0).
    """
    bin_found= False
    for saRD_bin_case in range(0,len(bins_saRD100)):
        if ((SaRotD100_val>=bins_saRD100[saRD_bin_case][0]) and (SaRotD100_val<bins_saRD100[saRD_bin_case][1])):
            sa_bin= saRD_bin_case
            bin_found= True
            break
    if not bin_found:
        if SaRotD100_val>bins_saRD100[-1][1]:
            sa_bin= len(bins_saRD100) - 1
    return sa_bin

def get_probability_histogram(Tj, rjb, SaRotD100_val):
    """
    This function assumes that 1.0<=rjb<=499.9   
    This function assumes that 0.0<=SaRotD100_val<1000.0
    As the histograms exist only for 11 periods, this function takes the
    histogram corresponding to the closest defined period.
    """    
    periods_exist= get_defined_periods()
    period_to_use= takeClosest(Tj, periods_exist) 
    eta_distrib_fle = h5py.File(model_file_name)
    rjb_bin_case= define_rjb_bin(rjb)
    saRD_bin_case= define_SaRotD100_bin(SaRotD100_val)
    gr_period_rjb_sa= eta_distrib_fle['{:.2f}'.format(period_to_use)+' sec/'+'rjb '+str(bins_rjb[rjb_bin_case])+'/SaRD100 '+str(bins_saRD100[saRD_bin_case])]
    if 'Histograms' in gr_period_rjb_sa.keys():
        histogram_data= gr_period_rjb_sa['Histograms'][:]
        hist_found= True
    elif 'Histograms' in  eta_distrib_fle['{:.2f}'.format(period_to_use)+' sec/'+'rjb '+str(bins_rjb[rjb_bin_case])+'/All SaRotD100'].keys():
        histogram_data= eta_distrib_fle['{:.2f}'.format(period_to_use)+' sec/'+'rjb '+str(bins_rjb[rjb_bin_case])+'/All SaRotD100/Histograms'][:]
        hist_found= True
    else:
        hist_found= False
        histogram_data= []
    eta_distrib_fle.close()
    return histogram_data, hist_found

def get_final_probability_matrix(Tj, rjb, SaRotD100_val):
    histogram_data, hist_found= get_probability_histogram(Tj, rjb, SaRotD100_val)
    if hist_found:
        matrix_found= True
        histogram_data= histogram_data.astype('float64')
        eta_edges, eta_edges_midp= get_eta_edges()    
        probabilities= histogram_data*(eta_edges_midp[1]-eta_edges_midp[0])
        for theta in range(0,len(probabilities[:,0])):
            probabilities[theta,:]= probabilities[theta,:] / np.sum(probabilities[theta,:])
    else:
        probabilities= -999.9
        matrix_found= False
    return probabilities, matrix_found






