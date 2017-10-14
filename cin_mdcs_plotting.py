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
* Matplotlib (1.4.3 or later)

WHAT THIS FILE IN PARTICULAR DOES
=================================
This Python code contains plotting functions.

"""
import numpy as np
import matplotlib.pyplot as plt

def plot_CS_with_samples(ln_mean, ln_stdev, periods, samples, filename, filetype):
    fig = plt.figure(figsize=(8,6))
    ax= fig.add_subplot(111)
    ax.plot(periods, np.exp(samples), linestyle='None', marker='o', markersize=4.0, markeredgecolor= '0.5', markerfacecolor='None')    
    ax.plot(periods, np.exp(ln_mean), linestyle='-', linewidth=3.0, color='k', marker='None')
    ax.plot(periods, np.exp(ln_mean+ln_stdev), linestyle='--', linewidth=2.0, color='k', marker='None')
    ax.plot(periods, np.exp(ln_mean-ln_stdev), linestyle='--', linewidth=2.0, color='k', marker='None')
    ax.loglog()
    ax.set_xlabel('T (s)', fontsize= 16.0)
    ax.set_ylabel(r'$Sa_{RotD100}(T)$', fontsize= 16.0)
    ax.grid(True, which='both')
    fig.tight_layout()
    plt.savefig(filename+'.'+filetype, dpi=300, format=filetype)
    plt.close()  
    plt.clf()   

def plot_CS_with_samples_nonlog(ln_mean, ln_stdev, periods, samples, filename, filetype):
    fig = plt.figure(figsize=(8,6))
    ax= fig.add_subplot(111)
    ax.plot(periods, np.exp(samples), linestyle='None', marker='o', markersize=4.0, markeredgecolor= '0.5', markerfacecolor='None')    
    ax.plot(periods, np.exp(ln_mean), linestyle='-', linewidth=3.0, color='k', marker='None')
    ax.plot(periods, np.exp(ln_mean+ln_stdev), linestyle='--', linewidth=2.0, color='k', marker='None')
    ax.plot(periods, np.exp(ln_mean-ln_stdev), linestyle='--', linewidth=2.0, color='k', marker='None')
    ax.set_xlabel('T (s)', fontsize= 16.0)
    ax.set_ylabel(r'$Sa_{RotD100}(T)$', fontsize= 16.0)
    ax.grid(True, which='both')
    fig.tight_layout()
    plt.savefig(filename+'.'+filetype, dpi=300, format=filetype)
    plt.close()  
    plt.clf()    

def plot_directional_spectrum(theta, T_star, saRD100_star, all_periods,
                              mean_per_period, stdev_per_period, filename, filetype):
    fig = plt.figure(figsize=(8,5))                              
    ax= fig.add_subplot(1,1,1)
    ax.plot(all_periods, mean_per_period, linestyle='-', linewidth=3.0, color='k', marker='o', markersize=6.0)
    ax.plot(all_periods, mean_per_period+stdev_per_period, linestyle='--', linewidth=2.0, color='k', marker='None')
    ax.plot(all_periods, mean_per_period-stdev_per_period, linestyle='--', linewidth=2.0, color='k', marker='None')    
    if ((theta==0.0) or (theta==180.0)):
        ax.plot(T_star, saRD100_star, linestyle='None', marker='*', markersize=8.0)
    ax.set_xlabel('Period (s)', fontsize=16.0)    
    ax.set_ylabel('Sa(T) (g)', fontsize=16.0)
    ax.set_title('Angle: '+'{:.1f}'.format(theta)+r'$^{\circ}$'+r' - $T^*$: '+'{:.2f}'.format(T_star)+' - '+r'$Sa_{RotD100}(T^*)$: '+'{:.3f}'.format(saRD100_star)+' g', fontsize=16.0)
    fig.tight_layout()
    plt.savefig(filename+'.'+filetype, dpi=300, format=filetype)
    plt.close()  
    plt.clf() 

def plot_directional_spectrum_log(theta, T_star, saRD100_star, all_periods,
                                  ln_mean_per_period, ln_stdev_per_period, filename, filetype):
    fig = plt.figure(figsize=(8,5))                              
    ax= fig.add_subplot(1,1,1)
    ax.plot(all_periods, np.exp(ln_mean_per_period), linestyle='-', linewidth=3.0, color='k', marker='o', markersize=6.0)
    ax.plot(all_periods, np.exp(ln_mean_per_period+ln_stdev_per_period), linestyle='--', linewidth=2.0, color='k', marker='None')
    ax.plot(all_periods, np.exp(ln_mean_per_period-ln_stdev_per_period), linestyle='--', linewidth=2.0, color='k', marker='None')    
    if ((theta==0.0) or (theta==180.0)):
        ax.plot(T_star, saRD100_star, linestyle='None', marker='*', markersize=8.0)
    ax.set_xlabel('Period (s)', fontsize=16.0)    
    ax.set_ylabel('Sa(T) (g)', fontsize=16.0)
    ax.set_title('Angle: '+'{:.1f}'.format(theta)+r'$^{\circ}$'+r' - $T^*$: '+'{:.2f}'.format(T_star)+' - '+r'$Sa_{RotD100}(T^*)$: '+'{:.3f}'.format(saRD100_star)+' g', fontsize=16.0)
    fig.tight_layout()
    plt.savefig(filename+'.'+filetype, dpi=300, format=filetype)
    plt.close()  
    plt.clf() 

def plot_final_distribution_analytical(edges, y_vals, Tj, angle, typekey, filename, filetype):
    """
    "y_vals" are the heights of the bars delimited by "edges"
    """
    fig = plt.figure(figsize=(8,5))                              
    ax= fig.add_subplot(1,1,1)    
    ax.bar(edges[:-1], y_vals, width=edges[1]-edges[0], color='k', alpha=0.5)
    if typekey=='logarithmic':
        ax.set_xlabel('LN[Sa(T) (g)]', fontsize=16.0)
    elif typekey=='linear':
        ax.set_xlabel('Sa(T) (g)', fontsize=16.0)
    ax.set_ylabel('Density', fontsize=16.0)
    ax.tick_params(labelsize=14.0) 
    ax.set_title('T = '+'{:.3f}'.format(Tj)+' s, '+'{:.0f}'.format(angle)+r'$^{\circ}$', fontsize=16.0)
    fig.tight_layout()
    plt.savefig(filename+'.'+filetype, dpi=300, format=filetype)
    plt.close()  
    plt.clf()  


