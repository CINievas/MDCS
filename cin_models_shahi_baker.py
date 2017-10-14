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
* Scipy (0.15.1 or later)

WHAT THIS FILE IN PARTICULAR DOES
=================================
This Python code contains implementations of some of the models from the
following publication:

Shahi, S.K. and Baker, J.W. (2014). NGA-West2 models for ground motion
directionality. Earthquake Spectra 30, 1285-1300. DOI: 10.1193/040913EQS097M

This code is the personal implementation of Nievas, Cecilia I. of said
publication, and no endorsement from Shahi, S.K. and Baker, J.W. is implied.

"""
import numpy as np
from scipy.interpolate import interp2d
from scipy.optimize import fsolve

def get_gamma_Shahi_and_Baker(Ti,Tj):
    """
    Model of Shahi and Baker (2014).
    This function reads the value of gamma from Table A1 in the electronic
    supplement of Shahi and Baker (2014) to be used for the model that
    describes the relationship between the orientations of SaRotD100 at
    different periods.
    WARNING: Coefficients along the diagonal are 999.9, simulating the 
    infinity in the original table of Shahi and Baker (2014). This brings
    some issues when needing to interpolate between infinity and something else.
    See "get_gamma_Shahi_and_Baker_enhanced".
    """
    periods_exist= np.array([0.01,0.02,0.03,0.05,0.07,0.10,0.15,0.20,0.25,
                             0.30,0.40,0.50,0.75,1.00,1.50,2.00,3.00,4.00,
                             5.00,7.50,10.0]) # periods for which gamma is defined
    table_vals= np.array([[999.9,0.579,0.186,0.07,0.042,0.031,0.022,0.021,0.02,0.02,0.02,0.018,0.014,0.013,0.01,0.007,0.004,0.005,0.007,0.007,0.007],
                          [0.579,999.9,0.188,0.071,0.042,0.031,0.022,0.021,0.019,0.02,0.02,0.018,0.014,0.013,0.01,0.007,0.004,0.005,0.007,0.007,0.007],
                          [0.186,0.188,999.9,0.072,0.042,0.03,0.022,0.021,0.019,0.019,0.02,0.018,0.013,0.012,0.009,0.007,0.004,0.006,0.007,0.007,0.007],
                          [0.07,0.071,0.072,999.9,0.041,0.028,0.02,0.019,0.017,0.017,0.016,0.015,0.012,0.01,0.007,0.006,0.004,0.005,0.005,0.006,0.005],
                          [0.042,0.042,0.042,0.041,999.9,0.031,0.019,0.017,0.016,0.016,0.015,0.013,0.01,0.009,0.007,0.005,0.003,0.004,0.004,0.004,0.005],
                          [0.031,0.031,0.03,0.028,0.031,999.9,0.02,0.015,0.014,0.014,0.011,0.011,0.009,0.007,0.005,0.003,0.003,0.003,0.003,0.003,0.003],
                          [0.022,0.022,0.022,0.02,0.019,0.02,999.9,0.019,0.013,0.013,0.01,0.009,0.006,0.005,0.003,0.002,0.000001,0.001,0.002,0.002,0.002],
                          [0.021,0.021,0.021,0.019,0.017,0.015,0.019,999.9,0.021,0.016,0.013,0.011,0.007,0.005,0.003,0.003,0.001,0.002,0.003,0.002,0.002],
                          [0.02,0.019,0.019,0.017,0.016,0.014,0.013,0.021,999.9,0.026,0.015,0.01,0.007,0.005,0.004,0.004,0.003,0.003,0.003,0.003,0.003],
                          [0.02,0.02,0.019,0.017,0.016,0.014,0.013,0.016,0.026,999.9,0.019,0.013,0.007,0.007,0.005,0.003,0.004,0.004,0.005,0.004,0.005],
                          [0.02,0.02,0.02,0.016,0.015,0.011,0.01,0.013,0.015,0.019,999.9,0.024,0.011,0.01,0.008,0.005,0.004,0.006,0.007,0.005,0.005],
                          [0.018,0.018,0.018,0.015,0.013,0.011,0.009,0.011,0.01,0.013,0.024,999.9,0.016,0.013,0.008,0.007,0.004,0.005,0.005,0.005,0.005],
                          [0.014,0.014,0.013,0.012,0.01,0.009,0.006,0.007,0.007,0.007,0.011,0.016,999.9,0.022,0.013,0.011,0.006,0.008,0.009,0.009,0.009],
                          [0.013,0.013,0.012,0.01,0.009,0.007,0.005,0.005,0.005,0.007,0.01,0.013,0.022,999.9,0.02,0.015,0.01,0.01,0.01,0.011,0.01],
                          [0.01,0.01,0.009,0.007,0.007,0.005,0.003,0.003,0.004,0.005,0.008,0.008,0.013,0.02,999.9,0.024,0.012,0.011,0.012,0.013,0.013],
                          [0.007,0.007,0.007,0.006,0.005,0.003,0.002,0.003,0.004,0.003,0.005,0.007,0.011,0.015,0.024,999.9,0.019,0.016,0.015,0.016,0.014],
                          [0.004,0.004,0.004,0.004,0.003,0.003,0.000001,0.001,0.003,0.004,0.004,0.004,0.006,0.01,0.012,0.019,999.9,0.029,0.024,0.019,0.017],
                          [0.005,0.005,0.006,0.005,0.004,0.003,0.001,0.002,0.003,0.004,0.006,0.005,0.008,0.01,0.011,0.016,0.029,999.9,0.04,0.025,0.021],
                          [0.007,0.007,0.007,0.005,0.004,0.003,0.002,0.003,0.003,0.005,0.007,0.005,0.009,0.01,0.012,0.015,0.024,0.04,999.9,0.034,0.027],
                          [0.007,0.007,0.007,0.006,0.004,0.003,0.002,0.002,0.003,0.004,0.005,0.005,0.009,0.011,0.013,0.016,0.019,0.025,0.034,999.9,0.057],
                          [0.007,0.007,0.007,0.005,0.005,0.003,0.002,0.002,0.003,0.005,0.005,0.005,0.009,0.01,0.013,0.014,0.017,0.021,0.027,0.057,999.9]])
    interpfunc= interp2d(periods_exist,periods_exist,table_vals)
    gamma_val= interpfunc(Ti,Tj)
    return gamma_val
    
def get_gamma_Shahi_and_Baker_enhanced(Ti,Tj):   
    """
    This function "solves" the problem of not being able to interpolate gamma
    around the diagonal in function "get_gamma_Shahi_and_Baker".
    If possible to get gamma directly from the original table, it is done by
    calling the function "get_gamma_Shahi_and_Baker". If this is not possible,
    it estimates the value of gamma from the value of the expected angle,
    using function "get_gamma_from_expectation".
    The output is the resulting value of gamma, and a boolean that is equal to
    True if the fsolve algorithm has converged or if the value has been read
    directly from the table.
    WARNING: derived_gamma will have a value even if the algorithm hasn't
    converged. Check the value of solution_found to know this.    
    """
    periods_exist= np.array([0.01,0.02,0.03,0.05,0.07,0.10,0.15,0.20,0.25,
                             0.30,0.40,0.50,0.75,1.00,1.50,2.00,3.00,4.00,
                             5.00,7.50,10.0]) # periods for which gamma is defined    
    i_after= np.searchsorted(periods_exist, Ti)
    j_after= np.searchsorted(periods_exist, Tj)
    surrounding_gammas= np.zeros([4])
    surrounding_gammas[0]= get_gamma_Shahi_and_Baker(periods_exist[i_after-1],periods_exist[j_after-1])
    surrounding_gammas[1]= get_gamma_Shahi_and_Baker(periods_exist[i_after-1],periods_exist[j_after])
    surrounding_gammas[2]= get_gamma_Shahi_and_Baker(periods_exist[i_after],periods_exist[j_after-1])
    surrounding_gammas[3]= get_gamma_Shahi_and_Baker(periods_exist[i_after],periods_exist[j_after])
    where_diagonal= np.where(surrounding_gammas==999.9)[0]
    if len(where_diagonal)>0:
        gamma_val, sol_found= get_gamma_from_expectation(Ti, Tj)
        if sol_found==1:
            solution_found= True
        else:
            solution_found= False
    else:
        gamma_val= get_gamma_Shahi_and_Baker(Ti,Tj)        
        solution_found= True        
    return gamma_val, solution_found
    
    
def pdf_angle_SaRotD100_Ti_Tj(Ti,Tj,angles):
    """
    Model of Shahi and Baker (2014).
    This function calculates the probability density of the angles between
    the orientations of SaRotD100 at different periods, according to the model
    of Shahi and Baker (2014):
    PDF(x)= gamma * exp(-gamma*x) / (1 - exp(-90*gamma)) for x>=90 degrees
    angles= array of angles between the directions of maximum response at Ti
    and Tj, for which the probability density will be calculated.
    WARNING: Coefficients along the diagonal are 999.9, simulating the 
    infinity in the original table of Shahi and Baker (2014). This brings
    some issues when needing to interpolate between infinity and something else.   
    See "pdf_angle_SaRotD100_Ti_Tj_enhanced".
    """
    gamma= get_gamma_Shahi_and_Baker(Ti,Tj)
    pdf_vals= np.zeros([len(angles)])
    for i,theta in enumerate(angles):
        if theta>90:
            pdf_vals[i]=0.0
        else:
            pdf_vals[i]= gamma * np.exp(-gamma * theta) / (1.0 - np.exp(-90.0 * gamma))
    return pdf_vals 
            


def cdf_angle_SaRotD100_Ti_Tj_enhanced(Ti,Tj,angles):
    """
    Model of Shahi and Baker (2014).
    This function calculates the cumulative density of the angles between
    the orientations of SaRotD100 at different periods, according to the model
    of Shahi and Baker (2014):
    PDF(x)= gamma * exp(-gamma*x) / (1 - exp(-90*gamma)) for x>=90 degrees
    angles= array of angles between the directions of maximum response at Ti
    and Tj, for which the probability density will be calculated.
    This function is not limited by the issue of the coefficients at the
    diagonal, as it uses "get_gamma_Shahi_and_Baker_enhanced" instead of
    "get_gamma_Shahi_and_Baker".
    I integrated the expression for the PDF manually, and also verified it
    with an online integral calculator.
    """
    gamma, solfound= get_gamma_Shahi_and_Baker_enhanced(Ti,Tj)
    if solfound:
        cdf_vals= np.zeros([len(angles)])
        for i,theta in enumerate(angles):
            if theta>90:
                cdf_vals[i]=1.0
            else:
                cdf_vals[i]= (1.0 - np.exp(-gamma * theta)) / (1.0 - np.exp(-gamma * 90.0))
    else:
        cdf_vals= -999.9*np.ones([len(angles)])         
    return cdf_vals         
            
def get_expectation_angle_SaRotD100_Ti_Tj_analytical_enhanced(Ti,Tj):
    """
    Model of Shahi and Baker (2014).
    The function "get_expectation_angle_SaRotD100_Ti_Tj_analytical" has the
    problem that around the diagonal interpolation is carried out with
    "infinity". This function solves this problem, by first calculating the
    expected angles for each of the period pais for which coefficients are
    available and then interpolating between expected angles.
    """
    periods_exist= np.array([0.01,0.02,0.03,0.05,0.07,0.10,0.15,0.20,0.25,
                             0.30,0.40,0.50,0.75,1.00,1.50,2.00,3.00,4.00,
                             5.00,7.50,10.0]) # periods for which gamma is defined
    expected_angles_all= np.zeros([len(periods_exist),len(periods_exist)])
    for i,ti in enumerate(periods_exist):     
        for j,tj in enumerate(periods_exist): 
            gamma= get_gamma_Shahi_and_Baker(ti,tj)
            expected_angles_all[i,j]= (1.0 - (90.0*gamma + 1.0)*np.exp(-90.0 * gamma)) / (gamma * (1.0 - np.exp(-90.0 * gamma)))
    interpfunc= interp2d(periods_exist,periods_exist,expected_angles_all,kind='linear')
    expected_val= interpfunc(Ti,Tj)
    return expected_val
    
def get_gamma_from_expectation(Ti, Tj):
    """
    This function "solves" the problem of not being able to interpolate gamma
    around the diagonal in function "get_gamma_Shahi_and_Baker".
    First, it defines the expected angle between SaRotD100 at Ti and Tj.
    Then, it reads the four possible values of gamma. This assumes that neither
    Ti nor Tj are defined in periods_exist. If they are defined, it shouldn't
    be a problem, because these four possible values are used as initial
    guesses for the solution.
    Then it tries to get gamma from the analytical expression of expectation
    as a function of gamma (the same used in function
    "get_expectation_angle_SaRotD100_Ti_Tj_analytical_enhanced").
    The output is the resulting value of gamma, and an integer that is equal
    to 1 if the fsolve algorithm has converged.
    WARNING: derived_gamma will have a value even if the algorithm hasn't
    converged. Check the value of solution_found to know this.
    """
    periods_exist= np.array([0.01,0.02,0.03,0.05,0.07,0.10,0.15,0.20,0.25,
                             0.30,0.40,0.50,0.75,1.00,1.50,2.00,3.00,4.00,
                             5.00,7.50,10.0]) # periods for which gamma is defined    
    expected_angle= get_expectation_angle_SaRotD100_Ti_Tj_analytical_enhanced(Ti,Tj)
    # Get initial guesses:
    i_after= np.searchsorted(periods_exist, Ti)
    j_after= np.searchsorted(periods_exist, Tj)
    possible_gammas= np.zeros([4])
    possible_gammas[0]= get_gamma_Shahi_and_Baker(periods_exist[i_after-1],periods_exist[j_after-1])
    possible_gammas[1]= get_gamma_Shahi_and_Baker(periods_exist[i_after-1],periods_exist[j_after])
    possible_gammas[2]= get_gamma_Shahi_and_Baker(periods_exist[i_after],periods_exist[j_after-1])
    possible_gammas[3]= get_gamma_Shahi_and_Baker(periods_exist[i_after],periods_exist[j_after])
    useful_gammas= possible_gammas[np.where(possible_gammas!=999.9)[0]]
    func= lambda gamma : expected_angle - ((1.0 - (90.0*gamma + 1.0)*np.exp(-90.0 * gamma)) / (gamma * (1.0 - np.exp(-90.0 * gamma))))
    solution_found= 0
    guess_pos= 0
    while ((solution_found!=1) and (guess_pos<len(useful_gammas))):
        derived_gamma,_,solution_found,_= fsolve(func, useful_gammas[guess_pos], full_output=True)
        guess_pos= guess_pos + 1
    return derived_gamma, solution_found



            
            
            
            