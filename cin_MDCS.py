# -*- coding: utf-8 -*-
"""
Copyright (c) Nievas, Cecilia I., 2016-2017 

This Python code implements the procedure to calculate Multidirectional
Conditional Spectra, as described in the following publication:

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
* OpenQuake Hazard Library (install OpenQuake: https://github.com/gem/oq-engine/blob/master/README.md)
* Numpy (1.9.2 or later) 
* Scipy (0.15.1 or later)
* h5py (2.5.0 or later)
* The accompanying files in this repository:
    - cin_models_shahi_baker.py
    - cin_model_correlation_sarotd100.py
    - cin_eta_model.py
    - cin_mdcs_plotting.py
	- __init__.py
 
If the user is not familiar with h5py, they can modify the code to use it
with a different input/output methodology.

"""
import numpy as np
import h5py
import os
import datetime
from scipy.stats import norm
import cin_models_shahi_baker as cin_shahi_baker
import cin_model_correlation_sarotd100 as cin_corr_SaRotD100
import cin_read_eta_model as cin_eta_model
import cin_mdcs_plotting as auxplots
from openquake.hazardlib.gsim.base import (RuptureContext,
                                           SitesContext,
                                           DistancesContext)
from openquake.hazardlib.imt import SA
from openquake.hazardlib import const
from openquake.hazardlib.gsim.hong_goda_2007 import HongGoda2007

# The output will be placed in the directory from which this file is run (a
# subfolder will be created).
main_path= os.curdir

#############################################################################
# CONTROL PARAMETERS AND INPUT DATA
#
# THE USER NEEDS TO PROVIDE THESE

plot_key_sa= True # Set to true if you want plots of the samples of SaRotD100
plot_key_goggles= True # Set to true if you want plots of the samples of Eta

# General array with all periods Ti at which the MDCS will be calculated.
# It is adjusted below to comply with the maximum and minimum periods for
# which the input models are defined, and to make sure that T* is included.
all_periods= np.logspace(-2.0,1.0,num=50,base=10.0)

# The MDCS will be calculated for global angles (angle between the direction
# of SaRotD100(T*) and the direction being calculated) between 0º and 90º,
# every "global_delta" degrees. The choice of numbers like 1º, 5º or 10º is
# OK, but other numbers might require the code to be modified. ONLY USE
# INTEGERS. Note that no error will be raised if this number is selected
# wrongly. Note that output results need to be interpreted according to this
# choice.
global_delta= 1.0 # degrees

# You need to provide the following parameters:
T_star= 0.25 # sec - The period T* upon which everything will be conditioned.
saRD100_star= 270 # cm/s2 - SaRotD100(T*) at the conditioning period T*
saRD100_star= saRD100_star / 980.665 # convert saRD100_star to g, because the Hong and Goda GMPE is in g
m_star= 5.5 # Mw* mean causal earthquake magnitude (from disaggregation)
rjb_star= 20.0 # km mean causal Joyner-Boore distance (from disaggregation)
vs_30= 760.0 # m/sec
#############################################################################

# Create a directory where results will be saved. The directory name includes
# date and time in which the analysis is started.
time_now= datetime.datetime.now()
time_name= str(time_now.year)+'_'+str(time_now.month).zfill(2)+'_'+str(time_now.day).zfill(2)+'_'+str(time_now.hour).zfill(2)+'_'+str(time_now.minute).zfill(2)+'_'+str(time_now.second).zfill(2)
print 'Started at '+time_name
out_path= os.path.join(main_path, 'out_' +time_name + '_T_star_'+'{:.0f}'.format(T_star*100)+'_sa_star_'+'{:.0f}'.format(saRD100_star*100))
os.mkdir(out_path)
# Create a TXT file with the input parameters:
foutput= open(os.path.join(out_path,'00_input_data.txt'), "w")
foutput.write(out_path+'\n')  
foutput.write('Tstar = '+'{:.2f}'.format(T_star)+' sec.\n')  
foutput.write('SaRotD100 star = '+'{:.3f}'.format(saRD100_star)+' g.\n')  
foutput.write('Mw star = '+'{:.2f}'.format(m_star)+'.\n')  
foutput.write('rjb star = '+'{:.2f}'.format(rjb_star)+' km.\n')  
foutput.write('Vs30 star = '+'{:.2f}'.format(vs_30)+' m/sec.\n')
foutput.write('Global angles between 0 and 90, every '+'{:.1f}'.format(global_delta)+' degrees.\n')
foutput.write('   WARNING: READ THE CODE. IF THE "global_delta" HAS NOT BEEN PROPERLY SELECTED, RESULTS MIGHT NOT MAKE SENSE.\n')
foutput.close()

# Adjust array of periods to comply with the maximum and minimum periods for
# which the input models are defined, and to make sure that T* is included.
all_periods= all_periods[np.where(all_periods<=3.0)[0]] # condition for Hong and Goda (2007) GMPE
all_periods= all_periods[np.where(all_periods>=0.1)[0]] # condition for Hong and Goda (2007) Eta model
if T_star not in all_periods: # insert T* in all_periods if it isn't there
    j_pos_T_star= np.searchsorted(all_periods,T_star)
    all_periods= np.insert(all_periods,j_pos_T_star,T_star)
    

#############################################################################
# DEFINITION OF CONDITIONAL SPECTRUM IN TERMS OF SaRotD100(T)
#############################################################################
print 'Discretising SaRotD100 Conditional Spectrum.'
# Define input parameters for OpenQuake. The OpenQuake implementation of the
# GMPE of Hong & Goda (2007) is used herein.
sctx = SitesContext()
sctx.vs30 = np.array([vs_30])
dctx = DistancesContext()
dctx.rjb = np.array([rjb_star])
rctx = RuptureContext()
rctx.mag = m_star
rctx.rake = 0.0 
stddev_types = [const.StdDev.TOTAL]
gsim = HongGoda2007()

# Calculate the number of standard deviations, epsi_star, by which the target
# acceleration value saRD100_star exceeds its corresponding mean:
saRD100_star_ln_mean, saRD100_star_ln_stdev= gsim.get_mean_and_stddevs(sctx, rctx, dctx, SA(T_star), stddev_types)  # output is in LN(g)
epsi_star= (np.log(saRD100_star) - saRD100_star_ln_mean)/ saRD100_star_ln_stdev
epsi_star= epsi_star[0,0]

# Correlation between epsilons of SaRotD100 for T_star and all_periods:
correl_all= np.ones([len(all_periods)])
for j,Tj in enumerate(all_periods):
    correl_all[j]= cin_corr_SaRotD100.corr_coeff_different_periods_SaRotD100(T_star,Tj)

# Mean and standard deviation for m_star, rjb_star and all_periods (from GMPE):    
saRD100_ln_mean= np.zeros([len(all_periods)]) 
saRD100_ln_stdev= np.zeros([len(all_periods)]) 
imts = [SA(period) for period in all_periods]
for j, imt in enumerate(imts): # each imt is one period Tj from all_periods
    aux_mean, aux_stdev= gsim.get_mean_and_stddevs(sctx, rctx, dctx, imt, stddev_types)   
    saRD100_ln_mean[j]= aux_mean # output is in LN(g)
    saRD100_ln_stdev[j]= aux_stdev[0][0] # output is in LN(g)
    
# Conditional means and standard deviations at all_periods:
saRD100_condit_ln_mean= saRD100_ln_mean + correl_all * saRD100_ln_stdev * epsi_star # Conditional Mean Spectrum at each period Ti, output is in LN(g)
saRD100_condit_ln_stdev= saRD100_ln_stdev * np.sqrt(np.ones_like(correl_all) - correl_all**2.0) # Conditional Standard Deviation at each period Ti

# Discretise each lognormal distribution with mean saRD100_condit_ln_mean and
# standard deviation saRD100_condit_ln_stdev. Each bin defined by cdf_edges
# has a 2.5% of occurrence. 
cdf_edges= np.linspace(0.0, 1.0, num=41)
cdf_levels= np.linspace(0.0125, 0.9875, num=40) # mid-points of cdf_edges

discr_saRD100_all= np.zeros([len(all_periods),len(cdf_levels)]) # discrete values of SaRotD100 (each row corresponds to a period)
discr_saRD100_prob_all= np.zeros([len(all_periods),len(cdf_levels)]) # this is trivial, all values should be 0.025 (it is done like this so as to be able to define the discretization in whichever way one wants)

for j,Tj in enumerate(all_periods):
    discr_saRD100_all[j,:]= norm.ppf(cdf_levels, loc= saRD100_condit_ln_mean[j], scale=saRD100_condit_ln_stdev[j])
    for jj in range(0,len(discr_saRD100_all[j,:])):
        discr_saRD100_prob_all[j,jj]= cdf_edges[jj+1] - cdf_edges[jj]

# Plot the conditional spectrum with the discretisation points:
if plot_key_sa:
    filename= os.path.join(out_path,'condit_spectr_samples_log_T_star_'+'{:.0f}'.format(T_star*100)+'_sa_star_'+'{:.0f}'.format(saRD100_star*100)) 
    auxplots.plot_CS_with_samples(saRD100_condit_ln_mean,saRD100_condit_ln_stdev,
                                  all_periods,discr_saRD100_all,filename,'png')
    filename= os.path.join(out_path,'condit_spectr_samples_nonlog_T_star_'+'{:.0f}'.format(T_star*100)+'_sa_star_'+'{:.0f}'.format(saRD100_star*100))
    auxplots.plot_CS_with_samples_nonlog(saRD100_condit_ln_mean,saRD100_condit_ln_stdev,
                                         all_periods,discr_saRD100_all,filename,'png')
                                  
#############################################################################
# DISCRETISE ANGLES BETWEEN SaRotD100(T*) and SaRotD100(all_periods)
#############################################################################
print 'Discretising angles between SaRotD100(Tstar) and SaRotD100(all_periods).'
discr_angle_91= np.linspace(0.0,90.0,num=91)
discr_angle_edges_91= np.concatenate((np.array([0.0,0.5]),np.linspace(1.5,89.5,num=89),np.array([90.0])))
discr_angle_prob_91= np.zeros([len(all_periods),len(discr_angle_91)])

for j,Tj in enumerate(all_periods):
    if j!=j_pos_T_star:
        cdf_vals_edges_all= cin_shahi_baker.cdf_angle_SaRotD100_Ti_Tj_enhanced(T_star,Tj,discr_angle_edges_91)
        for jj in range(0,len(discr_angle_91)):
            discr_angle_prob_91[j,jj]= cdf_vals_edges_all[jj+1] - cdf_vals_edges_all[jj]
    else:
        discr_angle_prob_91[j,0]= 1.00

# discr_angle_prob_181 contains the probability of the angle between the 
# maximum response direction at T* and Ti falling in the bin defined by
# discr_angle_edges_91. First bin is 0º-0.5º, last bin is 179.5º-180º, all
# other bins are 1º-wide.     
discr_angle_prob_181= np.zeros([len(all_periods),181])   
discr_angle_prob_181[:,:91]= discr_angle_prob_91 / 2.0
discr_angle_prob_181[:,91:]= np.fliplr(discr_angle_prob_91[:,:90]) / 2.0
discr_angle_prob_181[:,90]= 2.0 * discr_angle_prob_181[:,90]

#############################################################################
# DEFINE ETA AT ALL LOCAL ANGLES AND ASSEMBLE ALL
#############################################################################
print 'Defining eta and assembling all data.'
eta_edges, eta_edges_midp= cin_eta_model.get_eta_edges()

num_global_theta_cases= int(90.0/global_delta)+1
global_theta_values= np.linspace(0.0,90.0,num=num_global_theta_cases)

final_expect_mean_all= np.zeros([len(all_periods),num_global_theta_cases])
final_expect_ln_mean_all= np.zeros([len(all_periods),num_global_theta_cases])

num_sa_bins= int(np.ceil(np.exp(np.max(discr_saRD100_all))/0.001))
final_dist_sa_edges= np.linspace(0.0,np.exp(np.max(discr_saRD100_all)),num=num_sa_bins+1)
final_dist_ln_sa_edges= np.logspace(np.min(discr_saRD100_all)+np.log(eta_edges_midp[0]), np.max(discr_saRD100_all), num=num_sa_bins+1, base=np.exp(1))
final_distributions_all= np.zeros([len(all_periods),num_global_theta_cases,num_sa_bins])
final_distributions_ln_all= np.zeros([len(all_periods),num_global_theta_cases,num_sa_bins])

for j,Tj in enumerate(all_periods): # Go period by period
    print '  Working on T = '+'{:.3f}'.format(Tj)
    for k, ln_sa_k in enumerate(discr_saRD100_all[j,:]): # Take each value of SaRotD100 from the destribution defined by the CS (each value with a 2.5% of occurrence)
        sa_k= np.exp(ln_sa_k) # in g
        sa_k_cms2= sa_k * 980.665 # in cm/s2
        prob_sa_jk= discr_saRD100_prob_all[j,k]
        eta_prob_matrix_91, matrix_found= cin_eta_model.get_final_probability_matrix(Tj, round(rjb_star,2), round(sa_k_cms2,1)) # 91 angles (rows), 40 columns (eta bins)
        if matrix_found:
            eta_prob_matrix_181= np.zeros([181,len(eta_prob_matrix_91[0,:])])
            eta_prob_matrix_181[:91,:]= eta_prob_matrix_91 
            eta_prob_matrix_181[91:,:]= np.flipud(eta_prob_matrix_91[:90]) # 181 angles (rows), 40 columns (eta bins)  
            for global_theta_pos, global_theta in enumerate(global_theta_values):
                for alpha in np.linspace(0.0,180.0,num=181):
                    prob_alpha= discr_angle_prob_181[j,int(alpha)]
                    local_theta= np.abs(global_theta-alpha)
                    min_val_eta= np.abs(np.cos(np.deg2rad(local_theta)))
                    first_bin=  np.searchsorted(eta_edges,min_val_eta) - 1
                    first_bin_mp= (min_val_eta + eta_edges[first_bin+1])/2.0
                    adjusted_eta_midp= np.concatenate((np.array([first_bin_mp]),eta_edges_midp[first_bin+1:])) # adjust the first relevant eta value according to cos(theta)
                    for eta_adjusted_pos, eta_mp_val in enumerate(adjusted_eta_midp):
                        prob_position= int(eta_adjusted_pos + first_bin)
                        final_expect_mean_all[j,global_theta_pos]= final_expect_mean_all[j,global_theta_pos] + prob_sa_jk * prob_alpha * eta_prob_matrix_181[int(local_theta),prob_position] * eta_mp_val * sa_k
                        final_expect_ln_mean_all[j,global_theta_pos]= final_expect_ln_mean_all[j,global_theta_pos] + np.log(eta_mp_val * sa_k) * prob_sa_jk * prob_alpha * eta_prob_matrix_181[int(local_theta),prob_position]
                        final_distributions_all[j,global_theta_pos,np.searchsorted(final_dist_sa_edges, eta_mp_val*sa_k)-1]= final_distributions_all[j,global_theta_pos,np.searchsorted(final_dist_sa_edges, eta_mp_val*sa_k)-1] + prob_sa_jk * prob_alpha * eta_prob_matrix_181[int(local_theta),prob_position]
                        final_distributions_ln_all[j,global_theta_pos,np.searchsorted(final_dist_ln_sa_edges, eta_mp_val*sa_k)-1]= final_distributions_ln_all[j,global_theta_pos,np.searchsorted(final_dist_ln_sa_edges, eta_mp_val*sa_k)-1] + prob_sa_jk * prob_alpha * eta_prob_matrix_181[int(local_theta),prob_position]
        else: # probability matrix not defined for this case of T, rjb and SaRotD100
            print '     Eta matrix not found for T='+'{:.3f}'.format(Tj)+', rjb='+'{:.1f}'.format(rjb_star)+', SaRotD100='+'{:.3f}'.format(sa_k_cms2)          

print 'Calculating dispersion.'
final_cum_sum_stdev= np.zeros([len(all_periods),num_global_theta_cases]) 
final_cum_sum_ln_stdev= np.zeros([len(all_periods),num_global_theta_cases]) 
for j,Tj in enumerate(all_periods):
    print '  Working on T = '+'{:.3f}'.format(Tj)
    for k, ln_sa_k in enumerate(discr_saRD100_all[j,:]):
        sa_k= np.exp(ln_sa_k) # in g
        sa_k_cms2= sa_k * 980.665 # in cm/s2
        prob_sa_jk= discr_saRD100_prob_all[j,k]
        eta_prob_matrix_91, matrix_found= cin_eta_model.get_final_probability_matrix(Tj, round(rjb_star,2), round(sa_k_cms2,1)) # 91 angles (rows), 40 columns (eta bins)
        if matrix_found:
            eta_prob_matrix_181= np.zeros([181,len(eta_prob_matrix_91[0,:])])
            eta_prob_matrix_181[:91,:]= eta_prob_matrix_91 
            eta_prob_matrix_181[91:,:]= np.flipud(eta_prob_matrix_91[:90]) # 181 angles (rows), 40 columns (eta bins)  
            for global_theta_pos, global_theta in enumerate(global_theta_values):
                for alpha in np.linspace(0.0,180.0,num=181):
                    prob_alpha= discr_angle_prob_181[j,int(alpha)]
                    local_theta= np.abs(global_theta-alpha)
                    min_val_eta= np.abs(np.cos(np.deg2rad(local_theta)))
                    first_bin=  np.searchsorted(eta_edges,min_val_eta) - 1
                    first_bin_mp= (min_val_eta + eta_edges[first_bin+1])/2.0
                    adjusted_eta_midp= np.concatenate((np.array([first_bin_mp]),eta_edges_midp[first_bin+1:])) # adjust the first relevant eta value according to cos(theta)
                    for eta_adjusted_pos, eta_mp_val in enumerate(adjusted_eta_midp):
                        prob_position= int(eta_adjusted_pos + first_bin)
                        final_cum_sum_stdev[j,global_theta_pos]= final_cum_sum_stdev[j,global_theta_pos] + ((eta_mp_val * sa_k - final_expect_mean_all[j,global_theta_pos])**2.0) * (prob_sa_jk * prob_alpha * eta_prob_matrix_181[int(local_theta),prob_position])
                        final_cum_sum_ln_stdev[j,global_theta_pos]= final_cum_sum_ln_stdev[j,global_theta_pos] + ((np.log(eta_mp_val * sa_k) - final_expect_ln_mean_all[j,global_theta_pos])**2.0) * (prob_sa_jk * prob_alpha * eta_prob_matrix_181[int(local_theta),prob_position])
        else: # probability matrix not defined for this case of T, rjb and SaRotD100
            print '     Eta matrix not found for T='+'{:.3f}'.format(Tj)+', rjb='+'{:.1f}'.format(rjb_star)+', SaRotD100='+'{:.3f}'.format(sa_k)          

final_stdev_all= np.sqrt(final_cum_sum_stdev) # one row per period, one column per angle
final_ln_stdev_all= np.sqrt(final_cum_sum_ln_stdev) # one row per period, one column per angle

#############################################################################
# SAVING TO HDF5 FILE
#############################################################################
print 'Saving to HDF5 file.'
results_fle = h5py.File(os.path.join(out_path,'00_results.hdf5'))

dataset= results_fle.create_dataset('Periods (s)', all_periods.shape, dtype='float64')
dataset[:]= all_periods 

results_fle.attrs['T star (s)']= T_star
results_fle.attrs['SaRotD100 star (g)']= saRD100_star
results_fle.attrs['Mw star']= m_star
results_fle.attrs['rjb star (km)']= rjb_star
results_fle.attrs['Vs30 star (m/s)']= vs_30
results_fle.attrs['Epsilon star']= epsi_star
results_fle.attrs['Global Theta every (deg)']= global_delta

discr_group= results_fle.create_group('Discretisation SaRotD100')
dataset= discr_group.create_dataset('Values (LN(g))', discr_saRD100_all.shape, dtype='float64')
dataset[:]= discr_saRD100_all 
dataset= discr_group.create_dataset('Probabilities', discr_saRD100_prob_all.shape, dtype='float64')
dataset[:]= discr_saRD100_prob_all 

discr_group= results_fle.create_group('Discretisation Alpha Angles')
dataset= discr_group.create_dataset('Probabilities', discr_angle_prob_181.shape, dtype='float64')
dataset[:]= discr_angle_prob_181 

stats_group= results_fle.create_group('Statistics per Period and Angle')
dataset= stats_group.create_dataset('Means (g)', final_expect_mean_all.shape, dtype='float64')
dataset[:]= final_expect_mean_all
dataset= stats_group.create_dataset('Log. Means (LN(g))', final_expect_ln_mean_all.shape, dtype='float64')
dataset[:]= final_expect_ln_mean_all 
dataset= stats_group.create_dataset('St. Devs. (g)', final_stdev_all.shape, dtype='float64')
dataset[:]= final_stdev_all
dataset= stats_group.create_dataset('Log. St. Devs. (LN(g))', final_ln_stdev_all.shape, dtype='float64')
dataset[:]= final_ln_stdev_all

distrib_group= results_fle.create_group('Distributions per Period and Angle')
linear_group= distrib_group.create_group('Linear Scale')
dataset= linear_group.create_dataset('Probabilities', final_distributions_all.shape, dtype='float64')
dataset[:]= final_distributions_all
dataset= linear_group.create_dataset('Sa Edges (g)', final_dist_sa_edges.shape, dtype='float64')
dataset[:]= final_dist_sa_edges
log_group= distrib_group.create_group('Log. Scale')
dataset= log_group.create_dataset('Probabilities', final_distributions_ln_all.shape, dtype='float64')
dataset[:]= final_distributions_ln_all
dataset= log_group.create_dataset('Sa Edges (g)', final_dist_ln_sa_edges.shape, dtype='float64')
dataset[:]= final_dist_ln_sa_edges

results_fle.close()


#############################################################################
# PLOT RESULTING DIRECTIONAL SPECTRA
#############################################################################
print 'Plotting.'
# THESE ANGLES NEED TO EXIST, i.e. THEY NEED TO BE PART OF global_theta_values. 
# If they don't exist, no error may be raised, but the plots will be wrong.
angles_of_interest= [0,30,60,90] 
for theta in angles_of_interest:
    theta_pos= np.searchsorted(global_theta_values, theta) 
    mean_per_period= final_expect_mean_all[:,theta_pos]
    stdev_per_period= final_stdev_all[:,theta_pos]
    ln_mean_per_period= final_expect_ln_mean_all[:,theta_pos]
    ln_stdev_per_period= final_ln_stdev_all[:,theta_pos]
    filename= os.path.join(out_path,'spectrum_mean_'+str(theta)+'deg_T_star_'+'{:.0f}'.format(T_star*100)+'_sa_star_'+'{:.0f}'.format(saRD100_star*100)+'_Tj_'+'{:.0f}'.format(Tj*1000))
    auxplots.plot_directional_spectrum(theta, T_star, saRD100_star, all_periods,
                                       mean_per_period, stdev_per_period, filename,'png') 
    filename= os.path.join(out_path,'spectrum_log_'+str(theta)+'deg_T_star_'+'{:.0f}'.format(T_star*100)+'_sa_star_'+'{:.0f}'.format(saRD100_star*100)+'_Tj_'+'{:.0f}'.format(Tj*1000))
    auxplots.plot_directional_spectrum_log(theta, T_star, saRD100_star, all_periods,
                                           ln_mean_per_period, ln_stdev_per_period, filename,'png')
                                           
for theta in angles_of_interest:    
    theta_pos= np.searchsorted(global_theta_values, theta)
    for j,Tj in enumerate(all_periods): 
        filename= os.path.join(out_path,'hist_linear_T_'+'{:.0f}'.format(Tj*100)+'_angle_'+'{:.0f}'.format(theta))
        auxplots.plot_final_distribution_analytical(final_dist_sa_edges, final_distributions_all[j,theta_pos,:], Tj, theta, 'linear', filename,'png')
        filename= os.path.join(out_path,'hist_log_T_'+'{:.0f}'.format(Tj*100)+'_angle_'+'{:.0f}'.format(theta))
        auxplots.plot_final_distribution_analytical(final_dist_ln_sa_edges, final_distributions_ln_all[j,theta_pos,:], Tj, theta, 'logarithmic', filename,'png')



#############################################################################
# THE END
#############################################################################

time_end= datetime.datetime.now()
time_name_end= str(time_end.year)+'_'+str(time_end.month).zfill(2)+'_'+str(time_end.day).zfill(2)+'_'+str(time_end.hour).zfill(2)+'_'+str(time_end.minute).zfill(2)+'_'+str(time_end.second).zfill(2)
print 'Started at '+time_name+', finished at '+time_name_end



