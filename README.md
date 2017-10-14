Copyright (c) Nievas, Cecilia I., 2016-2017 

This set of Python codes implements the procedure to calculate Multidirectional
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
* OpenQuake Hazard Library (install OpenQuake:
					https://github.com/gem/oq-engine/blob/master/README.md)
* Numpy (1.9.2 or later) 
* Scipy (0.15.1 or later)
* Matplotlib (1.4.3 or later)
* h5py (2.5.0 or later)
* The accompanying files in this repository:
	- cin_MDCS.py (main code)
    - cin_models_shahi_baker.py
    - cin_model_correlation_sarotd100.py
    - cin_read_eta_model.py
    - cin_mdcs_plotting.py
	- __init__.py
	- cin_eta_distribution_empirical_RESORCE_Gaussian_Kernel.hdf5

WHAT THIS SET OF CODES DOES
===========================	
This set of Python codes implements the procedure to calculate Multidirectional
Conditional Spectra. The particular models used (implemented herein as
cin_models_shahi_baker.py, cin_model_correlation_sarotd100.py, and
cin_eta_model.py) are not an integral part of the methodology or the concept of
the Multidirectional Conditional Spectrum in itself.

The model in cin_model_correlation_sarotd100.py is exploratory and should not
be considered conclusive, as further research is needed.

Similarly, the Kernel Density Estimations provided in
cin_eta_distribution_empirical_RESORCE_Gaussian_Kernel.hdf5 or, alternatively, in
the accompanying spreadsheets (eta_empirical_Gaussian_Kernel.zip), are a
representation of eta(T,xi,theta) based on an analysis carried out using the
RESORCE database (Akkar et al., 2014), and not necessarily a universally
applicable model.

The models of Shahi and Baker (2014) implemented in cin_models_shahi_baker.py
are the personal implementation of Nievas, Cecilia I. of said publication,
and no endorsement from Shahi, S.K. and Baker, J.W. is implied.

If the user is not familiar with h5py, they can modify the code to use it
with a different input/output methodology. This will require the modification
not only of cin_MDCS.py, but also of cin_read_eta_model.py, as it is directly
reading from cin_eta_distribution_empirical_RESORCE_Gaussian_Kernel.hdf5. The
spreadsheets provided contain the same information as this hdf5 file.

cin_MDCS.py starts with a series of parameters that need to be manually set by
the user. Note that introduction of non-realistic values for these parameters
may lead to results being wrong without warning messages being raised.

An idea of how long it takes to run on a laptop computer with an Intel(R)
Core(TM) i7-6700HQ CPU (2.6 GHz) with 16 GB of RAM:
- If global_delta= 1.0 degree, it takes about 5 minutes per oscillator period.
- If global_delta= 10.0 degrees, it takes about 2 minutes per oscillator period.
Note that the amount of time it takes to run can be highly variable and non-linear.

All files contained in this repository should be placed within the same directory
for the code to run.

REFERENCES
===========================	
Akkar S, Sandkkaya M, Senyurt M, Azari Sisi A, Ay B, Traversa P, Douglas J, Cotton F, Luzi L, Hernandez B,
Godey S. 2014. Reference database for seismic ground-motion in Europe (RESORCE). Bulletin of Earthquake
Engineering 12, 311–339. DOI: 10.1007/s10518-013-9506-8.

Shahi S K, Baker JW. 2014. NGA-West2 models for ground motion directionality. Earthquake Spectra 30, 1285–
1300. DOI: 10.1193/040913EQS097M.


