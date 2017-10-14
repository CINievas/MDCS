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

WHAT THIS FILE IN PARTICULAR DOES
=================================
This Python code contains a model for the correlation between the residuals of
SaRotD100 at two different periods. The form of the equation is that of Baker
and Cornell (2006) for T >= 0.189 sec. For details regarding the fitting of
this model, please refer to:
Nievas, Cecilia I. & Sullivan, Timothy J. (2017) "A Multidirectional 
Conditional Spectrum". Earthquake Engineering & Structural Dynamics, UNDER
REVISION
This model is exploratory and should not be considered conclusive, as
further research is needed.

"""

import numpy as np


def corr_coeff_different_periods_SaRotD100(Ti,Tj):
    """
    Correlation between the residuals of SaRotD100 at two different periods.
    The form of the equation is that of Baker and Cornell (2006)
    for T >= 0.189 sec.
    The equation has been fitted for periods in the range [0.2 s, 3.00 s].
    This model is exploratory and should not be considered conclusive or suitable
    for generalised application. Further research is needed for its validation.
    """   
    coeff= 0.146
    periods = np.array([Ti, Tj])
    t_max = np.max(periods)
    t_min = np.min(periods)
    if np.fabs(t_min < 1.0E-9):
        if np.fabs(t_max < 1.0E-9):
            # At 0., 0. the correlation should be 1.0
            return 1.0
        else:
            return 0.0
    cterm = coeff * np.log(t_max / t_min)
    if cterm > (np.pi / 2.0): # It should never ever happen
        return 0.0
    else:
        return 1.0 - np.cos((np.pi / 2.) - cterm)


