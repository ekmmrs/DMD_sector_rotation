# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 16:25:16 2019

@author: Colton Smith
"""

import numpy as np
from pydmd import DMD

vals = np.array([[-2,6,1,1,-1],[-1,5,1,2,-1],[0,4,2,1,-1],[1,3,2,2,-1],[2,2,3,1,-1],[3,1,3,2,-1]])

dmd = DMD(svd_rank = 2)
vals_sub = vals[:5,:]
dmd.fit(vals_sub.T)
dmd.dmd_time['tend'] *= (1+1/6)
recon = dmd.reconstructed_data.real.T

print('Actual :',vals[5,:])
print('Predicted :',recon[5,:])