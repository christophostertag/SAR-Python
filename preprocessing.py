import numpy as np
import os

# LOAD CALIBRATION PARAMETERS
dir = os.path.join('calibration', 'parameters')

err = np.load(os.path.join(dir,'err.npy'), allow_pickle=False)
K = np.load(os.path.join(dir,'K.npy'), allow_pickle=False)
dist_coeffs = np.load(os.path.join(dir,'dist_coeffs.npy'), allow_pickle=False)
rvecs = np.load(os.path.join(dir,'rvecs.npy'), allow_pickle=False)
tvecs = np.load(os.path.join(dir,'tvecs.npy'), allow_pickle=False)