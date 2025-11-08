
from pathlib import Path
import numpy as np
from numpy import asarray
from numpy import savetxt
import pandas as pd
import csv
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from scipy.io import savemat



filename="od_pair.npy"

data = np.load(filename);

#data.tofile('wx2.csv', sep = ',')
# Save the dictionary to a .mat file
savemat("od_pair.mat", {"data": data})
print(filename, data.shape)
