import numpy as np  
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patches as mpatches 
import matplotlib.lines as mlines
import pandas as pd
from astropy.io import fits 
import scipy
from scipy import signal 
from scipy import interpolate

#in one of the .fits files
hdu = fits.open('GJ229_R01_20191118101349_deg0_sp.fits')
spec_data = hdu[1].data

#raw flux vs wavelength of order 15
wave = spec_data[0][15]
flux = spec_data[1][15]

plt.plot(wave, flux, marker='o', color='royalblue')

plt.show()