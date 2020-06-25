import numpy as np  
import matplotlib.pyplot as plt
from matplotlib import colors
import pandas as pd
from astropy.io import fits 
import scipy
from scipy import signal 


#read in 16 files (simulated data for GJ229 with injected planet)
def make_list(n):
      for _ in range(n):
            yield np.array([])

date_01_counts, date_02_counts, date_03_counts, date_04_counts, date_05_counts, date_06_counts, date_07_counts, date_08_counts, date_09_counts, date_10_counts, date_11_counts, date_12_counts, date_13_counts, date_14_counts, date_15_counts, date_16_counts= make_list(16)

file_name = ['SimGJ229idate01Hmag8RV10P10d.txt', 'SimGJ229idate02Hmag8RV10P10d.txt', 'SimGJ229idate03Hmag8RV10P10d.txt', 'SimGJ229idate04Hmag8RV10P10d.txt', 'SimGJ229idate05Hmag8RV10P10d.txt', 'SimGJ229idate06Hmag8RV10P10d.txt', 'SimGJ229idate07Hmag8RV10P10d.txt', 'SimGJ229idate08Hmag8RV10P10d.txt', 'SimGJ229idate09Hmag8RV10P10d.txt', 'SimGJ229idate10Hmag8RV10P10d.txt', 'SimGJ229idate11Hmag8RV10P10d.txt', 'SimGJ229idate12Hmag8RV10P10d.txt', 'SimGJ229idate13Hmag8RV10P10d.txt', 'SimGJ229idate14Hmag8RV10P10d.txt', 'SimGJ229idate15Hmag8RV10P10d.txt', 'SimGJ229idate16Hmag8RV10P10d.txt']
date_name = ['date_01', 'date_02', 'date_03', 'date_04', 'date_05', 'date_06', 'date_07', 'date_08', 'date_09', 'date_10', 'date_11', 'date_12', 'date_13', 'date_14', 'date_15', 'date_16']
#date_name_counts = [date_01_counts, date_02_counts, date_03_counts, date_04_counts, date_05_counts, date_06_counts, date_07_counts, date_08_counts, date_09_counts, date_10_counts, date_11_counts, date_12_counts, date_13_counts, date_14_counts, date_15_counts, date_16_counts]

# for i in range(16):
#       date_name[i]=np.loadtxt(file_name[2], usecols=(3,4))
#       date_name_wavelength[i]=np.append(date_name_wavelength[i], date_name[i][:,0])
#       print(date_name_wavelength[i])

data = [np.loadtxt(open(file), usecols=(3,4)) for file in file_name]
wavelength=data[0][:,0]

#define the count data
date_01_counts=data[0][:,1]
date_02_counts=data[1][:,1]
date_03_counts=data[2][:,1]
date_04_counts=data[3][:,1]
date_05_counts=data[4][:,1]
date_06_counts=data[5][:,1]
date_07_counts=data[6][:,1]
date_08_counts=data[7][:,1]
date_09_counts=data[8][:,1]
date_10_counts=data[9][:,1]
date_11_counts=data[10][:,1]
date_12_counts=data[11][:,1]
date_13_counts=data[12][:,1]
date_14_counts=data[13][:,1]
date_15_counts=data[14][:,1]
date_16_counts=data[15][:,1]

total_counts=[date_01_counts, date_02_counts, date_03_counts, date_04_counts, date_05_counts, date_06_counts, date_07_counts, date_08_counts, date_09_counts, date_10_counts, date_11_counts, date_12_counts, date_13_counts, date_14_counts, date_15_counts, date_16_counts]

#read in telluric line models
telluric_files = ['palomar_500.txt','palomar_1000.txt', 'palomar_5000.txt', 'palomar_10000.txt', 'palomar_20000.txt']
telluric_data = [np.loadtxt(open(file), usecols=(1,2)) for file in telluric_files]
telluric_wavelength=telluric_data[0][:,0]
telluric_500_counts=telluric_data[0][:,1]
telluric_1000_counts=telluric_data[1][:,1]
telluric_5000_counts=telluric_data[2][:,1]
telluric_10000_counts=telluric_data[3][:,1]
telluric_20000_counts=telluric_data[4][:,1]

#read in Phoenix model
phoenix_model = fits.open('lte03700-5.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits')
phoenix_data = phoenix_model[0].data

phoenix_wave_model = fits.open('WAVE_PHOENIX-ACES-AGSS-COND-2011.fits')
phoenix_wave_data = phoenix_wave_model[0].data

# #plot of date_01
# plt.xlabel('Wavelength ($\mu$m)')
# plt.ylabel('Norm Counts + Offset', color='black')
# plt.xlim([1.5, 1.76])
# plt.ylim([0, 16.5])

# plt.scatter(wavelength, date_01_counts, c='royalblue', alpha=1, marker='o', s=.1)

# plt.show()

#plot of all the data
plt.xlabel('Wavelength ($\mu$m)')
plt.ylabel('Norm Counts + Offset', color='black')
plt.xlim([1.57, 1.61])
plt.ylim([0,16.5])
plt.tick_params('y', colors='black', direction='in', which='both')

for i in range(16):
      plt.scatter(wavelength, total_counts[i]+i, alpha=1, marker='o', s=.1)

plt.savefig('GJ229_simulation_spectra.pdf')

plt.show()

#plot phoenix/telluric model over simulated spectra
#plt.xlim([1.50, 1.75])
#plt.ylim([0,1.5])

#wavelength data in angstroms
#normalized phoenix data
filtered_data=scipy.signal.medfilt(phoenix_data)
phoenix_data_norm=(filtered_data/phoenix_data)
#phoenix_data_norm=phoenix_data/np.linalg.norm(phoenix_data)

#normalized telluric data
telluric_500_counts_norm=telluric_500_counts/np.sum(telluric_500_counts)

#normalized simulated data
date_01_counts_norm=date_01_counts/np.sum(date_01_counts)

#plt.plot(telluric_wavelength, telluric_500_counts, alpha=.5, marker='o', color='pink', markersize=.1)
plt.plot(phoenix_wave_data/10000, filtered_data, alpha=.5, marker='o', color='red', markersize=.1)
#plt.plot(phoenix_wave_data/10000, filtered_data, alpha=.5, marker='o', color='blue', markersize=.1)
#plt.plot(wavelength, date_01_counts, alpha=.5, marker='o', color='orange', markersize=.1)

#legend
red_patch = mpatches.Patch(color='red', label='Phoenix Model Spectra')
pink_patch= mpatches.Patch(color='pink', label='Telluric Model Spectra')
orange_patch= mpatches.Patch(color='orange', label='Simulated Spectra')
plt.legend(handles=[red_patch, pink_patch, orange_patch])

plt.savefig('GJ229_models_and_simulated_spectra_large.pdf')
plt.savefig('GJ229_models_and_simulated_spectra_large.png', dpi=300)

plt.show()

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

wvs = wavelength
data = date_01_counts
noise = date_01_noise
transmission = date_01_trans
signal = (data - noise)/transmission
#wvs_signal = wvs*None #correct the wvs vector for the RV of the star (incl. both the barycentric RV and the planet induced RV)
wvs_signal= wvs*date_01_planet_rv*date_01_planet_rv

# model_spline = # spline inteprolation using (wvs_signal,signal)
model_spline = scipy.interpolation.cubicspline(wvs_signal, signal)
#scipy.interpolate.cubicspline

#use a for loop over the RV
science_baryrv = date_01_bary

# # for loop over rv
# if 1:
#       rv = None # The parameter we are after
#       model = interpolate.splev(wvs*(1 - (rv - science_baryrv) / c_kms), model_spline, der=0) * transmission
#
#
#       Npix = np.size(data)
#       sigmas_vec = np.ones(np.size(data))*np.std(noise)
#       norm_model = model / sigmas_vec
#       norm_data = model / sigmas_vec
#       max_like_amplitude = np.sum(norm_data * norm_model) / np.sum((norm_model) ** 2)
#
#       data_model = max_like_amplitude * model
#       residuals = data - data_model
#       HPFchi2 = np.nansum((ravelresiduals / sigmas_vec) ** 2)
#
#       max_like_amplitude_err = np.sqrt((HPFchi2 / Npix) / np.sum(norm_model ** 2))
#
#       logdet_Sigma = np.sum(2 * np.log(sigmas_vec))
#       minus2logL = Npix * (1 + np.log(HPFchi2 / Npix) + logdet_Sigma + np.log(2 * np.pi))
#
#       slogdet_icovphi0 = np.log(1 / np.sum((norm_HPFmodel) ** 2))
#       logpost = -0.5 * logdet_Sigma - 0.5 * slogdet_icovphi0 - (Npixs_HPFdata - 1 + 2 - 1) / (2) * np.log(HPFchi2)
#
#       # store:
#       # max_like_amplitude
#       # max_like_amplitude_err
#       # minus2logL
#       # logpost
#
# #plot minus2logL as function of the RV
#
# # plot posterior
# posterior = np.exp(logpost_out-np.max(logpost_out))
#
# plt.plot(rv_samples,posterior)