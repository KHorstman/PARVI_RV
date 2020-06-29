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


#read in 16 files (simulated data for GJ229 with injected planet)
def make_list(n):
      for _ in range(n):
            yield np.array([])

date_01_counts, date_02_counts, date_03_counts, date_04_counts, date_05_counts, date_06_counts, date_07_counts, date_08_counts, date_09_counts, date_10_counts, date_11_counts, date_12_counts, date_13_counts, date_14_counts, date_15_counts, date_16_counts= make_list(16)

#file_name = ['SimGJ229idate01Hmag8RV10P10d.txt', 'SimGJ229idate02Hmag8RV10P10d.txt', 'SimGJ229idate03Hmag8RV10P10d.txt', 'SimGJ229idate04Hmag8RV10P10d.txt', 'SimGJ229idate05Hmag8RV10P10d.txt', 'SimGJ229idate06Hmag8RV10P10d.txt', 'SimGJ229idate07Hmag8RV10P10d.txt', 'SimGJ229idate08Hmag8RV10P10d.txt', 'SimGJ229idate09Hmag8RV10P10d.txt', 'SimGJ229idate10Hmag8RV10P10d.txt', 'SimGJ229idate11Hmag8RV10P10d.txt', 'SimGJ229idate12Hmag8RV10P10d.txt', 'SimGJ229idate13Hmag8RV10P10d.txt', 'SimGJ229idate14Hmag8RV10P10d.txt', 'SimGJ229idate15Hmag8RV10P10d.txt', 'SimGJ229idate16Hmag8RV10P10d.txt']
file_name = ['SimGJ229idate01Hmag8RV10P10d.txt', 'SimGJ229idate02Hmag8RV10P10d.txt']
date_name = ['date_01', 'date_02', 'date_03', 'date_04', 'date_05', 'date_06', 'date_07', 'date_08', 'date_09', 'date_10', 'date_11', 'date_12', 'date_13', 'date_14', 'date_15', 'date_16']
#date_name_counts = [date_01_counts, date_02_counts, date_03_counts, date_04_counts, date_05_counts, date_06_counts, date_07_counts, date_08_counts, date_09_counts, date_10_counts, date_11_counts, date_12_counts, date_13_counts, date_14_counts, date_15_counts, date_16_counts]

# for i in range(16):
#       date_name[i]=np.loadtxt(file_name[2], usecols=(3,4))
#       date_name_wavelength[i]=np.append(date_name_wavelength[i], date_name[i][:,0])
#       print(date_name_wavelength[i])

data = [np.loadtxt(open(file), usecols=(0,1,2,3,4,5,6)) for file in file_name]
wavelength=data[0][:,3]

#define the count data
date_01_counts=data[0][:,4]
date_02_counts=data[1][:,4]
# date_03_counts=data[2][:,4]
# date_04_counts=data[3][:,4]
# date_05_counts=data[4][:,4]
# date_06_counts=data[5][:,4]
# date_07_counts=data[6][:,4]
# date_08_counts=data[7][:,4]
# date_09_counts=data[8][:,4]
# date_10_counts=data[9][:,4]
# date_11_counts=data[10][:,4]
# date_12_counts=data[11][:,4]
# date_13_counts=data[12][:,4]
# date_14_counts=data[13][:,4]
# date_15_counts=data[14][:,4]
# date_16_counts=data[15][:,4]

#add other date for date_01 for forward modeling approach
date_01_bary=data[0][:,1]
date_01_planet_rv= data[0][:,2]
date_01_noise=data[0][:,5]
date_01_trans=data[0][:,6]

#use date_02 instead because injected RV for planet is 0 for date_01
date_02_bary=data[1][:,1]
date_02_planet_rv= data[1][:,2]
date_02_noise=data[1][:,5]
date_02_trans=data[1][:,6]

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

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# #plot of date_01
# plt.xlabel('Wavelength ($\mu$m)')
# plt.ylabel('Norm Counts + Offset', color='black')
# plt.xlim([1.5, 1.76])
# plt.ylim([0, 16.5])

# plt.scatter(wavelength, date_01_counts, c='royalblue', alpha=1, marker='o', s=.1)

# plt.show()

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# #plot of all the data
# plt.xlabel('Wavelength ($\mu$m)')
# plt.ylabel('Norm Counts + Offset', color='black')
# plt.xlim([1.57, 1.61])
# plt.ylim([0,16.5])
# plt.tick_params('y', colors='black', direction='in', which='both')

# for i in range(16):
#       plt.plot(wavelength, total_counts[i]+i, alpha=1, marker='o', markersize=.1)

# plt.savefig('GJ229_simulation_spectra.pdf')
# plt.savefig('GJ229_simulation_spectra.png', dpi=300)

# plt.show()

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

#plot phoenix/telluric model over simulated spectra
plt.xlim([1.66, 1.68])
plt.ylim([0,1])

#wavelength data in angstroms
#normalized phoenix data
filtered_data=scipy.signal.medfilt(phoenix_data)
phoenix_data_norm=((phoenix_data-filtered_data)/(2*10**12))+.85
#phoenix_data_norm=phoenix_data/np.linalg.norm(phoenix_data)

#normalized telluric data
telluric_500_counts_norm=telluric_500_counts

#normalized simulated data
date_01_counts_norm=date_01_counts

plt.plot(telluric_wavelength, telluric_500_counts_norm, alpha=1, marker='o', color='royalblue', markersize=.1)
plt.plot(phoenix_wave_data/10000, phoenix_data_norm, alpha=.5, marker='o', color='red', markersize=.1)
#plt.plot(phoenix_wave_data/10000, filtered_data, alpha=.5, marker='o', color='blue', markersize=.1)
plt.plot(wavelength, date_01_counts_norm, alpha=1, marker='o', color='green', markersize=.1)

#legend
red_patch= mpatches.Patch(color='red', label='Phoenix Model Spectra')
pink_patch= mpatches.Patch(color='royalblue', label='Telluric Model Spectra')
orange_patch= mpatches.Patch(color='green', label='Simulated Spectra')
plt.legend(handles=[red_patch, pink_patch, orange_patch], loc=1)

plt.savefig('GJ229_models_and_simulated_spectra_large.pdf')
plt.savefig('GJ229_models_and_simulated_spectra_large.png', dpi=300)

plt.show()

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

#forward modeling approach
#plt.xlim([1.50, 1.80])

c_ms=3*10**8
c_kms=3*10**5

wvs = wavelength
data = date_02_counts
noise = date_02_noise
transmission = date_02_trans
signal = (data - noise)/transmission
#wvs_signal = wvs*None #correct the wvs vector for the RV of the star (incl. both the barycentric RV and the planet induced RV)
#FIX UNITS
RV=(date_02_planet_rv/1000+date_02_bary)
wvs_signal= wvs*(1+(RV/c_kms))
#print(wvs_signal)

# model_spline = # spline inteprolation using (wvs_signal,signal)
model_spline = scipy.interpolate.splrep(wvs_signal, signal)

#other cubic spline attempt that produces a graphable object when defining the range
#to plug into model in for loop, must use .splrep to find the B-spline representation 
#model_spline = scipy.interpolate.interp1d(wvs_signal, signal, kind='cubic')
#xs= np.linspace(1.52, 1.73, np.size(wvs_signal))
#print(model_spline(xs))

#some test plot to look at the data
#plt.plot(wvs, signal, alpha=1, marker='o', color='royalblue', markersize=.1)
#plt.plot(wvs, data, alpha=1, marker='o', color='pink', markersize=.1)
#plt.plot(wvs_signal, signal, alpha=1, marker='o', color='red', markersize=.1)
#plt.plot(wvs_signal, model_spline(xs), alpha=1, marker='o', color='green', markersize=.1)
#plt.show()

#use a for loop over the RV
science_baryrv = date_02_bary

#keep rv in km/s
#should recover 6 m/s from model for injected planet RV 
rv=np.array(np.arange(0, 45, 1))
#print(rv[0])

#print(np.size(wvs))
#print(np.size(model_spline(xs)))


# for loop over rv
      # store:
      # max_like_amplitude
      # max_like_amplitude_err
      # minus2logL
      # logpost

max_like_amplitude_out=np.array([])
max_like_amplitude_err_out=np.array([])
minus2logL_out=np.array([])
logpost_out=np.array([])

# - science_baryrv)
for i in range(len(rv)):
      #The parameter we are after is rv
      wvs_shifted= wvs*(1 + (rv[i] / c_kms))
      model = scipy.interpolate.splev(wvs_shifted, model_spline, der=0) * transmission
      #plt.plot(wvs_shifted, model, alpha=.5, marker='o', color='royalblue', markersize=.1)
      #plt.plot(wvs_signal, signal, alpha=.5, marker='o', color='red', markersize=.1)
      #plt.show()

      #for now make sigmas_vec=1
      #sigmas_vector=1

      Npix = np.size(data)
      sigmas_vec = np.ones(np.size(data))*np.std(noise)
      #are norm_model and norm_data the same on purpose?
      norm_model = model / sigmas_vec
      norm_data = model / sigmas_vec
      max_like_amplitude = np.sum(norm_data * norm_model) / np.sum((norm_model) ** 2)
      max_like_amplitude_out=np.append(max_like_amplitude_out, max_like_amplitude)

      data_model = max_like_amplitude * model
      residuals = data - data_model
      HPFchi2 = np.nansum((residuals / sigmas_vec) ** 2)

      max_like_amplitude_err = np.sqrt((HPFchi2 / Npix) / np.sum(norm_model ** 2))
      max_like_amplitude_err_out=np.append(max_like_amplitude_err_out, max_like_amplitude_err)

      logdet_Sigma = np.sum(2 * np.log(sigmas_vec))
      minus2logL = Npix * (1 + np.log(HPFchi2 / Npix) + logdet_Sigma + np.log(2 * np.pi))
      minus2logL_out=np.append(minus2logL_out, minus2logL)

      slogdet_icovphi0 = np.log(1 / np.sum((norm_data) ** 2))
      logpost = -0.5 * logdet_Sigma - 0.5 * slogdet_icovphi0 - (Npix - 1 + 2 - 1) / (2) * np.log(HPFchi2)
      logpost_out=np.append(logpost_out, logpost)

#plot minus2logL as function of the rv
print(rv)
print(minus2logL_out)
print(max_like_amplitude_out)
print(max_like_amplitude_err_out)
plt.plot(rv, minus2logL_out, alpha=1, marker='o', color='pink', markersize=.1)
plt.savefig('rv_logL_date_02.png', dpi=300)
plt.show()

#plot posterior
posterior = np.exp(logpost_out-np.max(logpost_out))
plt.plot(rv, posterior, alpha=1, marker='o', color='royalblue', markersize=.1)
plt.savefig('rv_posterior_date_02.png', dpi=300)

#plt.plot(rv_samples,posterior)
plt.show()