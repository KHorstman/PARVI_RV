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
#from wavcal import convolve_spectrum_line_width,convolve_spectrum_pixel_width
import multiprocessing as mp
import itertools

#read in one of the .fits files (GJ229)
GJ229_one = fits.open('GJ229_R01_20191118101349_deg0_sp.fits')
spec_data = GJ229_one[1].data
#GJ229_one.info()
#print(np.shape(spec_data))

#read in standard star (93CET for Nov observations)
CET_nov = fits.open('93CET_R01_20191118083912_deg0_sp.fits')
spec_data_CET = CET_nov[1].data
#print(np.shape(spec_data_CET))

#phoenix model for CET (wavelength only)
phoenix_wave = fits.open('WAVE_PHOENIX-ACES-AGSS-COND-2011.fits')
phoenix_wvs = phoenix_wave[0].data/1.e4

crop_phoenix = np.where((phoenix_wvs>1.8-(2.6-1.8)/2)*(phoenix_wvs<2.6+(2.6-1.8)/2))
phoenix_wvs = phoenix_wvs[crop_phoenix]*1.e3

phoenix_model = fits.open('93CET_phoenix_model_11000K_4logg.fits')
phoenix_data = phoenix_model[0].data[crop_phoenix]

#phoenix model for GJ229
phoenix_model_GJ229 = fits.open('lte03700-5.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits')
phoenix_data_GJ229 = phoenix_model_GJ229[0].data[crop_phoenix]

#raw flux vs wavelength of order 33
wave = spec_data[0][33]
flux = spec_data[1][33]

wave_CET=spec_data_CET[0][33]
flux_CET=spec_data_CET[1][33]

#plot everything to check 
plt.xlim([1586.3, 1601.9])
plt.ylim([-2,2])

# plt.plot(wave, flux, marker='o', color='royalblue', markersize=.1)
# plt.plot(wave_CET, flux_CET, marker='o', color='pink', markersize=.1)
# plt.plot(phoenix_wvs, phoenix_data/1.e11, marker='o', color='green', markersize=.1)

# plt.show()

#print(phoenix_wvs)
#print(wave_CET)

line_widths_parvi=np.full((1, len(phoenix_wvs)), .01296)
line_widths_parvi=line_widths_parvi[0]

phoenix_data_int=scipy.interpolate.interp1d(phoenix_wvs, phoenix_data, kind='linear')
flux_CET_int=scipy.interpolate.interp1d(wave_CET, flux_CET, kind='linear')
spectral_data_int=scipy.interpolate.interp1d(wave, flux, kind='linear')
phoenix_data_GJ229_int=scipy.interpolate.interp1d(phoenix_wvs, phoenix_data_GJ229)

xnew=np.arange(1586.2, 1602, 0.01)
phoenix_data_func=phoenix_data_int(xnew)
flux_CET_func=flux_CET_int(xnew)
spectral_flux_func=spectral_data_int(xnew)
phoenix_data_GJ229_func=phoenix_data_GJ229_int(xnew)

#function to broaden spectral line widths due to instrument transmission
def _task_convolve_spectrum_line_width(paras):
    indices,wvs,spectrum,line_widths = paras

    conv_spectrum = np.zeros(np.size(indices))
    dwvs = wvs[1::]-wvs[0:(np.size(wvs)-1)]
    dwvs = np.append(dwvs,dwvs[-1])
    for l,k in enumerate(indices):
        pwv = wvs[k]
        sig = line_widths[k]
        w = int(np.round(sig/dwvs[k]*10.))
        stamp_spec = spectrum[np.max([0,k-w]):np.min([np.size(spectrum),k+w])]
        stamp_wvs = wvs[np.max([0,k-w]):np.min([np.size(wvs),k+w])]
        stamp_dwvs = stamp_wvs[1::]-stamp_wvs[0:(np.size(stamp_spec)-1)]
        gausskernel = 1/(np.sqrt(2*np.pi)*sig)*np.exp(-0.5*(stamp_wvs-pwv)**2/sig**2)
        conv_spectrum[l] = np.sum(gausskernel[1::]*stamp_spec[1::]*stamp_dwvs)
    return conv_spectrum

def convolve_spectrum_line_width(wvs,spectrum,line_widths,mypool=None):
    if mypool is None:
        return _task_convolve_spectrum_line_width((np.arange(np.size(spectrum)).astype(np.int),wvs,spectrum,line_widths))
    else:
        conv_spectrum = np.zeros(spectrum.shape)

    chunk_size=100
    N_chunks = np.size(spectrum)//chunk_size
    indices_list = []
    for k in range(N_chunks-1):
        indices_list.append(np.arange(k*chunk_size,(k+1)*chunk_size).astype(np.int))
    indices_list.append(np.arange((N_chunks-1)*chunk_size,np.size(spectrum)).astype(np.int))
    outputs_list = mypool.map(_task_convolve_spectrum_line_width, zip(indices_list,
                                                            itertools.repeat(wvs),
                                                            itertools.repeat(spectrum),
                                                            itertools.repeat(line_widths)))
    for indices,out in zip(indices_list,outputs_list):
        conv_spectrum[indices] = out

    return conv_spectrum

#saving large data set manipulations as txt files so I dont have to recalculate everytime
# wave_CET = [x if x in wave_CET else 0 for x in phoenix_wavelength]
# wave_CET=np.array(wave_CET)
# print(wave_CET)
# np.savetxt('wave_CET.txt', wave_CET, fmt='%d')

#flatten the shape of the curves
filtered_phoenix_data=scipy.signal.medfilt(phoenix_data_func)
filtered_CET_data=scipy.signal.medfilt(flux_CET_func)
filtered_spectral_data=scipy.signal.medfilt(spectral_flux_func)
filtered_phoenix_data=phoenix_data_func-filtered_phoenix_data
filtered_CET_data=flux_CET_func-filtered_CET_data
filtered_spectral_data=(spectral_flux_func-filtered_spectral_data)/100

#broaden the lines of the steallar model 
#specpool = mp.Pool(processes=4)
convolve_phoenix_data=convolve_spectrum_line_width(xnew, phoenix_data_func, line_widths_parvi, mypool=None)
convolve_phoenix_data_GJ229=convolve_spectrum_line_width(xnew, phoenix_data_GJ229_func, line_widths_parvi, mypool=None)

filtered_phoenix_GJ229_data=scipy.signal.medfilt(convolve_phoenix_data_GJ229)
filtered_phoenix_GJ229_data=(convolve_phoenix_data_GJ229-filtered_phoenix_GJ229_data)/3.e11
#plt.plot(xnew, phoenix_data_func, marker='o', color='royalblue', markersize=.1)
#plt.plot(xnew, convolve_phoenix_data, marker='o', color='orange', markersize=.1)

#plt.show()

#get telluric data
adjusted_phoenix_data=convolve_phoenix_data/1.e11
telluric_amp=flux_CET_func/adjusted_phoenix_data

filtered_telluric_data=scipy.signal.medfilt(telluric_amp)
filtered_telluric_data=(telluric_amp-filtered_telluric_data)*10

model_spectra=filtered_phoenix_GJ229_data*filtered_spectral_data

#plt.plot(wave, flux, marker='o', color='royalblue', markersize=.1)
#plt.plot(xnew, flux_CET_func, marker='o', color='pink', markersize=.1)
#plt.plot(xnew, phoenix_data_func/1.e11, marker='o', color='green', markersize=.1)
plt.plot(xnew, filtered_spectral_data, marker='o', color='blue', markersize=.1)
#plt.plot(xnew, filtered_CET_data, marker='o', color='red', markersize=.1)
#plt.plot(xnew, filtered_phoenix_GJ229_data, marker='o', color='darkgreen', markersize=.1)
#plt.plot(xnew, filtered_telluric_data, marker='o', color='orange', markersize=.1)
plt.plot(xnew, model_spectra, marker='o', color='purple', markersize=.1)

plt.show()