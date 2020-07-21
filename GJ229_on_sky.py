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
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
from astropy import units as u
from astropy.utils import iers
from astropy.utils.iers import conf as iers_conf


#read in one of the .fits files (GJ229)
GJ229_one = fits.open('GJ229_R01_20191118101601_deg0_sp.fits')
spec_data = GJ229_one[1].data
hdu=fits.PrimaryHDU()
#print(GJ229_one[0].header)
#print(np.shape(spec_data))

#set up the script to calculate barycentric motions for earth compared to different stars
#print(iers_conf.iers_auto_url)
#default_iers = iers_conf.iers_auto_url
#print(default_iers)
iers_conf.iers_auto_url = 'https://datacenter.iers.org/data/9/finals2000A.all'
iers_conf.iers_auto_url_mirror = 'ftp://cddis.gsfc.nasa.gov/pub/products/iers/finals2000A.all'
iers.IERS_Auto.open()  # Note the URL
palomar = EarthLocation.from_geodetic(lat=33.3563*u.deg, lon=116.8650*u.deg, height=1712*u.m)
sc = SkyCoord(228.6122764979232 * u.deg, -18.4391807524145 * u.deg)

#change date into correct format
date=GJ229_one[0].header["DIRNAME"]
date= date[0:4]+"-"+date[4:6]+"-"+date[6:8]+"T"+date[8:10]+":"+date[10:12]+":"+date[12:14]
print(date)

barycorr = sc.radial_velocity_correction(obstime=Time(str(date), format='isot', scale="utc"), location=palomar)
print(barycorr.to(u.km/u.s).value)

#read in standard star (93CET for Nov observations)
CET_nov = fits.open('93CET_R01_20191118083912_deg0_sp.fits')
spec_data_CET = CET_nov[1].data
#print(np.shape(spec_data_CET))

#phoenix model for CET (wavelength only)
phoenix_wave = fits.open('WAVE_PHOENIX-ACES-AGSS-COND-2011.fits')
phoenix_wvs = phoenix_wave[0].data/1.e1

crop_phoenix = np.where((phoenix_wvs>1400) & (phoenix_wvs<1800))
phoenix_wvs = phoenix_wvs[crop_phoenix]
#print(phoenix_wvs)

phoenix_model = fits.open('93CET_phoenix_model_11000K_4logg.fits')
phoenix_data = phoenix_model[0].data[crop_phoenix]

#phoenix model for GJ229
phoenix_model_GJ229 = fits.open('lte03700-5.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits')
phoenix_data_GJ229 = phoenix_model_GJ229[0].data[crop_phoenix]
#plt.plot(phoenix_wvs, phoenix_data_GJ229, marker='o', color='darkgreen', markersize=.1, label="GJ229 model")
#plt.show()

#raw flux vs wavelength of order 33
wave = spec_data[0][35]
flux = spec_data[1][35]
noise = spec_data[2][35]

#print(np.where(np.isfinite(flux)))
# print(flux)
# print(noise)
#print(np.std(flux/1.e10))

wave_CET=spec_data_CET[0][35]
flux_CET=spec_data_CET[1][35]

#plot everything to check 
#plt.xlim([1640, 1656])
#plt.ylim([0,2])

#plt.plot(noise, flux, marker='o', color='royalblue', markersize=.1)
#plt.plot(wave_CET, flux_CET, marker='o', color='pink', markersize=.1)
# plt.plot(phoenix_wvs, phoenix_data/1.e11, marker='o', color='green', markersize=.1)

#plt.show()

line_widths_parvi=np.full((1, len(phoenix_wvs)), .01296)
line_widths_parvi=line_widths_parvi[0]

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
    #chunk_size=np.size(spectrum)//N_chunks
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
# filtered_phoenix_data=scipy.signal.medfilt(phoenix_data_func, 101)
# filtered_CET_data=scipy.signal.medfilt(flux_CET_func, 101)
# filtered_spectral_data=scipy.signal.medfilt(spectral_flux_func, 101)
# filtered_phoenix_data=phoenix_data_func-filtered_phoenix_data
# filtered_CET_data=flux_CET_func-filtered_CET_data
# filtered_spectral_data=spectral_flux_func-filtered_spectral_data

#broaden the lines of the steallar model 
#specpool = mp.Pool(processes=4)
convolve_phoenix_data_CET=convolve_spectrum_line_width(phoenix_wvs, phoenix_data, line_widths_parvi, mypool=None)
#specpool.join()
convolve_phoenix_data_GJ229=convolve_spectrum_line_width(phoenix_wvs, phoenix_data_GJ229, line_widths_parvi, mypool=None)

#get telluric data
phoenix_data_CET_int=scipy.interpolate.interp1d(phoenix_wvs, convolve_phoenix_data_CET, kind='linear', bounds_error=False, fill_value=1)
#CET_int=scipy.interpolate.interp1d(wave_CET, flux_CET, kind='linear', bounds_error=False, fill_value=1)

#wvs_new=np.arange(wave[0], wave[2047], 0.001) #interpolate based on the science 
phoenix_data_CET_func=phoenix_data_CET_int(wave_CET)
#CET_func=CET_int(wvs_new)

#new_vector = np.array(pd.DataFrame(flux_CET).interpolate(method="linear").fillna(method="bfill").fillna(method="ffill"))
flux_CET[np.where(np.isnan(flux_CET))] = np.nanmedian(flux_CET)
telluric_amp=(flux_CET/phoenix_data_CET_func)
transmission=telluric_amp
transmission_spline = scipy.interpolate.splrep(wave_CET, transmission)

#filter on sky data
# print(flux)
# plt.plot(wave, flux)
# plt.show()
filtered_spectral_data=scipy.signal.medfilt(flux, 101)
filtered_spectral_data=flux-filtered_spectral_data
nans=np.where(np.isfinite(filtered_spectral_data))
filtered_spectral_data=filtered_spectral_data[nans]

#plt.plot(wave, filtered_spectral_data, marker='o', color='orange', markersize=.1)
#plt.plot(wvs_new, telluric_amp/np.std(telluric_amp), marker='o', color='pink', markersize=.1)

#plt.show()

# phoenix_data_int=scipy.interpolate.interp1d(phoenix_wvs, phoenix_data, kind='linear')
# flux_CET_int=scipy.interpolate.interp1d(wave_CET, flux_CET, kind='linear')
# spectral_data_int=scipy.interpolate.interp1d(wave, flux, kind='linear') #do not need to interpolate
# phoenix_data_GJ229_int=scipy.interpolate.interp1d(phoenix_wvs, phoenix_data_GJ229)

# xnew=np.arange(wave[100], wave[2000], 0.001) #interpolate based on the science 
# xnew_2=np.arange(1000, 2000, 0.001)
# phoenix_data_func=phoenix_data_int(xnew)
# flux_CET_func=flux_CET_int(xnew)
# spectral_flux_func=spectral_data_int(xnew)
# phoenix_data_GJ229_func=phoenix_data_GJ229_int(xnew)

# filtered_phoenix_GJ229_data=scipy.signal.medfilt(convolve_phoenix_data_GJ229, 101)
# filtered_phoenix_GJ229_data=convolve_phoenix_data_GJ229/filtered_phoenix_GJ229_data
# #plt.plot(xnew, phoenix_data_func, marker='o', color='royalblue', markersize=.1)
# #plt.plot(xnew, convolve_phoenix_data, marker='o', color='orange', markersize=.1)

# #plt.show()

# # convolve_telluric_data=convolve_spectrum_line_width(xnew, telluric_amp, line_widths_parvi, mypool=None)

# filtered_telluric_data=scipy.signal.medfilt(telluric_amp, 101)
# filtered_telluric_data=(telluric_amp-filtered_telluric_data)

# model_spectra=convolve_phoenix_data_GJ229*telluric_amp
# filtered_model_data=scipy.signal.medfilt(model_spectra, 101)
# filtered_model_data=(model_spectra-filtered_model_data)

# #plt.plot(wave, flux, marker='o', color='royalblue', markersize=.1)
# #plt.plot(xnew, flux_CET_func, marker='o', color='pink', markersize=.1)
# #plt.plot(xnew, filtered_phoenix_data/np.std(filtered_phoenix_data), marker='o', color='pink', markersize=.1, label='phoenix model A0')
# # plt.plot(xnew, filtered_spectral_data/np.std(filtered_spectral_data), marker='o', color='blue', markersize=.1, label='on sky data', alpha=.5)
# # #plt.plot(xnew, filtered_CET_data/np.std(filtered_CET_data), marker='o', color='red', markersize=.1, label='AO')
# # plt.plot(xnew, filtered_telluric_data/np.std(filtered_telluric_data), marker='o', color='green', markersize=.1, label='telluric data', alpha=.5)
# # plt.plot(xnew, filtered_model_data/np.std(filtered_model_data), marker='o', color='red', markersize=.1, label='model', alpha=.5)

# # plt.legend()

# # plt.show()

#forward modeling approach

c_ms=3*10**8
c_kms=3*10**5

#crop wavelength to avoid small amplitude
#crop_wavelength = np.where((wvs_new>wave[110]) & (wvs_new<wave[1990]))
#print(xnew[crop_wavelength])

#spectra= spectral_flux_func/telluric_amp
GJ229_rv=np.full((1, len(wave)), 0)
GJ229_baryrv=np.full((1, len(wave)), 0)

wvs = wave  #wvs_new
noise=np.random.normal(10, 1, len(wave))
data = filtered_spectral_data
#print(data)
#print(data[crop_wavelength])
#signal=filtered_model_data
# #RV=GJ229_rv+GJ229_baryrv
# #wvs_signal=xnew*(1-(GJ229_baryrv/c_kms))
# #test=spectral_flux_func/telluric_amp
# #shift=np.full((1, len(xnew)), (1 - ((-45) / c_kms)))
# #print(len(wvs_signal[0]))
# #print(len(signal))

# #print(wvs_signal)
# #plt.plot(xnew, filtered_model_data/np.std(filtered_model_data), marker='o', color='red', markersize=.1, label='model', alpha=.5)
# # plt.plot(xnew, filtered_phoenix_GJ229_data/np.std(filtered_phoenix_GJ229_data), marker='o', color='pink', markersize=.1, label='phoenix model GJ229')
# # plt.plot(xnew, filtered_spectral_data/np.std(filtered_spectral_data), marker='o', color='blue', markersize=.1, label='on sky data', alpha=.5)
# # plt.plot(xnew, filtered_telluric_data/np.std(filtered_telluric_data), marker='o', color='green', markersize=.1, label='telluric data', alpha=.5)
# # plt.legend()
# # plt.show()

#model_spline
#plt.plot(phoenix_wvs, convolve_phoenix_data_CET)
#plt.show()
model_spline=scipy.interpolate.splrep(phoenix_wvs, convolve_phoenix_data_GJ229)
# # plt.plot(wvs, signal, marker='o', color='royalblue', markersize=.1, label='model (from model_spline)', alpha=.5)
# # plt.plot(wvs, filtered_spectral_data, marker='o', color='red', markersize=.1, label='data', alpha=.5)
# # plt.legend()
# # plt.show()

# wvs=wvs[crop_wavelength]
# GJ229_baryrv=GJ229_baryrv[0][crop_wavelength]

#wvs=wvs.tolist()
#signal=signal.tolist()
#RV=RV.tolist()
#GJ229_baryrv=GJ229_baryrv.tolist()
#wvs_signal=wvs_signal.tolist()
#data=data[crop_wavelength]

#radial velocity range but dont know what it should be including barycentric motion (havent gotten rid of it yet)
rv=np.array(np.arange(-20, 0, 1))

minus2logL_out=np.array([])
logpost_out=np.array([])

for i in range(len(rv)):
    #The parameter we are after is rv
    wvs_shifted= wvs*(1 - ((rv[i]) / c_kms))
    model = scipy.interpolate.splev(wvs_shifted, model_spline, der=0)
    transmission = scipy.interpolate.splev(wvs, transmission_spline, der=0)
    model= model*transmission       #[crop_wavelength]
    print(len(model))
    #make model same size array as on sky data
    #print(len(wvs_shifted[0]))
    #print(len(model[0]))
    #model=scipy.interpolate.interp1d(wvs_shifted, model, kind='linear', bounds_error=False, fill_value=1)
    #new_range=np.array(np.arange(wave[0], wave[2047], .00804740910))
    #model=model(new_range)
    #filter model
    model_filter=scipy.signal.medfilt(model, 101)
    model=model-model_filter
    model=model[nans]
    #print(len(data))
    # print([i])
    # print(np.isnan(np.sum(data)))
    # print(np.isnan(np.sum(model)))

    # plt.plot(new_range[nans], model*5, alpha=1, marker='o', color='royalblue', markersize=1, label='model')
    # plt.plot(wave[nans], data, alpha=.5, marker='o', color='red', markersize=.1, label='data')
    # plt.legend()
    # plt.show()

    Npix = np.size(data)
    #made noise vector 1 just because I dont know what it is right now
    sigmas_vec = np.ones(np.size(data)) #*np.std(noise)
    norm_model = model / sigmas_vec
    norm_data = data / sigmas_vec
    max_like_amplitude = np.sum(norm_data * norm_model) / np.sum((norm_model) ** 2)
    #print((norm_data*norm_model).tolist())
    #print(np.where(np.isfinite(norm_data*norm_model)))
    #print(np.nansum(norm_data*norm_model))
    #print(np.sum(norm_model)**2)
    #max_like_amplitude_out_01[j].append(max_like_amplitude)
    #print(np.isnan(np.sum(data)))

    data_model = max_like_amplitude * model
    residuals = data - data_model
    HPFchi2 = np.nansum((residuals / sigmas_vec) ** 2)
    #print(HPFchi2)
    #print(Npix)
    #print(np.sum(residuals))

    max_like_amplitude_err = np.sqrt((HPFchi2 / Npix) / np.sum(norm_model ** 2))
    #max_like_amplitude_err_out_01[j].append(max_like_amplitude_err)

    logdet_Sigma = np.sum(2 * np.log(sigmas_vec))
    minus2logL = Npix * (1 + np.log(HPFchi2 / Npix) + logdet_Sigma + np.log(2 * np.pi))
    #print(HPFchi2)
    minus2logL_out=np.append(minus2logL_out, minus2logL)

    slogdet_icovphi0 = np.log(1 / np.sum((norm_data) ** 2))
    logpost = -0.5 * logdet_Sigma - 0.5 * slogdet_icovphi0 - (Npix - 1 + 2 - 1) / (2) * np.log(HPFchi2)
    logpost_out=np.append(logpost_out, logpost)

#plot minus2logL as function of the rv
#print(rv)
#print(minus2logL_out)
#print(max_like_amplitude_out)
#print(max_like_amplitude_err_out)
#print(minus2logL_out)
plt.plot(rv, minus2logL_out, alpha=1, marker='o', color='royalblue', markersize=.1)
plt.xlabel('RV (km/s')
plt.title('-2logL')
#plt.savefig('rv_logL_date_02.png', dpi=300)
plt.show()

#plot posterior
posterior = np.exp(logpost_out-np.max(logpost_out))
plt.plot(rv, posterior, alpha=1, marker='o', color='royalblue', markersize=.1)
plt.xlabel('RV (km/s')
plt.title('Posterior')
#plt.savefig('rv_posterior_date_02.png', dpi=300)
plt.show()

xmax = rv[np.argmax(posterior)]
ymax = posterior.max()
xmin = rv[np.argmin(minus2logL_out)]

print(xmax)
print(xmin)

#time is in unix (UTC)/a billion