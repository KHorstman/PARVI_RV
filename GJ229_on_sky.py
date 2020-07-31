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
import matplotlib.cm as cm

#read in one of the .fits files (GJ229)
GJ229_nov = fits.open("GJ229_R01_20191118101601_deg0_sp.fits")
#"GJ229_R01_20191118101601_deg0_sp.fits"
spec_data = GJ229_nov[1].data

#set up the script to calculate barycentric motions for earth compared to different stars
iers_conf.iers_auto_url = 'https://datacenter.iers.org/data/9/finals2000A.all'
iers_conf.iers_auto_url_mirror = 'ftp://cddis.gsfc.nasa.gov/pub/products/iers/finals2000A.all'
iers.IERS_Auto.open()  # Note the URL
palomar = EarthLocation.from_geodetic(lat=33.3563*u.deg, lon=116.8650*u.deg, height=1712*u.m)

#sky coordinates for GJ229
#coords=["06:08:28.1354366884 -21:50:35.095535467", "06 08 28.1354366884 -21 50 35.095535467"]
sc = SkyCoord("06 10 34.6152510171 -21 51 52.658021185", unit=(u.hourangle, u.deg))

# #find distance between two objects in the sky (GJ229 and HR1544)
# c1=SkyCoord("06 10 34.6152510171 -21 51 52.658021185", unit=(u.hourangle, u.deg))
# c2=SkyCoord("04 50 36.7229825 +08 54 00.649333", unit=(u.hourangle, u.deg))
# sep = c1.separation(c2)
# print('The separation is')
# print(sep)
# exit()

#change date into correct format
date=GJ229_nov[0].header["DIRNAME"]
date= date[0:4]+"-"+date[4:6]+"-"+date[6:8]+"T"+date[8:10]+":"+date[10:12]+":"+date[12:14]
date_JD=Time(date, format='isot', scale='utc')
date_JD=date_JD.mjd
#print(date_JD)

barycorr = sc.radial_velocity_correction(obstime=Time(str(date), format='isot', scale="utc"), location=palomar)
barycorr = barycorr.to(u.km/u.s).value
#print(barycorr)

#read in standard star (93CET for Nov observations)
CET_nov = fits.open('93CET_R01_20191118083912_deg0_sp.fits')
spec_data_CET = CET_nov[1].data

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

#raw flux vs wavelength for aribitrary order 
wave = spec_data[0][35]
flux = spec_data[1][35]
noise = spec_data[2][35]

for order in range(0,43):
    plt.plot(spec_data[0][order], spec_data[1][order], label='Order'+f"{[order]}")
plt.legend()
plt.show()

wave_CET=spec_data_CET[0][35]
flux_CET=spec_data_CET[1][35]

#print(flux)

#get rid of nans in the noise array 
noise[np.where(np.isnan(noise))] = np.nanmedian(noise)

#define the line widths based on the pixel/wavelength conversion of the instrument
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

#broaden the lines of the steallar model 
#specpool = mp.Pool(processes=4)
convolve_phoenix_data_CET=convolve_spectrum_line_width(phoenix_wvs, phoenix_data, line_widths_parvi, mypool=None)
#specpool.join()
convolve_phoenix_data_GJ229=convolve_spectrum_line_width(phoenix_wvs, phoenix_data_GJ229, line_widths_parvi, mypool=None)

#get telluric data
#define new wavelength range based on science (ie the wavelength range and step size of the standard star obs)
phoenix_data_CET_int=scipy.interpolate.interp1d(phoenix_wvs, convolve_phoenix_data_CET, kind='linear', bounds_error=False, fill_value=1)
phoenix_data_CET_func=phoenix_data_CET_int(wave_CET)
#flux_CET[np.where(np.isnan(flux_CET))] = np.nanmedian(flux_CET)
telluric_amp=(flux_CET/phoenix_data_CET_func)
#print(telluric_amp)
#telluric_nans=np.where(np.isfinite(telluric_amp))
transmission=telluric_amp
#print(transmission)
# plt.plot(wave_CET[telluric_nans], transmission/np.std(transmission), alpha=1, marker='o', color='royalblue', markersize=1, label='model')
# plt.show()
# transmission_spline = scipy.interpolate.splrep(wave_CET[telluric_nans], transmission)
# plt.plot(wave_CET, scipy.interpolate.splev(wave, transmission_spline))
# plt.show()

#filter on sky data
#filtered_spectral_data=scipy.signal.medfilt(flux, 501)
filtered_spectral_data=np.nanmean(flux)
filtered_spectral_data=flux-filtered_spectral_data
#filtered_spectral_data[np.where(np.isnan(filtered_spectral_data))] = np.nanmedian(filtered_spectral_data)
nans=np.where(np.isfinite(filtered_spectral_data))
filtered_spectral_data=filtered_spectral_data[nans]

#forward modeling approach

c_kms=2.99792e5

#crop_data = np.where((filtered_spectral_data>1622) & (filtered_spectral_data<1640))

wvs = wave
data = filtered_spectral_data

#print(wvs, data)

#model_spline
model_spline=scipy.interpolate.splrep(phoenix_wvs, convolve_phoenix_data_GJ229)

#radial velocity range
rv=np.array(np.arange(0, 20, .1))

minus2logL_out=np.array([])
logpost_out=np.array([])

# #see if the tellurics are shifted
# telluric_files = ['palomar_500.txt','palomar_1000.txt', 'palomar_5000.txt', 'palomar_10000.txt', 'palomar_20000.txt']
# telluric_data = [np.loadtxt(open(file), usecols=(1,2)) for file in telluric_files]
# telluric_wavelength=telluric_data[0][:,0]*1.e3
# #print(telluric_wavelength)
# telluric_500_counts=telluric_data[0][:,1]
# telluric_1000_counts=telluric_data[1][:,1]
# telluric_5000_counts=telluric_data[2][:,1]
# telluric_10000_counts=telluric_data[3][:,1]
# telluric_20000_counts=telluric_data[4][:,1]
# crop_telluric=np.where((telluric_wavelength>1400) & (telluric_wavelength<1800))
# telluric_wavelength=telluric_wavelength[crop_telluric]
# telluric_500_counts=telluric_500_counts[crop_telluric]

for i in range(len(rv)):
    #The parameter we are after is rv
    wvs_shifted= wvs*(1 - ((rv[i]-barycorr) / c_kms))
    model = scipy.interpolate.splev(wvs_shifted, model_spline, der=0)
    #transmission = scipy.interpolate.splev(wvs, transmission_spline, der=0)
    # plt.plot(wvs, transmission)
    # plt.show()
    model= model*transmission 
    #print(model)
    #filter model
    #telluric_nans=np.where(np.isfinite(transmission))
    model=model[nans]
    #model_filter=scipy.signal.medfilt(model, 501)
    model_filter=np.nanmean(model)
    model=model-model_filter
    #model=model[nans]
    # print(np.isnan(np.sum(model)))
    #transmission_test=scipy.signal.medfilt(transmission[nans], 501)
    transmission_test=np.nanmean(transmission[nans])
    transmission_test=transmission[nans]-transmission_test
    # plt.plot(wvs[nans], transmission_test)
    # plt.show()
    # plt.plot(wvs[nans], (model/np.std(model)), alpha=1, marker='o', color='royalblue', markersize=.5, label='model')
    # plt.plot(wvs[nans], (data/np.std(data)), alpha=.5, marker='o', color='red', markersize=.1, label='data')
    # plt.plot(wvs[nans], (transmission_test/np.std(transmission_test))+5, alpha=.5, marker='o', color='green', markersize=.1, label='transmission from standard star')
    # # #plt.plot(telluric_wavelength, ((telluric_500_counts/np.std(telluric_500_counts))-6)*10, alpha=.5, marker='o', color='orange', markersize=.1, label='simulated transmission (ATRAN)')
    # plt.plot(wvs[nans], (scipy.interpolate.splev(wvs_shifted, model_spline, der=0)[nans])/np.std(scipy.interpolate.splev(wvs_shifted, model_spline, der=0)[nans]), marker='o', color='purple', markersize=.1, label='convolved stellar lines')
    # # #plt.title('ATRAN vs Standard Star Telluric Data Epoch 2')
    # # #plt.xlim([1500, 1520])
    # # #plt.ylim([-15, 1])
    # plt.legend()
    # plt.show()
    # exit()

    Npix = np.size(data)
    sigmas_vec = np.ones(np.size(data))*np.std(noise)
    norm_model = model / sigmas_vec
    norm_data = data / sigmas_vec
    max_like_amplitude = np.sum(norm_data * norm_model) / np.sum((norm_model) ** 2)

    data_model = max_like_amplitude * model
    residuals = data - data_model
    HPFchi2 = np.nansum((residuals / sigmas_vec) ** 2)

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
plt.plot(rv, minus2logL_out, alpha=1, marker='o', color='royalblue', markersize=.1)
plt.xlabel('RV (km/s)')
plt.title('-2logL')
#plt.savefig('rv_logL_date_02.png', dpi=300)
plt.show()

#plot posterior
posterior = np.exp(logpost_out-np.max(logpost_out))
plt.plot(rv, posterior, alpha=1, marker='o', color='royalblue', markersize=.1)
plt.xlabel('RV (km/s)')
plt.title('Posterior')
#plt.savefig('rv_posterior_date_02.png', dpi=300)
plt.show()

xmax = rv[np.argmax(posterior)]
ymax = posterior.max()
xmin = rv[np.argmin(minus2logL_out)]

print(xmax)
print(xmin)

def get_err_from_posterior(x,posterior):
            ind = np.argsort(posterior)
            cum_posterior = np.zeros(np.shape(posterior))
            cum_posterior[ind] = np.cumsum(posterior[ind])
            cum_posterior = cum_posterior/np.max(cum_posterior)
            argmax_post = np.argmax(cum_posterior)
            if len(x[0:np.min([argmax_post+1,len(x)])]) < 2:
                    lx = x[0]
            else:
                    tmp_cumpost = cum_posterior[0:np.min([argmax_post+1,len(x)])]
                    tmp_x= x[0:np.min([argmax_post+1,len(x)])]
                    deriv_tmp_cumpost = np.insert(tmp_cumpost[1::]-tmp_cumpost[0:np.size(tmp_cumpost)-1],np.size(tmp_cumpost)-1,0)
                    try:
                        whereinflection = np.where(deriv_tmp_cumpost<0)[0][0]
                        where2keep = np.where((tmp_x<=tmp_x[whereinflection])+(tmp_cumpost>=tmp_cumpost[whereinflection]))
                        tmp_cumpost = tmp_cumpost[where2keep]
                        tmp_x = tmp_x[where2keep]
                    except:
                        pass
                    lf = scipy.interpolate.interp1d(tmp_cumpost,tmp_x,bounds_error=False,fill_value=x[0])
                    lx = lf(1-0.6827)
            if len(x[argmax_post::]) < 2:
                    rx=x[-1]
            else:
                    tmp_cumpost = cum_posterior[argmax_post::]
                    tmp_x= x[argmax_post::]
                    deriv_tmp_cumpost = np.insert(tmp_cumpost[1::]-tmp_cumpost[0:np.size(tmp_cumpost)-1],np.size(tmp_cumpost)-1,0)
                    try:
                        whereinflection = np.where(deriv_tmp_cumpost>0)[0][0]
                        where2keep = np.where((tmp_x>=tmp_x[whereinflection])+(tmp_cumpost>=tmp_cumpost[whereinflection]))
                        tmp_cumpost = tmp_cumpost[where2keep]
                        tmp_x = tmp_x[where2keep]
                    except:
                        pass
                    rf = scipy.interpolate.interp1d(tmp_cumpost,tmp_x,bounds_error=False,fill_value=x[-1])
                    rx = rf(1-0.6827)
            return lx,x[argmax_post],rx,argmax_post
        
error_posterior=get_err_from_posterior(rv, posterior)
print(error_posterior)
