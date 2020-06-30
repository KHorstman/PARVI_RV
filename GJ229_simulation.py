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
from scipy import optimize


#read in 16 files (simulated data for GJ229 with injected planet)
def make_list(n):
      for _ in range(n):
            yield np.array([])

date_01_counts, date_02_counts, date_03_counts, date_04_counts, date_05_counts, date_06_counts, date_07_counts, date_08_counts, date_09_counts, date_10_counts, date_11_counts, date_12_counts, date_13_counts, date_14_counts, date_15_counts, date_16_counts= make_list(16)

file_name = ['SimGJ229idate01Hmag8RV10P10d.txt', 'SimGJ229idate02Hmag8RV10P10d.txt', 'SimGJ229idate03Hmag8RV10P10d.txt', 'SimGJ229idate04Hmag8RV10P10d.txt', 'SimGJ229idate05Hmag8RV10P10d.txt', 'SimGJ229idate06Hmag8RV10P10d.txt', 'SimGJ229idate07Hmag8RV10P10d.txt', 'SimGJ229idate08Hmag8RV10P10d.txt', 'SimGJ229idate09Hmag8RV10P10d.txt', 'SimGJ229idate10Hmag8RV10P10d.txt', 'SimGJ229idate11Hmag8RV10P10d.txt', 'SimGJ229idate12Hmag8RV10P10d.txt', 'SimGJ229idate13Hmag8RV10P10d.txt', 'SimGJ229idate14Hmag8RV10P10d.txt', 'SimGJ229idate15Hmag8RV10P10d.txt', 'SimGJ229idate16Hmag8RV10P10d.txt']
#file_name = ['SimGJ229idate01Hmag8RV10P10d.txt', 'SimGJ229idate02Hmag8RV10P10d.txt']
date_name = ['date_01', 'date_02', 'date_03', 'date_04', 'date_05', 'date_06', 'date_07', 'date_08', 'date_09', 'date_10', 'date_11', 'date_12', 'date_13', 'date_14', 'date_15', 'date_16']
#date_name_counts = [date_01_counts, date_02_counts, date_03_counts, date_04_counts, date_05_counts, date_06_counts, date_07_counts, date_08_counts, date_09_counts, date_10_counts, date_11_counts, date_12_counts, date_13_counts, date_14_counts, date_15_counts, date_16_counts]

# for i in range(16):
#       date_name[i]=np.loadtxt(file_name[2], usecols=(3,4))
#       date_name_wavelength[i]=np.append(date_name_wavelength[i], date_name[i][:,0])
#       print(date_name_wavelength[i])

#load in files
data = [np.loadtxt(open(file), usecols=(0,1,2,3,4,5,6)) for file in file_name]
wavelength=data[0][:,3]

#define the count data
# date_01_counts=data[0][:,4]
# date_02_counts=data[1][:,4]
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

#define a list consiting of empty arrays
counts=[np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([])]
bary=[np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([])]
planet_rv=[np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([])]
noise=[np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([])]
transmission=[np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([])]
date=[np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([])]

#date=[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]

# for i in range(len(counts)):
#       counts[i].append(data[i][:,4])
#       bary[i].append(data[i][:,1])
#       planet_rv[i].append(data[i][:,2])
#       noise[i].append(data[i][:,5])
#       transmission[i].append(data[i][:,6])
#       date[i].append(data[i][:,0])

for i in range(len(counts)):
      counts[i]=np.append(counts[i],data[i][:,4])
      bary[i]=np.append(bary[i], data[i][:,1])
      planet_rv[i]=np.append(planet_rv[i], data[i][:,2])
      noise[i]=np.append(data[i][:,5], noise[i])
      transmission[i]=np.append(data[i][:,6], transmission[i])
      date[i]=np.append(data[i][:,0], date[i]) 

#print(counts)

#change list back into array format for data processing
counts=np.array(counts)
bary=np.array(bary)
planet_rv=np.array(planet_rv)
noise=np.array(noise)
transmission=np.array(transmission)
date=np.array(date)


# #add other date for date_01 for forward modeling approach
# date_01_bary=data[0][:,1]
# date_01_planet_rv= data[0][:,2]
# date_01_noise=data[0][:,5]
# date_01_trans=data[0][:,6]

# #use date_02 instead because injected RV for planet is 0 for date_01
# date_02_bary=data[1][:,1]
# date_02_planet_rv= data[1][:,2]
# date_02_noise=data[1][:,5]
# date_02_trans=data[1][:,6]

#total_counts=[date_01_counts, date_02_counts, date_03_counts, date_04_counts, date_05_counts, date_06_counts, date_07_counts, date_08_counts, date_09_counts, date_10_counts, date_11_counts, date_12_counts, date_13_counts, date_14_counts, date_15_counts, date_16_counts]

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
# plt.xlim([1.66, 1.68])
# plt.ylim([0,1])

# #wavelength data in angstroms
# #normalized phoenix data
# filtered_data=scipy.signal.medfilt(phoenix_data)
# phoenix_data_norm=((phoenix_data-filtered_data)/(2*10**12))+.85
# #phoenix_data_norm=phoenix_data/np.linalg.norm(phoenix_data)

# #normalized telluric data
# telluric_500_counts_norm=telluric_500_counts

# #normalized simulated data
# date_01_counts_norm=date_01_counts

# plt.plot(telluric_wavelength, telluric_500_counts_norm, alpha=1, marker='o', color='royalblue', markersize=.1)
# plt.plot(phoenix_wave_data/10000, phoenix_data_norm, alpha=.5, marker='o', color='red', markersize=.1)
# #plt.plot(phoenix_wave_data/10000, filtered_data, alpha=.5, marker='o', color='blue', markersize=.1)
# plt.plot(wavelength, date_01_counts_norm, alpha=1, marker='o', color='green', markersize=.1)

# #legend
# red_patch= mpatches.Patch(color='red', label='Phoenix Model Spectra')
# pink_patch= mpatches.Patch(color='royalblue', label='Telluric Model Spectra')
# orange_patch= mpatches.Patch(color='green', label='Simulated Spectra')
# plt.legend(handles=[red_patch, pink_patch, orange_patch], loc=1)

# plt.savefig('GJ229_models_and_simulated_spectra_large.pdf')
# plt.savefig('GJ229_models_and_simulated_spectra_large.png', dpi=300)

# plt.show()

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

#forward modeling approach
#plt.xlim([1.50, 1.80])

c_ms=3*10**8
c_kms=3*10**5

wvs = wavelength
#data = np.array([date_01_counts, date_02_counts])
data = counts
noise = noise
transmission = transmission
signal = (data - noise)/transmission
#wvs_signal = wvs*None #correct the wvs vector for the RV of the star (incl. both the barycentric RV and the planet induced RV)
#RV=np.array([(date_01_planet_rv/1000+date_01_bary), (date_02_planet_rv/1000+date_02_bary)])
RV=(planet_rv/1000)+bary
wvs_signal= wvs*(1+(RV/c_kms))

hehe=str(signal[0])[1:-1]
hoho=str(wvs_signal[0])[1:-1]


wvs=wvs.tolist()
data=data.tolist()
noise=noise.tolist()
transmission=transmission.tolist()
#signal=signal.tolist()
RV=RV.tolist()
#wvs_signal=wvs_signal.tolist()

# model_spline = # spline inteprolation using (wvs_signal,signal)
# model_spline=[[],[]]
# for i in range(len(model_spline)):
#       model_spline[i].append(scipy.interpolate.splrep(wvs_signal[i], signal[i]))
#model_spline_test = scipy.interpolate.splrep(wvs_signal[0], signal[0])
#model_spline=str(model_spline)[1:-1]

#bootleg model_spline for every file --- will try to fix this using a for loop later
model_spline=[scipy.interpolate.splrep(wvs_signal[0], signal[0]), scipy.interpolate.splrep(wvs_signal[1], signal[1]), scipy.interpolate.splrep(wvs_signal[2], signal[2]), scipy.interpolate.splrep(wvs_signal[3], signal[3]), scipy.interpolate.splrep(wvs_signal[4], signal[4]), scipy.interpolate.splrep(wvs_signal[5], signal[5]), scipy.interpolate.splrep(wvs_signal[6], signal[6]), scipy.interpolate.splrep(wvs_signal[7], signal[7]), scipy.interpolate.splrep(wvs_signal[8], signal[8]), scipy.interpolate.splrep(wvs_signal[9], signal[9]), scipy.interpolate.splrep(wvs_signal[10], signal[10]), scipy.interpolate.splrep(wvs_signal[11], signal[11]), scipy.interpolate.splrep(wvs_signal[12], signal[12]), scipy.interpolate.splrep(wvs_signal[13], signal[13]), scipy.interpolate.splrep(wvs_signal[14], signal[14]), scipy.interpolate.splrep(wvs_signal[15], signal[15])]

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
science_baryrv=bary
#print(science_baryrv)

#keep rv in km/s
#should recover 6 m/s from model for injected planet RV 
rv=np.array(np.arange(-.015, .015, .0001))
#print(rv[0])

#print(np.size(wvs))
#print(np.size(model_spline(xs)))


# for loop over rv
      # store:
      # max_like_amplitude
      # max_like_amplitude_err
      # minus2logL
      # logpost

max_like_amplitude_out_01=[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
max_like_amplitude_err_out_01=[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
minus2logL_out_01=[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
logpost_out_01=[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
error_posterior=[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]

rv_samples=[]

# - science_baryrv)
for j in range(16):
      for i in range(len(rv)):
            #The parameter we are after is rv
            wvs_shifted= wvs*(1 + ((rv[i]+science_baryrv[j]) / c_kms))
            model = scipy.interpolate.splev(wvs_shifted, model_spline[j], der=0) * transmission[j]
            #plt.plot(wvs_shifted, model, alpha=.5, marker='o', color='royalblue', markersize=.1)
            #plt.plot(wvs_signal, signal, alpha=.5, marker='o', color='red', markersize=.1)
            #plt.show()

            #for now make sigmas_vec=1
            #sigmas_vector=1

            Npix = np.size(data[j])
            sigmas_vec = np.ones(np.size(data[j]))*np.std(noise[j])
            #are norm_model and norm_data the same on purpose?
            norm_model = model / sigmas_vec
            norm_data = model / sigmas_vec
            max_like_amplitude = np.sum(norm_data * norm_model) / np.sum((norm_model) ** 2)
            max_like_amplitude_out_01[j].append(max_like_amplitude)

            data_model = max_like_amplitude * model
            residuals = data[j] - data_model
            HPFchi2 = np.nansum((residuals / sigmas_vec) ** 2)

            max_like_amplitude_err = np.sqrt((HPFchi2 / Npix) / np.sum(norm_model ** 2))
            max_like_amplitude_err_out_01[j].append(max_like_amplitude_err)

            logdet_Sigma = np.sum(2 * np.log(sigmas_vec))
            minus2logL = Npix * (1 + np.log(HPFchi2 / Npix) + logdet_Sigma + np.log(2 * np.pi))
            minus2logL_out_01[j].append(minus2logL)

            slogdet_icovphi0 = np.log(1 / np.sum((norm_data) ** 2))
            logpost = -0.5 * logdet_Sigma - 0.5 * slogdet_icovphi0 - (Npix - 1 + 2 - 1) / (2) * np.log(HPFchi2)
            logpost_out_01[j].append(logpost)

      #plot minus2logL as function of the rv
      #print(rv)
      #print(minus2logL_out)
      #print(max_like_amplitude_out)
      #print(max_like_amplitude_err_out)
      #plt.plot(rv*1000, minus2logL_out_01[j], alpha=1, marker='o', color='pink', markersize=.1)
      #plt.savefig('rv_logL_date_01.png', dpi=300)
      #plt.show()

      #plot posterior
      posterior = np.exp(logpost_out_01[j]-np.max(logpost_out_01[j]))
      #plt.plot(rv*1000, posterior[j], alpha=1, marker='o', color='royalblue', markersize=.1)
      #plt.savefig('rv_posterior_date_01.png', dpi=300)

      xmax = rv[np.argmax(posterior)]
      ymax = posterior.max()

      #print(xmax*1000)

      rv_samples.append(xmax*1000)

      #find error bars for each rv
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
      
      error_posterior[j].append(get_err_from_posterior(rv, posterior))

print(rv_samples[0])
print(error_posterior[0][0][2])

min_error_posterior=[]
max_error_posterior=[]

for i in range(len(error_posterior)):
      min_error_posterior.append(error_posterior[i][0][0])
      max_error_posterior.append(error_posterior[i][0][2])

min_error_posterior=np.array(min_error_posterior)*1000
max_error_posterior=np.array(max_error_posterior)*1000

min_error_posterior= abs(rv_samples-min_error_posterior)
max_error_posterior=abs(max_error_posterior-rv_samples)
print(min_error_posterior)
print(max_error_posterior)


#plot most likely RV as a function of time
date_for_rv =[]
rv_from_simulation= []

for i in range(len(date)):
      date_for_rv.append(date[i][1])
      rv_from_simulation.append(planet_rv[i][1])

print(rv_from_simulation)

date_for_rv=np.array(date_for_rv)
date_for_rv=(date_for_rv+50000)

#make a sine function
def sin_func(x, a, b):
    return a * np.sin(b * x)

params, params_covariance = optimize.curve_fit(sin_func, date_for_rv, rv_samples, p0=[10, 10])

plt.figure(figsize=(20,15))

plt.xlabel('Modified Julian Date')
plt.ylabel('Radial Velocity (m/s)', color='black')
plt.plot(date_for_rv, rv_samples, label='RV estimate from likelihood function', marker='o', color='royalblue')
plt.scatter(date_for_rv, rv_from_simulation, label='RV injected', marker='*', color='red', s=30, zorder=4)
plt.errorbar(date_for_rv, rv_samples, yerr=[min_error_posterior, max_error_posterior], capsize=3, fmt='none', c='gray', zorder=1)

plt.legend(loc=1)

plt.savefig('most_likely_rv.pdf')
plt.savefig('most_likely_rv.png', dpi=300)

plt.show()