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
import multiprocessing as mp
import itertools
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
from astropy import units as u
from astropy.utils import iers
from astropy.utils.iers import conf as iers_conf
import matplotlib.cm as cm
from matplotlib.pyplot import figure
from PyAstronomy import pyasl
import wget
from scipy.optimize import lsq_linear
from astropy import stats

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

def save_atrangrid(filelist_atran,line_width,atrangridname,mypool=None):
    #Broaden and save a grid of telluric models using atran.
    filelist_atran.sort() 
    #pull water vapor/angle from the file name 
    water_list = np.array([int(atran_filename.split("_")[-5]) for atran_filename in filelist_atran])
    angle_list = np.array([float(atran_filename.split("_")[-3]) for atran_filename in filelist_atran])
    water_unique = np.unique(water_list)
    angle_unique = np.unique(angle_list)
    #print(water_unique, angle_unique)
    atran_spec_list = []
    for water in water_unique:
        for angle in angle_unique:
            print(water,angle)
            #pulls file with unique water/angle 
            atran_filename = filelist_atran[np.where((water==water_list)*(angle==angle_list))[0][0]]
            atran_arr = np.loadtxt(atran_filename).T
            #pull wavelength soln and flux for each model 
            atran_wvs = atran_arr[1,:]
            atran_spec = atran_arr[2,:]
            #atran_line_widths = np.array(pd.DataFrame(line_width).interpolate(method="linear").fillna(method="bfill").fillna(method="ffill"))[:, 0]
            #convolve line widths based on the transmission of the instrument
            atran_spec_conv = convolve_spectrum_line_width(atran_wvs,atran_spec,line_width,mypool=mypool)
            atran_spec_list.append(atran_spec_conv)
            print('it worked!')
            #atran_spec_list.append(atran_spec)
    atran_grid = np.zeros((np.size(water_unique),np.size(angle_unique),np.size(atran_wvs)))
    for water_id,water in enumerate(water_unique):
        for angle_id,angle in enumerate(angle_unique):
            #associates the right flux appended to spec list with the right water/angle
            atran_grid[water_id,angle_id,:] = atran_spec_list[np.where((water_list==water)*(angle_list==angle))[0][0]]
    hdulist = fits.HDUList()
    hdulist.append(fits.PrimaryHDU(data=atran_grid))
    hdulist.append(fits.ImageHDU(data=water_unique))
    hdulist.append(fits.ImageHDU(data=angle_unique))
    hdulist.append(fits.ImageHDU(data=atran_wvs))
    #print(hdulist)
    # try:
    hdulist.writeto(atrangridname, overwrite=True)
    # except TypeError:
    #     hdulist.writeto(atrangridname, clobber=True)
    hdulist.close()

def save_phoenixgrid(filelist_phoenix,line_width,phoenixgridname,mypool=None):
    #Broaden and save a grid of phoenix models
    filelist_phoenix.sort() 
    #pull water vapor/angle from the file name 
    teff_list = np.array([str(phoenix_filename.split("-")[0][-4:]) for phoenix_filename in filelist_phoenix])
    teff_list = teff_list.astype(np.int)
    logg_list = np.array([float(phoenix_filename.split("-")[1]) for phoenix_filename in filelist_phoenix])
    teff_unique = np.unique(teff_list)
    print(teff_unique)
    logg_unique = np.unique(logg_list)
    phoenix_spec_list = []
    for teff in teff_unique:
        for logg in logg_unique:
            print(teff,logg)
            #pulls file with unique water/angle 
            #phoenix_filename = filelist_phoenix[np.where((teff==teff_list)*(logg==logg_list))[0][0]]
            phoenix_filename = fits.open("lte0"+f"{teff}"+"-"+f"{logg}"+"0-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits")
            #phoenix_arr = np.loadtxt(phoenix_filename).T
            phoenix_wave = fits.open('WAVE_PHOENIX-ACES-AGSS-COND-2011.fits')
            phoenix_wvs = phoenix_wave[0].data/1.e1
            #pull flux for each model 
            phoenix_spec = phoenix_filename[0].data
            crop_phoenix = np.where((phoenix_wvs>1400) & (phoenix_wvs<1800))
            phoenix_wvs=phoenix_wvs[crop_phoenix]
            phoenix_spec=phoenix_spec[crop_phoenix]
            print(phoenix_wvs)
            print(phoenix_spec)
            #convolve line widths based on the transmission of the instrument
            phoenix_spec_conv = convolve_spectrum_line_width(phoenix_wvs,phoenix_spec,line_width,mypool=mypool)
            phoenix_spec_list.append(phoenix_spec_conv)
            print('it worked!')
    phoenix_grid = np.zeros((np.size(teff_unique),np.size(logg_unique),np.size(phoenix_wvs)))
    for teff_id,teff in enumerate(teff_unique):
        for logg_id,logg in enumerate(logg_unique):
            #associates the right flux appended to spec list with the right teff/logg
            phoenix_grid[teff_id,logg_id,:] = phoenix_spec_list[np.where((teff_list==teff)*(logg_list==logg))[0][0]]
    hdulist = fits.HDUList()
    hdulist.append(fits.PrimaryHDU(data=phoenix_grid))
    hdulist.append(fits.ImageHDU(data=teff_unique))
    hdulist.append(fits.ImageHDU(data=logg_unique))
    hdulist.append(fits.ImageHDU(data=phoenix_wvs))
    #print(hdulist)
    # try:
    hdulist.writeto(phoenixgridname, overwrite=True)
    # except TypeError:
    #     hdulist.writeto(atrangridname, clobber=True)
    hdulist.close()
    
# def blaze_func_parameters(x, model, wvs):
#     blaze=[]
#     for i in range(x):
#         blaze.append(np.array(model*wvs**i))
#     return blaze

# def blaze_initial_model(lam,lam_long,lam_short):
#     c1=0.5445
#     c2=0.1611
#     const=c1-(c1-c2)/(lam_long-lam_short)*(lam_long-lam)
#     #dlam=(lam_short-lam_long)/2
#     lam0=(lam_short+lam_long)/2
#     a=-1*(-dlam**2 +lam0**2-const*lam0**2)/dlam**2
#     b=-2*(-lam0+const*lam0)/dlam**2
#     c=-(1-const)/dlam**2
#     return (a+b*lam+c*lam**2)