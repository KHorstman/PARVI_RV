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
import multiprocessing as mp
import itertools
import astropy
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
from astropy import units as u
from astropy.utils import iers
from astropy.utils.iers import conf as iers_conf
from astropy.time import Time
from astropy import stats
from PyAstronomy import pyasl
import time
import math
from function_library import convolve_spectrum_line_width, save_atrangrid, save_phoenixgrid, cost_func
from matplotlib import gridspec
from scipy import stats

#RV estimate for all orders in one file

#wrap code in if __name__ == '__main__': to use multiple cores at the same time 
if __name__ == '__main__':
    #open the fits file with the one dimensional stellar spectra
    GJ229_nov = fits.open("GJ229_R01_20191118101601_deg0_sp.fits")
    spec_data = GJ229_nov[1].data

    #phoenix model (wavelength only)
    phoenix_wave = fits.open('WAVE_PHOENIX-ACES-AGSS-COND-2011.fits')
    ph_wvs = phoenix_wave[0].data/1.e1

    #crop the model for only the wavelengths needed
    crop_phoenix = np.where((ph_wvs>1400) & (ph_wvs<1800))
    ph_wvs = ph_wvs[crop_phoenix]
    #read in model for GJ229
    phoenix_model = fits.open("lte04200-5.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits")
    phoenix_data = phoenix_model[0].data[crop_phoenix]

    #convolve phoenix spectrum with the line spread function of the instrument
    line_widths_parvi=np.full((1, len(ph_wvs)), .01296)
    line_widths_parvi=line_widths_parvi[0]
    convolve_phoenix_data_GJ229=convolve_spectrum_line_width(ph_wvs, phoenix_data, line_widths_parvi, mypool=None)

    #create the phoenix model to be used in the forward modeling code (using a spline so the wavelength array can be changed for different rv values)
    model_spline=scipy.interpolate.splrep(ph_wvs, convolve_phoenix_data_GJ229)

    #set up the script to calculate barycentric motions for earth
    iers_conf.iers_auto_url = 'https://datacenter.iers.org/data/9/finals2000A.all'
    iers_conf.iers_auto_url_mirror = 'ftp://cddis.gsfc.nasa.gov/pub/products/iers/finals2000A.all'
    iers.IERS_Auto.open()  # Note the URL
    palomar = EarthLocation.from_geodetic(lat=33.3563*u.deg, lon=116.8650*u.deg, height=1712*u.m)
    #sky coordinates for GJ229
    sc = SkyCoord("06 10 34.6152510171 -21 51 52.658021185", unit=(u.hourangle, u.deg))
    #change date into correct format
    date=GJ229_nov[0].header["DIRNAME"]
    date= date[0:4]+"-"+date[4:6]+"-"+date[6:8]+"T"+date[8:10]+":"+date[10:12]+":"+date[12:14]
    date_JD=Time(date, format='isot', scale='utc')
    date_JD=date_JD.mjd
    #print(date_JD)

    #barycentric correction based on time
    barycorr = sc.radial_velocity_correction(obstime=Time(str(date), format='isot', scale="utc"), location=palomar)
    #barcentric correction for GJ229 for the date associated with the file 
    barycorr = barycorr.to(u.km/u.s).value

    # #make a grid of ATRAN models 
    # #only need to do this once
    # specpool = mp.Pool(processes=4)
    # file_list_atran=[]
    # for degree in [0, 22.5, 45, 67.5, 89]:
    #     for water in [0, 500, 1000, 5000, 10000, 20000]:
    #         #I saved the files based on the ATRAN model inputs
    #         #right now fixed values (5617, 30, 2, 1.4, 1.85) have to do with Palomar/wavelength range
    #         #saved in order that the parameters appear on the ATRAN website
    #         name="atran_5617_30_"+f"{water}"+"_2_"+f"{degree}"+"_1.4_1.85.txt"
    #         file_list_atran.append(name)
    # #name your file
    # atran_grid_name='atran_grid_nm_add.fits'
    # #load in one example file to proper wavelength resolution
    # telluric_data = np.loadtxt('atran_5617_30_500_2_0_1.4_1.85.txt')
    # telluric_wavelength=telluric_data[:,1]*1.e3
    
    # save_atrangrid(file_list_atran, line_widths_parvi, atran_grid_name, mypool=specpool)

    for order in [22, 23, 24, 25, 26, 27, 28, 29, 33, 34, 35, 36, 37, 38, 39, 40, 41]:
        print(order)
        wave = spec_data[0][order]
        flux = spec_data[1][order]
        noise = spec_data[2][order]

        #get rid of nans in the noise array 
        noise[np.where(np.isnan(noise))] = np.nanmedian(noise)

        atran_grid_name='atran_grid_nm_add.fits'
        hdulist = fits.open(atran_grid_name)
        atran_grid =  hdulist[0].data
        water_unique =  hdulist[1].data
        angle_unique =  hdulist[2].data
        atran_wvs =  hdulist[3].data
        hdulist.close()
        atran_interpgrid = scipy.interpolate.RegularGridInterpolator((water_unique,angle_unique),atran_grid,method="linear",bounds_error=False,fill_value=0.0)

        #forward modeling approach
        nans=np.where(np.isfinite(flux))
        wvs = wave
        data=flux[nans]
        
        
        def cost_func(params):
            rv, water_unique, angle_unique = params
            #doppler shifted wavelength solution 
            c_kms=2.99792e5
            wvs_shifted= (wvs)*(1 - ((rv-barycorr) / c_kms))

            #stellar line spectrum
            phoenix_func=scipy.interpolate.splev(wvs_shifted, model_spline, der=0)

            #telluric line spectrum
            tel_func = scipy.interpolate.interp1d(atran_wvs,atran_interpgrid([water_unique,angle_unique])[0],bounds_error=False,fill_value=0)
            tel_func = tel_func(wvs)

            #model
            model= phoenix_func*tel_func

            #linear piecewise blaze function 
            blaze_chunks = 10
            x = wvs
            M_0 = np.zeros((np.size(x),(blaze_chunks+1)))
            x_knots = x[np.linspace(0,len(x)-1,blaze_chunks+1,endpoint=True).astype(np.int)]
            for piece_id in range(blaze_chunks):
                if piece_id  == 0:
                    where_chunk = np.where((x_knots[piece_id]<=x)*(x<=x_knots[piece_id+1]))
                else:
                    where_chunk = np.where((x_knots[piece_id]<x)*(x<=x_knots[piece_id+1]))
                M_0[where_chunk[0],piece_id] = 1-(x[where_chunk]-x_knots[piece_id])/(x_knots[piece_id+1]-x_knots[piece_id])
                M_0[where_chunk[0],1+piece_id] = (x[where_chunk]-x_knots[piece_id])/(x_knots[piece_id+1]-x_knots[piece_id])
            M = phoenix_func[:,None]*tel_func[:,None]*M_0
            M = M[nans[0],:]
            sigmas_vec = np.ones(np.size(data))*np.std(noise)
            M=M/sigmas_vec[:,None]
            d=data/sigmas_vec
            parameters = scipy.optimize.lsq_linear(M,d).x
            
            #apply the blaze function to the model
            model=np.dot(M, parameters)

            #define the residuals
            residuals = d - model

            #identify bad pixels using MAD 
            mad=astropy.stats.median_absolute_deviation(residuals)
            flagged_pixels=mad*10

            #refit the model after getting rid of bad pixels
            bad_pixels=np.where(np.absolute(residuals)<flagged_pixels)
            M = M[bad_pixels[0],:]
            sigmas_vec = np.ones(np.size(data[bad_pixels]))*np.std(noise[bad_pixels])
            M=M/sigmas_vec[:,None]
            d=data[bad_pixels]/sigmas_vec
            parameters = scipy.optimize.lsq_linear(M,d).x
            model=np.dot(M, parameters)

            #redefine the residuals
            residuals = d - model

            #define chi squared value to minimize 
            HPFchi2 = np.nansum((residuals / sigmas_vec) ** 2)

            return HPFchi2
        

        #initial guess for parameters
        params0=np.array([4.5, 1000, 37])
        #fixed_params=[wvs, barycorr, data, model_spline, atran_grid_name, noise]
        #how large of step size taken for minimize
        simplex_init_steps=np.array([.3, 2000, 20])
        initial_simplex = np.vstack((params0, params0+np.diag(simplex_init_steps)))

        rv_estimate=scipy.optimize.minimize(cost_func, params0, method="nelder-mead", options={"xatol": 1e-6, "maxiter": 1e5, "initial_simplex":initial_simplex, "disp":False})
        my_array=rv_estimate.x
        print(my_array)