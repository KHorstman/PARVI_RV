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
from astropy.time import Time
import matplotlib.cm as cm


# #plt.xlim([58805.4, 58806.6])

# data=np.loadtxt('GJ229_on_sky_order_35_first.dat')
# date=data[0]
# rv=data[1]
# min_error_posterior=data[2]
# max_error_posterior=data[3]

# print(min_error_posterior)

# actual_rv=[4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7]

# plt.xlabel('Modified Julian Date', fontsize=30)
# plt.ylabel('Radial Velocity (km/s)', color='black', fontsize=30)
# plt.scatter(date, rv, label='Stellar RV', marker='*', color='royalblue', s=200, zorder=4)
# plt.plot(date, actual_rv, color='red', linestyle='--', label='Known Stellar RV')
# #plt.plot([58859, 58860], actual_rv, color='red', linestyle='--', label='Known Stellar RV')
# plt.errorbar(date, rv, yerr=[min_error_posterior, max_error_posterior], capsize=3, fmt='none', c='gray', zorder=1)

# plt.legend(loc=1, fontsize=20)

# plt.tick_params(labelsize=20)

# plt.show()

# file_name=[]

# for file in range(0,43):
#     order="GJ229_on_sky_order"+f"{[file]}"+".dat"
#     file_name.append(order)

# #print(file_name)

# data = [np.loadtxt(open(file), usecols=(0,1,2,3)) for file in file_name]

# # # date=data[35]
# # print(data[35][2][2])

# # for i in range(43):
# #       plt.scatter(data[i][0], data[i][1]+i, alpha=1, marker='o')
# #       plt.plot([data[0][0][0], data[0][0][3]], [4.7+i, 4.7+i], alpha=.5, color='red')

# # plt.show()

# #yerr_low=data[i][2][2]
# #yerr_high=data[i][3][2]

# #rv as a function of order
# for i in range(43):
#       plt.scatter(i, data[i][1][0], alpha=1, marker='*', s=300)
#       #plt.errorbar(i, data[i][1][2], yerr=[yerr_low, yerr_high], capsize=3, fmt='none', c='gray', zorder=1)
# plt.plot([0,43], [4.7, 4.7], alpha=.5, color='red', label='Measured Stellar RV')

# plt.xticks(np.arange(0, 43, 1))
# plt.title("All Orders for ONE File", fontsize=30)
# plt.legend(loc=1, fontsize=20)
# plt.tick_params(labelsize=20)
# plt.xlabel("Order #", fontsize=30)
# plt.ylabel('Radial Velocity (km/s)', color='black', fontsize=30)

# plt.show()

# data for new files (second attempt)
# data=np.loadtxt('GJ229_on_sky_order[35]second_attempt.dat')
# date=data[0]
# rv=data[1]
# min_error_posterior=data[2]
# max_error_posterior=data[3]

# print(len(date))
# print(len(rv))

file_name=[]

for file in range(20,43):
    order="GJ229_on_sky_order"+f"{[file]}"+"second_attempt"+".dat"
    file_name.append(order)

print(file_name)

#data = [np.loadtxt(open(file), usecols=(0,1,2,3)) for file in file_name]
data = [np.loadtxt(open(file)) for file in file_name]
#print(data[15][2])

#key for how I created the files
# data[0]=the first order (with all its dates, rvs, and uncertainities for each)
# data[0][0]=date, data[0][1]=rv, etc.

print(data[22][2])

rv_file_one=[]
date_file_one=[]
min_error_file_one=[]
max_error_file_one=[]
order_file_one=[]

#rv as a function of order
for i in range(0,23):
    rv_file_one.append(data[i][1][14])
    date_file_one.append(data[i][0][14])
    min_error_file_one.append(data[i][2][14])
    max_error_file_one.append(data[i][3][14])
    order_file_one.append(i+20)
    #plt.scatter(i+20, data[i][1][0], alpha=1, marker='*', s=300)
    #plt.errorbar(i+20, data[i][1][0], yerr=[data[i][1][2], data[i][1][3]], capsize=3, fmt='none', c='gray', zorder=1)

colors = cm.rainbow(np.linspace(0, 1, len(rv_file_one)))
plt.scatter(order_file_one, rv_file_one, alpha=1, marker='*', s=300, color=colors)
plt.errorbar(order_file_one, rv_file_one, yerr=[min_error_file_one, max_error_file_one], capsize=3, fmt='none', c='gray', zorder=0)
#plt.plot([19,43], [4.7, 4.7], alpha=.5, color='red', label='Measured Stellar RV')
plt.ylim([-2, 0])

plt.xticks(np.arange(20, 43, 1))
plt.title("All Orders for ONE File", fontsize=30)
plt.legend(loc=1, fontsize=20)
plt.tick_params(labelsize=20)
plt.xlabel("Order #", fontsize=30)
plt.ylabel('Radial Velocity (km/s)', color='black', fontsize=30)

plt.show()