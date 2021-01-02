#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 14:41:01 2020

@author: natsukoyamaguchi

Plots histograms of Gender/Lifestyle categories with gaussian fit 

"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import iqr
import scipy.optimize as optimization

Data = np.loadtxt('/Users/natsukoyamaguchi/Desktop/NEWReaction Time Results from 4AL_4BL (Responses) - Sheet9.csv', delimiter = ',', skiprows = 1)
fig=plt.figure(figsize=(10,5), dpi= 400, facecolor='w', edgecolor='k')

rGender = Data[:,0]
rSports = Data[:,1]
rMusic = Data[:,2]
rGames = Data[:,3]

# Raw Average Reaction Time
rRA = Data[:,4]
rBA = Data[:,7]
rTA = Data[:,10]
rThA = Data[:,13]

# Raw Standard Deviation
rRSD = Data[:,5]
rBSD = Data[:,8]
rTSD = Data[:,11]
rThSD = Data[:,14]

# Raw Number of Trials 
rRN = Data[:,6]
rBN = Data[:,9]
rTN = Data[:,12]
rThN = Data[:,15]

# Only including those who performed all four tests 

Gender = []
Music = []
Sports = []
Games = []

RA = []
BA = []
TA = []
ThA = []

RSD = []
BSD = []
TSD = []
ThSD = []

RN = []
BN = []
TN = []
ThN = []


for i in range(0,len(rRA)):
    if rRA[i] != 0 and rBA[i] != 0 and rTA[i] != 0 and rThA[i] != 0 and rRSD[i] != 0 and rBSD[i] != 0 and rTSD[i] != 0 and rThSD[i] != 0:
        RA.append(rRA[i])
        BA.append(rBA[i])
        TA.append(rTA[i])
        ThA.append(rThA[i])    
        RSD.append(rRSD[i])
        BSD.append(rBSD[i])
        TSD.append(rTSD[i])
        ThSD.append(rThSD[i])
        RN.append(rRN[i])
        BN.append(rBN[i])
        TN.append(rTN[i])
        ThN.append(rThN[i])
        Gender.append(rGender[i])
        Music.append(rMusic[i])
        Games.append(rGames[i])
        Sports.append(rSports[i])
        

Average = [np.array(BA), np.array(RA), np.array(TA), np.array(ThA)]
SD = [np.array(BSD), np.array(RSD), np.array(TSD), np.array(ThSD)]


def poly(i):
    poly = np.polyfit(Average[i], SD[i], 1)
    return poly 

# Calculation of Normalized Standard Deviation 
    
def NSD(i):
    Pred = poly(i)[0]*Average[i] + poly(i)[1]
    NSD = SD[i] - Pred
    return NSD 

# Removing outliers based on IQR for the Average RT and NSD 

fBA = []
fRA = []
fTA =[]
fThA = []

fGender = []
fSports = []
fMusic = []
fGames = []

fBSD = []
fRSD = []
fTSD =[]
fThSD = []

fRN = []
fBN = []
fTN = []
fThN = []

for i in range(0, len(RA)):
  if BA[i] < np.percentile(BA, 75)+ 1.5*iqr(BA) and BA[i] > np.percentile(BA, 25) - 1.5*iqr(BA) and NSD(0)[i] < np.percentile(NSD(0), 75)+ 1.5*iqr(NSD(0)) and NSD(0)[i] > np.percentile(NSD(0), 25) - 1.5*iqr(NSD(0)) and RA[i] < np.percentile(RA, 75)+ 1.5*iqr(RA) and RA[i] > np.percentile(RA, 25) - 1.5*iqr(RA) and NSD(1)[i] < np.percentile(NSD(1), 75)+ 1.5*iqr(NSD(1)) and NSD(1)[i] > np.percentile(NSD(1), 25) - 1.5*iqr(NSD(1)) and TA[i] < np.percentile(TA, 75)+ 1.5*iqr(TA) and TA[i] > np.percentile(TA, 25) - 1.5*iqr(TA) and NSD(2)[i] < np.percentile(NSD(2), 75)+ 1.5*iqr(NSD(2)) and NSD(2)[i] > np.percentile(NSD(2), 25) - 1.5*iqr(NSD(2)) and ThA[i] < np.percentile(ThA, 75)+ 1.5*iqr(ThA) and ThA[i] > np.percentile(ThA, 25) - 1.5*iqr(ThA)  and NSD(3)[i] < np.percentile(NSD(3), 75)+ 1.5*iqr(NSD(3)) and NSD(3)[i] > np.percentile(NSD(3), 25) - 1.5*iqr(NSD(3)):
      fBA.append(BA[i])
      fRA.append(RA[i])
      fTA.append(TA[i])
      fThA.append(ThA[i])
      fGender.append(Gender[i])
      fSports.append(Sports[i])
      fGames.append(Games[i])
      fMusic.append(Music[i])
      fBSD.append(BSD[i])
      fRSD.append(RSD[i])
      fTSD.append(TSD[i])
      fThSD.append(ThSD[i])
      fBN.append(BN[i])
      fRN.append(RN[i])
      fTN.append(TN[i])
      fThN.append(ThN[i])
      
RMale = []
RFemale = []
BMale = []
BFemale = []
TMale = []
TFemale = []
ThMale = []
ThFemale = []

for i in range(0,len(fRA)):
  if fGender[i] == 1:
    RMale.append(fRA[i])
    BMale.append(fBA[i])
    TMale.append(fTA[i])
    ThMale.append(fThA[i])

for i in range(0,len(fRA)):
  if fGender[i] == 2:
    RFemale.append(fRA[i])
    BFemale.append(fBA[i])
    TFemale.append(fTA[i])
    ThFemale.append(fThA[i])
        

RVGames = []
RNVGames = []
BVGames = []
BNVGames = []
TVGames = []
TNVGames = []
ThVGames = []
ThNVGames = []

for i in range(0,len(fRA)):
  if fGames[i] == 1 :
    RNVGames.append(fRA[i])
    BNVGames.append(fBA[i])
    TNVGames.append(fTA[i])
    ThNVGames.append(fThA[i])


for i in range(0,len(fRA)):
  if fGames[i] == 4 or fGames[i] == 5 :
    RVGames.append(fRA[i])
    BVGames.append(fBA[i])
    TVGames.append(fTA[i])
    ThVGames.append(fThA[i])

# Plotting histograms

from scipy.stats import norm

O_array = [fBA, fRA, fTA, fThA]
M_array = [BMale, RMale, TMale, ThMale]
F_array = [BFemale, RFemale, TFemale, ThFemale]
G_array = [BVGames, RVGames, TVGames, ThVGames]
NG_array = [BNVGames, RNVGames, TNVGames, ThNVGames]

array_array = [O_array, M_array, F_array, G_array, NG_array]

def histogram(j, i):
    counts, edges, plot = plt.hist(array_array[j][i], bins=10, alpha=.0)
    bin_width = np.ptp(array_array[j][i]) / 10
    centre = edges + (bin_width/2)
    array = [counts, edges, centre]
    return array

x = np.linspace(100, 800, 500)

def gaus(j, i):
    mu, std = norm.fit(array_array[j][i])
    bin_width = np.ptp(array_array[j][i]) / 10
    p = norm.pdf(x, mu, std)* len(array_array[j][i]) * bin_width
    return p

def gaussian(x, a,c, s):   # Change histogram here 
    return a*(1/(s*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((histogram(1,3)[2][0:10]-c)/s)**2)))

def popt_gauss(j,i):
    amp1 = 100
    cen1 = np.mean(array_array[j][i])
    sigma1 = np.std(array_array[j][i])
    popt_gauss = optimization.curve_fit(gaussian, histogram(j, i)[2][0:10], histogram(j, i)[0], p0=[amp1, cen1, sigma1])
    return [popt_gauss[0][1], popt_gauss[0][2]]


def p(j,i):
    bin_width = np.ptp(array_array[j][i]) / 10
    p = norm.pdf(x, popt_gauss(j,i)[0], popt_gauss(j,i)[1])* len(array_array[j][i]) * bin_width
    return p

def peak(j,i):
    bin_width = np.ptp(array_array[j][i]) / 10
    peakpdf = 1/(popt_gauss(j,i)[1]*np.sqrt(2*np.pi))
    peak = peakpdf* len(array_array[j][i]) * bin_width
    return peak

def pr(j,i):
    print('peak location, sigma: ', popt_gauss(j,i))
    print('peak value: ', peak(j,i))
    plt.hist(array_array[j][i], bins=10, alpha=.6, color = 'steelblue', label = '3 LED Male')
    plt.plot(x, p(j,i), color = 'k', label = 'Normal fit')
    plt.xlabel("Average RT (ms)", fontweight="bold")
    plt.ylabel('Frequency', fontweight='bold')
    plt.xlim((175, 800))
    plt.legend()
    
pr(1,3)


