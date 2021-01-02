#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 16:00:45 2020

@author: natsukoyamaguchi

Code to plot histograms with gaussian fit, regression graphs, 
chi-square contour plots

"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import iqr
from scipy.stats import norm
import scipy.optimize as optimization

# Loading Raw Data File

Data = np.loadtxt('/Users/natsukoyamaguchi/Desktop/NEWReaction Time Results from 4AL_4BL (Responses) - Sheet9.csv', delimiter = ',', skiprows = 1)

# Change the figure ratio as desired
fig=plt.figure(figsize=(12,6), dpi= 400, facecolor='w', edgecolor='k')

############## Cleaning up raw data, removing zeros and outliers ##############

#Gender 
rGender = Data[:,0]

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

# Only including those with data for all four experiments  

Gender = []

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
fGenderR = []
fGenderB = []
fGenderT = []
fGenderTh = []
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
      fBSD.append(BSD[i])
      fGenderB.append(Gender[i])
      fBN.append(BN[i])
      fRA.append(RA[i])
      fRSD.append(RSD[i])
      fGenderR.append(Gender[i])
      fRN.append(RN[i])
      fTA.append(TA[i])
      fTSD.append(TSD[i])
      fGenderT.append(Gender[i])
      fTN.append(TN[i])
      fThA.append(ThA[i])
      fThSD.append(ThSD[i])
      fGenderTh.append(Gender[i])
      fThN.append(ThN[i])

print('N')
print(len(fRA))

################# Plotting histograms with Gaussian Fitting ###################

for i in range(1,5):
    if i == 1:
        data = fBA 
        label = "Audio"
        color = '#2C9DD1'
    elif i == 2:
        data = fRA 
        label = "1 LED"
        color = '#FF9B16'
    elif i == 3:
        data = fTA 
        label = "2 LED"
        color = '#34CF55'
    elif i == 4:
        data = fThA
        label = "3 LED"
        color = '#E03535'
    
    print(label)
    
    bin_width = 20
    bins = range(0, 900, bin_width)
    counts, edges, plot = plt.hist(data, bins=bins, alpha=.0)
    centre = edges + (bin_width/2)
    array = [counts, edges, centre]
    
    x = np.linspace(0, 900, 700)
    
    mu, std = norm.fit(data)
    p = norm.pdf(x, mu, std)* len(data) * bin_width
    
    def gaussian(x, a,c, s):   # Change histogram here 
        return a*(1/(s*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((array[2][0:len(counts)]-c)/s)**2)))

    amp1 = 120
    cen1 = np.mean(data)
    sigma1 = np.std(data)
    popt_gauss = optimization.curve_fit(gaussian, array[2][0:len(counts)], array[0], p0=[amp1, cen1, sigma1])
    
    
    po = norm.pdf(x, popt_gauss[0][1], popt_gauss[0][2])* len(data) * bin_width
       
    # plt.hist(data, color = color, bins=bins, alpha=.7, label = label)
    # plt.plot(x, po, color = '#939393')
    # plt.xlabel("Average RT (ms)", fontweight="bold")
    # plt.ylabel('Frequency', fontweight='bold')
    # plt.xlim((0, 900))
    # plt.legend()
    
    peakpdf = 1/(popt_gauss[0][2]*np.sqrt(2*np.pi))
    peak = peakpdf* len(data) * bin_width
    
    print('peak location, sigma: ', popt_gauss[0][1], popt_gauss[0][2])
    print('peak value: ', peak)
    
    
    nonzeroind = np.nonzero(counts)[0]
    
    
    sqrtN = counts[np.min(nonzeroind):np.max(nonzeroind)+1]/np.sqrt(counts[np.min(nonzeroind):np.max(nonzeroind)+1]) 
    sqrtNremoveNan = sqrtN[np.logical_not(np.isnan(sqrtN))]
    
    error = np.std(sqrtNremoveNan)
    
    result = x[np.logical_not(np.isnan(x))]
   
    
    pred = norm.pdf(centre[0:len(counts)], popt_gauss[0][1], popt_gauss[0][2])* len(data) * bin_width
    dif = (counts - pred)**2
    ratio = dif/(np.power(error,2))
    chisquare = np.sum(ratio)
    reducedchisquare = chisquare/(len(centre)-3)
    chi = [chisquare, reducedchisquare]

    print('Gaussian Fit ChiSq, RedChiSq: ', chi)

        
##################### Overall Averages, SD, T-test ############################
      
print('Averages')
print('Audio RT: ', np.mean(fBA))
print('1 LED RT: ', np.mean(fRA))
print('2 LED RT: ', np.mean(fTA))
print('3 LED RT: ', np.mean(fThA))

print('SD')
print('Audio RT: ', np.std(fBA))
print('1 LED RT: ', np.std(fRA))
print('2 LED RT: ', np.std(fTA))
print('3 LED RT: ', np.std(fThA))


print('T-test')
print('Audio vs 1 LED RT: ',stats.ttest_ind(fRA, fBA))
print('2 LED vs 1 LED RT: ', stats.ttest_ind(fRA, fTA))
print('3 LED vs 1 LED RT: ', stats.ttest_ind(fRA, fThA))
print('3 LED vs 2 LED RT: ',stats.ttest_ind(fTA, fThA))


############################# Regression Analysis #############################

# Getting error bars
fBAerr = []
fRAerr = []
fTAerr =[]
fThAerr = []

for i in range(0, len(fBA)):
    fBAerr.append(fBSD[i]/np.sqrt(fBN[i]))
    fRAerr.append(fRSD[i]/np.sqrt(fRN[i]))
    fTAerr.append(fTSD[i]/np.sqrt(fTN[i]))
    fThAerr.append(fThSD[i]/np.sqrt(fThN[i]))

    
# Creating array of x and y axis + errors 
x_axis = [np.array(fRA), np.array(fRA), np.array(fRA), np.array(fTA)]
y_axis = [np.array(fBA), np.array(fTA), np.array(fThA), np.array(fThA)]
y_SD = [np.array(fBAerr), np.array(fTAerr), np.array(fThAerr), np.array(fThAerr)]

# Defining fit functions 

# With Intercept 
def func1(x,a,b):
    return a*x + b

# No Intercept 
def func2(x,a):
    return a*x

# Getting best fit parameters 
def optimal(i):
    optimal = optimization.curve_fit(func1, x_axis[i], y_axis[i], sigma = y_SD[i])
    return optimal

def optimalNI(i):
    optimal = optimization.curve_fit(func2, x_axis[i], y_axis[i], sigma = y_SD[i])
    return optimal

print('Audio vs 1 LED Slope, Intercept: ', optimal(0)[0])
print('2 LED vs 1 LED Slope, Intercept: ', optimal(1)[0])
print('3 LED vs 1 LED Slope, Intercept: ', optimal(2)[0])
print('3 LED vs 2 LED Slope, Intercept: ', optimal(3)[0])

print('Audio vs 1 LED Slope (NI): ', optimalNI(0)[0])
print('2 LED vs 1 LED Slope (NI): ', optimalNI(1)[0])
print('3 LED vs 1 LED Slope (NI): ', optimalNI(2)[0])
print('3 LED vs 2 LED Slope (NI): ', optimalNI(3)[0])


# Splitting data based on gender 

fBAM = []
fBAF = []
fBAMerr = []
fBAFerr = []

fRAM = []
fRAF = []
fRAMerr = []
fRAFerr = []

fTAM = []
fTAF = []
fTAMerr = []
fTAFerr = []

fThAM = []
fThAF = []
fThAMerr = []
fThAFerr = []

for i in range(0, len(fBA)):
  if fGenderB[i] == 1:
      fBAM.append(fBA[i]) 
      fBAMerr.append(fBAerr[i])
  if fGenderB[i] == 2:
      fBAF.append(fBA[i]) 
      fBAFerr.append(fBAerr[i])
  if fGenderR[i] == 1:
      fRAM.append(fRA[i]) 
      fRAMerr.append(fRAerr[i])
  if fGenderR[i] == 2:
      fRAF.append(fRA[i]) 
      fRAFerr.append(fRAerr[i])
  if fGenderT[i] == 1:
      fTAM.append(fTA[i]) 
      fTAMerr.append(fTAerr[i])
  if fGenderT[i] == 2:
      fTAF.append(fTA[i]) 
      fTAFerr.append(fTAerr[i])
  if fGenderTh[i] == 1:
      fThAM.append(fThA[i]) 
      fThAMerr.append(fThAerr[i])
  if fGenderTh[i] == 2:
      fThAF.append(fThA[i]) 
      fThAFerr.append(fThAerr[i])
    
   
male_x = [np.array(fRAM), np.array(fRAM), np.array(fRAM), np.array(fTAM)]
male_y = [np.array(fBAM), np.array(fTAM), np.array(fThAM), np.array(fThAM)]
female_x = [np.array(fRAF), np.array(fRAF), np.array(fRAF), np.array(fTAF)]
female_y = [np.array(fBAF), np.array(fTAF), np.array(fThAF), np.array(fThAF)]
male_err_x = [np.array(fRAMerr), np.array(fRAMerr), np.array(fRAMerr), np.array(fTAMerr)]
male_err_y = [np.array(fBAMerr), np.array(fTAMerr), np.array(fThAMerr), np.array(fThAMerr)]
female_err_x = [np.array(fRAFerr), np.array(fRAFerr), np.array(fRAFerr), np.array(fTAFerr)]
female_err_y = [np.array(fBAFerr), np.array(fTAFerr), np.array(fThAFerr), np.array(fThAFerr)]

# Scatter plot
def scatter(i):
    x = np.linspace(0, 800, num=len(x_axis[i]))
    linear = optimal(i)[0][0]*x + optimal(i)[0][1]
    linearNI = optimalNI(i)[0][0]*x 
    plt.errorbar(male_x[i], male_y[i], xerr=male_err_x[i], yerr = male_err_y[i], fmt = 'none',color = 'dimgrey',  elinewidth=0.2, alpha = 0.8)
    plt.errorbar(female_x[i], female_y[i], xerr=female_err_x[i], yerr = female_err_y[i], fmt = 'none',color = 'dimgrey',  elinewidth=0.2, alpha = 0.8)
    plt.scatter(male_x[i], male_y[i], s =1, color = 'blue')
    plt.scatter(female_x[i], female_y[i], s =1, color = 'red')
    plt.plot(x, linear, color = 'green', label = "Intercept", linewidth = '0.8')
    plt.plot(x, linearNI, color = 'orange', label="No intercept",  linewidth = '0.8')
    
scatter(3) 
plt.xlabel('Two LED RT (ms)')
plt.ylabel('Three LED RT (ms)')
plt.xlim(0, 700)
plt.ylim(0,800)
plt.legend()
     
####################### Goodness of fit calculations ############################
 
slope = [optimal(0)[0][0], optimal(1)[0][0], optimal(2)[0][0], optimal(3)[0][0]]
intercept = [optimal(0)[0][1], optimal(1)[0][1], optimal(2)[0][1], optimal(3)[0][1]]
slope_NI = [optimalNI(0)[0][0], optimalNI(1)[0][0], optimalNI(2)[0][0], optimalNI(3)[0][0]]

# Correlation coefficient R calculation
def R(i):
    s1 = len(y_axis[i])*(np.sum(y_axis[i]*x_axis[i]))
    s_x = np.sum(x_axis[i])
    s_y = np.sum(y_axis[i])
    numerator = s1 - (s_x)*(s_y)
    d1 = len(y_axis[i])*np.sum(x_axis[i]**2) - (s_x)**2
    d2 = len(y_axis[i])*np.sum(np.power(y_axis[i],2)) - (s_y)**2
    denominator = np.sqrt(d1*d2)
    R = numerator/denominator 
    return R

print('Audio vs 1 LED R: ', R(0))
print('2 LED vs 1 LED R: ',R(1))
print('3 LED vs 1 LED R: ',R(2))
print('3 LED vs 2 LED R: ',R(3))


# Chi Square and Reduced Chi Square Calculation

def chisq(i):
  pred = slope[i]*(x_axis[i]) + intercept[i]
  dif = (y_axis[i] - pred)**2
  ratio = dif/(np.power(y_SD[i],2))
  chisquare = np.sum(ratio)
  reducedchisquare = chisquare/(len(x_axis[i])-2)
  chi = [chisquare, reducedchisquare]
  return chi

print('Audio vs 1 LED: ', chisq(0))
print('2 LED vs 1 LED: ',chisq(1))
print('3 LED vs 1 LED: ',chisq(2))
print('3 LED vs 2 LED: ',chisq(3))

def chisqNI(i):
  pred = slope_NI[i]*(x_axis[i])
  dif = (y_axis[i] - pred)**2
  ratio = dif/(np.power(y_SD[i],2))
  chisquare = np.sum(ratio)
  reducedchisquare = chisquare/(len(x_axis[i])-1)
  chi = [chisquare, reducedchisquare]
  return chi

print('Audio vs 1 LED (No Intercept): ',chisqNI(0))
print('2 LED vs 1 LED (No Intercept): ',chisqNI(1))
print('3 LED vs 1 LED (No Intercept): ',chisqNI(2))
print('3 LED vs 2 LED (No Intercept): ',chisqNI(3))

# Getting Parameter Errors

def slope_error(i):
  a = np.sum((x_axis[i]**2)/(np.power(y_SD[i],2)))
  b1 = intercept[i]*x_axis[i]-y_axis[i]*x_axis[i]
  b = np.sum((2*b1/(np.power(y_SD[i],2))))
  c1 = y_axis[i]**2 - 2*intercept[i]*y_axis[i] + (intercept[i]**2)
  c = np.sum(c1/(np.power(y_SD[i],2))) - (1+chisq(i)[0])
  delta = b**2 - (4*a*c)
  a1 = (-b + np.sqrt(delta))/(2*a)
  a2 = (-b - np.sqrt(delta))/(2*a)
  slo_error = [a1, a2]
  return slo_error

print('Slope Error (With Intercept)')
print('Audio vs 1 LED: ', slope[0]-slope_error(0)[0], slope[0]-slope_error(0)[1])
print('2 LED vs 1 LED: ', slope[1]-slope_error(1)[0], slope[1]-slope_error(1)[1])
print('3 LED vs 1 LED: ', slope[2]-slope_error(2)[0], slope[2]-slope_error(2)[1])
print('3 LED vs 2 LED: ', slope[3]-slope_error(3)[0], slope[3]-slope_error(3)[1])

def intercept_error(i):
  a = np.sum(1/(np.power(y_SD[i],2)))
  b1 = slope[i]*(x_axis[i])-y_axis[i]
  b = np.sum((2*b1)/(np.power(y_SD[i],2)))
  c1 = y_axis[i]**2 + (slope[i]**2)*(x_axis[i]**2)-(2*slope[i]*x_axis[i]*y_axis[i])
  c = np.sum((c1)/(np.power(y_SD[i],2))) - (1+chisq(i)[0])
  delta = b**2 - (4*a*c)
  b1 = (-b + np.sqrt(delta))/(2*a)
  b2 = (-b - np.sqrt(delta))/(2*a)
  int_error = [b1, b2]
  return int_error

print('Intercept Error (With Intercept)')
print('Audio vs 1 LED: ', intercept[0]-intercept_error(0)[0], intercept[0]-intercept_error(0)[1])
print('2 LED vs 1 LED: ', intercept[1]-intercept_error(1)[0], intercept[1]-intercept_error(1)[1])
print('3 LED vs 1 LED: ', intercept[2]-intercept_error(2)[0], intercept[2]-intercept_error(2)[1])
print('3 LED vs 2 LED: ', intercept[3]-intercept_error(3)[0], intercept[3]-intercept_error(3)[1])

def slopeNI_error(i):
  a = np.sum((x_axis[i]**2)/(np.power(y_SD[i],2)))
  b1 = -y_axis[i]*x_axis[i]
  b = np.sum((2*b1/(np.power(y_SD[i],2))))
  c1 = y_axis[i]**2 
  c = np.sum(c1/(np.power(y_SD[i],2))) - (1+chisqNI(i)[0])
  delta = b**2 - (4*a*c)
  a1 = (-b + np.sqrt(delta))/(2*a)
  a2 = (-b - np.sqrt(delta))/(2*a)
  slo_error = [a1, a2]
  return slo_error

print('Slope Error (No Intercept)')
print('Audio vs 1 LED: ', slope_NI[0]-slopeNI_error(0)[0], slope_NI[0]-slopeNI_error(0)[1])
print('2 LED vs 1 LED: ', slope_NI[1]-slopeNI_error(1)[0], slope_NI[1]-slopeNI_error(1)[1])
print('3 LED vs 1 LED: ', slope_NI[2]-slopeNI_error(2)[0], slope_NI[2]-slopeNI_error(2)[1])
print('3 LED vs 2 LED: ', slope_NI[3]-slopeNI_error(3)[0], slope_NI[3]-slopeNI_error(3)[1])

# Normalizing Red. Chi Square 

def factor(i):
    f = np.sqrt(((len(x_axis[i])-2)**-1)*chisq(i)[0])
    return f

print('Factor multiplied to sigma to normalize red. chisq (Intercept)')
print('Audio vs 1 LED: ', factor(0))
print('2 LED vs 1 LED: ', factor(1))
print('3 LED vs 1 LED: ', factor(2))
print('3 LED vs 2 LED: ', factor(3))

def factorNI(i):
    f = np.sqrt(((len(x_axis[i])-1)**-1)*chisqNI(i)[0])
    return f

print('Factor multiplied to sigma to normalize red. chisq (No Intercept)')
print('Audio vs 1 LED: ', factorNI(0))
print('2 LED vs 1 LED: ', factorNI(1))
print('3 LED vs 1 LED: ', factorNI(2))
print('3 LED vs 2 LED: ', factorNI(3))

def chisqNorm(i):
  pred = slope[i]*(x_axis[i]) + intercept[i]
  dif = (y_axis[i] - pred)**2
  ratio = dif/(np.power(y_SD[i]*factor(i),2))
  chisquare = np.sum(ratio)
  reducedchisquare = chisquare/(len(x_axis[i])-2)
  chi = [chisquare, reducedchisquare]
  return chi

print('Normalized Chi Sq (Intercept)')
print('Audio vs 1 LED: ', chisqNorm(0))
print('2 LED vs 1 LED: ',chisqNorm(1))
print('3 LED vs 1 LED: ',chisqNorm(2))
print('3 LED vs 2 LED: ',chisqNorm(3))

def chisqNINorm(i):
  pred = slope_NI[i]*(x_axis[i])
  dif = (y_axis[i] - pred)**2
  ratio = dif/(np.power(y_SD[i]*factorNI(i),2))
  chisquare = np.sum(ratio)
  reducedchisquare = chisquare/(len(x_axis[i])-1)
  chi = [chisquare, reducedchisquare]
  return chi

print('Normalized Chi Sq (No Intercept)')
print('Audio vs 1 LED (No Intercept): ',chisqNINorm(0))
print('2 LED vs 1 LED (No Intercept): ',chisqNINorm(1))
print('3 LED vs 1 LED (No Intercept): ',chisqNINorm(2))
print('3 LED vs 2 LED (No Intercept): ',chisqNINorm(3))

def slope_errorNorm(i):
  a = np.sum((x_axis[i]**2)/(np.power(y_SD[i]*factor(i),2)))
  b1 = intercept[i]*x_axis[i]-y_axis[i]*x_axis[i]
  b = np.sum((2*b1/(np.power(y_SD[i]*factor(i),2))))
  c1 = y_axis[i]**2 - 2*intercept[i]*y_axis[i] + (intercept[i]**2)
  c = np.sum(c1/(np.power(y_SD[i]*factor(i),2))) - (1+chisqNorm(i)[0])
  delta = b**2 - (4*a*c)
  a1 = (-b + np.sqrt(delta))/(2*a)
  a2 = (-b - np.sqrt(delta))/(2*a)
  slo_error = [a1, a2]
  return slo_error

print(slope_errorNorm(0))
print('NORM Slope Error (With Intercept)')
print('Audio vs 1 LED: ', slope[0]-slope_errorNorm(0)[0], slope[0]-slope_errorNorm(0)[1])
print('2 LED vs 1 LED: ', slope[1]-slope_errorNorm(1)[0], slope[1]-slope_errorNorm(1)[1])
print('3 LED vs 1 LED: ', slope[2]-slope_errorNorm(2)[0], slope[2]-slope_errorNorm(2)[1])
print('3 LED vs 2 LED: ', slope[3]-slope_errorNorm(3)[0], slope[3]-slope_errorNorm(3)[1])

def intercept_errorNorm(i):
  a = np.sum(1/(np.power(y_SD[i]*factor(i),2)))
  b1 = slope[i]*(x_axis[i])-y_axis[i]
  b = np.sum((2*b1)/(np.power(y_SD[i]*factor(i),2)))
  c1 = y_axis[i]**2 + (slope[i]**2)*(x_axis[i]**2)-(2*slope[i]*x_axis[i]*y_axis[i])
  c = np.sum((c1)/(np.power(y_SD[i]*factor(i),2))) - (1+chisqNorm(i)[0])
  delta = b**2 - (4*a*c)
  b1 = (-b + np.sqrt(delta))/(2*a)
  b2 = (-b - np.sqrt(delta))/(2*a)
  int_error = [b1, b2]
  return int_error

print('NORM Intercept Error (With Intercept)')
print('Audio vs 1 LED: ', intercept[0]-intercept_errorNorm(0)[0], intercept[0]-intercept_errorNorm(0)[1])
print('2 LED vs 1 LED: ', intercept[1]-intercept_errorNorm(1)[0], intercept[1]-intercept_errorNorm(1)[1])
print('3 LED vs 1 LED: ', intercept[2]-intercept_errorNorm(2)[0], intercept[2]-intercept_errorNorm(2)[1])
print('3 LED vs 2 LED: ', intercept[3]-intercept_errorNorm(3)[0], intercept[3]-intercept_errorNorm(3)[1])

def slopeNI_errorNorm(i):
  a = np.sum((x_axis[i]**2)/(np.power(y_SD[i]*factorNI(i),2)))
  b1 = -y_axis[i]*x_axis[i]
  b = np.sum((2*b1/(np.power(y_SD[i]*factorNI(i),2))))
  c1 = y_axis[i]**2 
  c = np.sum(c1/(np.power(y_SD[i]*factorNI(i),2))) - (1+chisqNINorm(i)[0])
  delta = b**2 - (4*a*c)
  a1 = (-b + np.sqrt(delta))/(2*a)
  a2 = (-b - np.sqrt(delta))/(2*a)
  slo_error = [a1, a2]
  return slo_error

print('NORM Slope Error (No Intercept)')
print('Audio vs 1 LED: ', slope_NI[0]-slopeNI_errorNorm(0)[0], slope_NI[0]-slopeNI_errorNorm(0)[1])
print('2 LED vs 1 LED: ', slope_NI[1]-slopeNI_errorNorm(1)[0], slope_NI[1]-slopeNI_errorNorm(1)[1])
print('3 LED vs 1 LED: ', slope_NI[2]-slopeNI_errorNorm(2)[0], slope_NI[2]-slopeNI_errorNorm(2)[1])
print('3 LED vs 2 LED: ', slope_NI[3]-slopeNI_errorNorm(3)[0], slope_NI[3]-slopeNI_errorNorm(3)[1])

def chisq_param (i, m,c):
    pred = m *x_axis[i] + c     
    dif = (y_axis[i] - pred)**2
    ratio = dif/(np.power(y_SD[i]*factor(i),2)) 
    chisquare = np.sum(ratio) 
    return chisquare

fig=plt.figure(figsize=(6,6), dpi= 400, facecolor='w', edgecolor='k')

############################ Chi Square Contour Plots ##########################

def contour(k):
    grad = np.linspace(optimal(k)[0][0]-1, optimal(k)[0][0]+1, num = 1000)  
    inter = np.linspace(optimal(k)[0][1]-190, optimal(k)[0][1]+190, num = 1200)
    chi_square_values = np.array([[chisq_param(k, grad[i], inter[j]) for j in range(len(inter))]for i in range(len(grad))])
    data = [[[grad[i], inter[j], chi_square_values[i, j]] for i in range(len(grad))] for j in range(len(inter))]
    data = np.array(data)
    x_values = data[:, :, 0]
    y_values = data[:, :, 1]
    values = data[:, :, 2] - np.min(chi_square_values)
    levels = [2.2957489, 6.1800743, 11.829158, 19.333909]
    contour = plt.contour(x_values, y_values, values, colors=('blue', 'indigo', 'forestgreen', 'orange'), levels=levels, linewidths = 0.5)
    fmt = {}
    strs = [ '1σ', '2σ', '3σ','4σ']
    for l,s in zip( contour.levels, strs ):
        fmt[l] = s
    plt.clabel(contour,contour.levels[::1],inline=False,fmt=fmt,fontsize=10, colors = 'black', use_clabeltext=True)
   
    plt.axhline(optimal(k)[0][1], linewidth = 0.2)
    plt.axvline(optimal(k)[0][0], linewidth = 0.2)
    bar = plt.colorbar(contour)
    bar.ax.set_yticklabels(['1σ', '2σ', '3σ','4σ'])
    plt.xlabel('Slope')
    plt.ylabel('Intercept')


contour(3)

# x and y axis limits 

#Audio vs 1 LED 
# plt.xlim(0.68,1.08)
# plt.ylim(-50, 30)
#2 vs 1 LED
# plt.xlim(0.3,1.75)
# plt.ylim(-45, 250)
#3 vs 1 LED
# plt.xlim(0.1,2)
# plt.ylim(10, 400)
#3 vs 2 LED
plt.xlim(0.95,1.4)
plt.ylim(-45, 100)

plt.title('3 LED vs 2 LED (X² - min(X²)) Map')