#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 10:53:58 2020

@author: natsukoyamaguchi

Plots the average RTs and errors for gender/lifesrtyle groups

"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import iqr

Data = np.loadtxt('/Users/natsukoyamaguchi/Desktop/NEWReaction Time Results from 4AL_4BL (Responses) - Sheet9.csv', delimiter = ',', skiprows = 1)
fig=plt.figure(figsize=(10,10), dpi= 400, facecolor='w', edgecolor='k')

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
      
print('Total N: ', len(fRA))
      
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
      
RGendermean = [np.mean(RMale),np.mean(RFemale)]
BGendermean = [np.mean(BMale),np.mean(BFemale)]
TGendermean = [np.mean(TMale),np.mean(TFemale)]
ThGendermean = [np.mean(ThMale),np.mean(ThFemale)]

RGenderSD = [np.std(RMale),np.std(RFemale)]
BGenderSD = [np.std(BMale),np.std(BFemale)]
TGenderSD = [np.std(TMale),np.std(TFemale)]
ThGenderSD = [np.std(ThMale),np.std(ThFemale)]

print('Gender Male vs Female')

print('Audio RT Mean: ', BGendermean)
print('1 LED RT Mean: ', RGendermean)
print('2 LED RT Mean: ', TGendermean)
print('3 LED RT Mean: ', ThGendermean)

print('Audio RT SD: ', BGenderSD)
print('1 LED RT SD: ', RGenderSD)
print('2 LED RT SD: ', TGenderSD)
print('3 LED RT SD: ', ThGenderSD)

print('Audio RT p-value: ', stats.ttest_ind(BMale, BFemale)[1])
print('1 LED RT p-value: ', stats.ttest_ind(RMale, RFemale)[1])
print('2 LED RT p-value: ', stats.ttest_ind(TMale, TFemale)[1])
print('3 LED RT p-value: ', stats.ttest_ind(ThMale, ThFemale)[1])

print('N:', len(BMale), len(BFemale), len(RMale), len(RFemale), len(TMale), len(TFemale), len(ThMale), len(ThFemale))


RMusicP = []
RNMusicP = []
BMusicP = []
BNMusicP = []
TMusicP = []
TNMusicP = []
ThMusicP = []
ThNMusicP = []

for i in range(0,len(fRA)):
  if fMusic[i] == 1:
    RNMusicP.append(fRA[i])
    BNMusicP.append(fBA[i])
    TNMusicP.append(fTA[i])
    ThNMusicP.append(fThA[i])

for i in range(0,len(fRA)):
  if fMusic[i] == 4 or fMusic[i] == 5 :
    RMusicP.append(fRA[i])
    BMusicP.append(fBA[i])
    TMusicP.append(fTA[i])
    ThMusicP.append(fThA[i])
      
RMusicmean = [np.mean(RNMusicP),np.mean(RMusicP)]
BMusicmean = [np.mean(BNMusicP),np.mean(BMusicP)]
TMusicmean = [np.mean(TNMusicP),np.mean(TMusicP)]
ThMusicmean = [np.mean(ThNMusicP),np.mean(ThMusicP)]

RMusicSD = [np.std(RNMusicP),np.std(RMusicP)]
BMusicSD = [np.std(BNMusicP),np.std(BMusicP)]
TMusicSD = [np.std(TNMusicP),np.std(TMusicP)]
ThMusicSD = [np.std(ThNMusicP),np.std(ThMusicP)]

print('Music Non-Players vs Players')

print('Audio RT Mean: ',BMusicmean)
print('1 LED RT Mean: ',RMusicmean)
print('2 LED RT Mean: ',TMusicmean)
print('3 LED RT Mean: ', ThMusicmean)

print('Audio RT SD: ',BMusicSD)
print('1 LED RT SD: ',RMusicSD)
print('2 LED RT SD: ',TMusicSD)
print('3 LED RT SD: ', ThMusicSD)

print('Audio RT p-value: ',stats.ttest_ind(BNMusicP, BMusicP)[1])
print('1 LED RT p-value: ',stats.ttest_ind(RNMusicP, RMusicP)[1])
print('2 LED RT p-value: ',stats.ttest_ind(TNMusicP, TMusicP)[1])
print('3 LED RT p-value: ', stats.ttest_ind(ThNMusicP, ThMusicP)[1])

print('N:', len(BNMusicP), len(BMusicP), len(RNMusicP), len(RMusicP), len(TNMusicP), len(TMusicP), len(ThNMusicP), len(ThMusicP))


RSport = []
RNSport = []
BSport = []
BNSport = []
TSport = []
TNSport = []
ThSport = []
ThNSport = []

for i in range(0,len(fRA)):
  if fSports[i] == 1 :
    RNSport.append(fRA[i])
    BNSport.append(fBA[i])
    TNSport.append(fTA[i])
    ThNSport.append(fThA[i])


for i in range(0,len(fRA)):
  if fSports[i] == 4 or fSports[i] == 5 :
    RSport.append(fRA[i])
    BSport.append(fBA[i])
    TSport.append(fTA[i])
    ThSport.append(fThA[i])

RSportsmean = [np.mean(RNSport),np.mean(RSport)]
BSportsmean = [np.mean(BNSport),np.mean(BSport)]
TSportsmean = [np.mean(TNSport),np.mean(TSport)]
ThSportsmean = [np.mean(ThNSport),np.mean(ThSport)]

RSportsSD = [np.std(RNSport),np.std(RSport)]
BSportsSD = [np.std(BNSport),np.std(BSport)]
TSportsSD = [np.std(TNSport),np.std(TSport)]
ThSportsSD = [np.std(ThNSport),np.std(ThSport)]
   
print('No Sports vs Sports')
 
print('Audio RT Mean: ', BSportsmean)
print('1 LED RT Mean: ', RSportsmean)
print('2 LED RT Mean: ', TSportsmean)
print('3 LED RT Mean: ', ThSportsmean)

print('Audio RT SD: ', BSportsSD)
print('1 LED RT SD: ', RSportsSD)
print('2 LED RT SD: ', TSportsSD)
print('3 LED RT SD: ', ThSportsSD)

print('Audio RT p-value: ', stats.ttest_ind(BNSport, BSport)[1])
print('1 LED RT p-value: ', stats.ttest_ind(RNSport, RSport)[1])
print('2 LED RT p-value: ', stats.ttest_ind(TNSport, TSport)[1])
print('3 LED RT p-value: ', stats.ttest_ind(ThNSport, ThSport)[1])

print('N:', len(BNSport), len(BSport), len(RNSport), len(RSport), len(TNSport), len(TSport), len(ThNSport), len(ThSport))

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

RGamesmean = [np.mean(RNVGames),np.mean(RVGames)]
BGamesmean = [np.mean(BNVGames),np.mean(BVGames)]
TGamesmean = [np.mean(TNVGames),np.mean(TVGames)]
ThGamesmean = [np.mean(ThNVGames),np.mean(ThVGames)]

RGamesSD = [np.std(RNVGames),np.std(RVGames)]
BGamesSD = [np.std(BNVGames),np.std(BVGames)]
TGamesSD = [np.std(TNVGames),np.std(TVGames)]
ThGamesSD = [np.std(ThNVGames),np.std(ThVGames)]

print('Non Gamers vs Gamers')

print('Audio RT Mean: ', BGamesmean)
print('1 LED RT Mean: ', RGamesmean)
print('2 LED RT Mean: ', TGamesmean)
print('3 LED RT Msan: ', ThGamesmean)

print('Audio RT SD: ', BGamesSD)
print('1 LED RT SD: ', RGamesSD)
print('2 LED RT SD: ', TGamesSD)
print('3 LED RT SD: ', ThGamesSD)

print('Audio RT p-value: ', stats.ttest_ind(BNVGames, BVGames)[1])
print('1 LED RT p-value: ', stats.ttest_ind(RNVGames, RVGames)[1])
print('2 LED RT p-value: ', stats.ttest_ind(TNVGames, TVGames)[1])
print('3 LED RT p-value: ', stats.ttest_ind(ThNVGames, ThVGames)[1])

print('N:', len(BNVGames), len(BVGames), len(RNVGames), len(RVGames), len(TNVGames), len(TVGames), len(ThNVGames), len(ThVGames))

#Gender but removing video games 
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

RGamesmean = [np.mean(RNVGames),np.mean(RVGames)]
BGamesmean = [np.mean(BNVGames),np.mean(BVGames)]
TGamesmean = [np.mean(TNVGames),np.mean(TVGames)]
ThGamesmean = [np.mean(ThNVGames),np.mean(ThVGames)]

RGamesSD = [np.std(RNVGames),np.std(RVGames)]
BGamesSD = [np.std(BNVGames),np.std(BVGames)]
TGamesSD = [np.std(TNVGames),np.std(TVGames)]
ThGamesSD = [np.std(ThNVGames),np.std(ThVGames)]

print('Non Gamers vs Gamers')

print('Audio RT Mean: ', BGamesmean)
print('1 LED RT Mean: ', RGamesmean)
print('2 LED RT Mean: ', TGamesmean)
print('3 LED RT Msan: ', ThGamesmean)

print('Audio RT SD: ', BGamesSD)
print('1 LED RT SD: ', RGamesSD)
print('2 LED RT SD: ', TGamesSD)
print('3 LED RT SD: ', ThGamesSD)

print('Audio RT p-value: ', stats.ttest_ind(BNVGames, BVGames)[1])
print('1 LED RT p-value: ', stats.ttest_ind(RNVGames, RVGames)[1])
print('2 LED RT p-value: ', stats.ttest_ind(TNVGames, TVGames)[1])
print('3 LED RT p-value: ', stats.ttest_ind(ThNVGames, ThVGames)[1])

print('N:', len(BNVGames), len(BVGames), len(RNVGames), len(RVGames), len(TNVGames), len(TVGames), len(ThNVGames), len(ThVGames))

GenderArray =[np.mean(BMale),np.mean(BFemale), np.mean(RMale),np.mean(RFemale), np.mean(TMale),np.mean(TFemale), np.mean(ThMale),np.mean(ThFemale)]
MusicArray = [np.mean(BMusicP),np.mean(BNMusicP), np.mean(RMusicP),np.mean(RNMusicP),np.mean(TMusicP),np.mean(TNMusicP), np.mean(ThMusicP), np.mean(ThNMusicP)]
SportsArray = [np.mean(BSport),np.mean(BNSport), np.mean(RSport), np.mean(RNSport),np.mean(TSport), np.mean(TNSport),np.mean(ThSport), np.mean(ThNSport)]
GamesArray = [np.mean(BVGames),np.mean(BNVGames),np.mean(RVGames), np.mean(RNVGames), np.mean(TVGames),np.mean(TNVGames), np.mean(ThVGames) , np.mean(ThNVGames)]

GenderErr = [np.std(BMale)/np.sqrt(len(BMale)),np.std(BFemale)/np.sqrt(len(BFemale)), np.std(RMale)/np.sqrt(len(RMale)),np.std(RFemale)/np.sqrt(len(RFemale)), np.std(TMale)/np.sqrt(len(TMale)),np.std(TFemale)/np.sqrt(len(TFemale)), np.std(ThMale)/np.sqrt(len(ThMale)),np.std(ThFemale)/np.sqrt(len(ThFemale))]
MusicErr = [np.std(BMusicP)/np.sqrt(len(BMusicP)),np.std(BNMusicP)/np.sqrt(len(BNMusicP)), np.std(RMusicP)/np.sqrt(len(RMusicP)),np.std(RNMusicP)/np.sqrt(len(RNMusicP)),np.std(TMusicP)/np.sqrt(len(TMusicP)),np.std(TNMusicP)/np.sqrt(len(TNMusicP)), np.std(ThMusicP)/np.sqrt(len(ThMusicP)),np.std(ThNMusicP)/np.sqrt(len(ThNMusicP))]
SportsErr = [np.std(BSport)/np.sqrt(len(BSport)),np.std(BNSport)/np.sqrt(len(BNSport)), np.std(RSport)/np.sqrt(len(RSport)),np.std(RNSport)/np.sqrt(len(RNSport)), np.std(TSport)/np.sqrt(len(TSport)),np.std(TNSport)/np.sqrt(len(TNSport)), np.std(ThSport)/np.sqrt(len(ThSport)), np.std(ThNSport)/np.sqrt(len(ThNSport))]
GamesErr = [np.std(BVGames)/np.sqrt(len(BVGames)),np.std(BNVGames)/np.sqrt(len(BNVGames)), np.std(RVGames)/np.sqrt(len(RVGames)),np.std(RNVGames)/np.sqrt(len(RNVGames)), np.std(TVGames)/np.sqrt(len(TVGames)),np.std(TNVGames)/np.sqrt(len(TNVGames)),np.std(ThVGames)/np.sqrt(len(ThVGames)),np.std(ThNVGames)/np.sqrt(len(ThNVGames))]


print(GenderErr)
print(MusicErr)
print(SportsErr)
print(GamesErr)

markerArray = ['o','o','s','s','D','D','^','^']
MFedgecolorArray = ['slateblue','hotpink','slateblue','hotpink','slateblue','hotpink','slateblue','hotpink']
HLedgecolorArray = ['orange','steelblue','orange','steelblue','orange','steelblue','orange','steelblue']
s = [15,15,15,15,15,15,17,17]

def GenderScatter(i):
    plt.scatter(GenderArray[i], 33-i, marker = markerArray[i], facecolors='none', edgecolor = MFedgecolorArray[i], s= s[i])

GenderScatter(0)
GenderScatter(1)
GenderScatter(2)
GenderScatter(3)
GenderScatter(4)
GenderScatter(5)
GenderScatter(6)
GenderScatter(7)

def Scatter(i):
    plt.scatter(GamesArray[i], 25-i, marker = markerArray[i], facecolors='none', edgecolor = HLedgecolorArray[i], s= s[i])
    plt.scatter(SportsArray[i], 17-i, marker = markerArray[i], facecolors='none', edgecolor = HLedgecolorArray[i], s= s[i])
    plt.scatter(MusicArray[i], 9-i, marker = markerArray[i], facecolors='none', edgecolor = HLedgecolorArray[i], s= s[i])

Scatter(0)
Scatter(1)
Scatter(2)
Scatter(3)
Scatter(4)
Scatter(5)
Scatter(6)
Scatter(7)

LW = 1


def GenErr(i): 
    plt.errorbar(GenderArray[i], 33-i, xerr = GenderErr[i], color = MFedgecolorArray[i], elinewidth = LW, capsize = 1, capthick = LW)

def Err(i):
    plt.errorbar(GamesArray[i], 25-i, xerr = GamesErr[i], color = HLedgecolorArray[i], elinewidth = LW, capsize = 1, capthick = LW)
    plt.errorbar(MusicArray[i], 9-i, xerr = MusicErr[i], color = HLedgecolorArray[i], elinewidth = LW, capsize = 1, capthick = LW)
    plt.errorbar(SportsArray[i], 17-i, xerr = SportsErr[i], color = HLedgecolorArray[i], elinewidth = LW, capsize = 1, capthick = LW)
   
GenErr(0)
GenErr(1)
GenErr(2)
GenErr(3)
GenErr(4)
GenErr(5)
GenErr(6)
GenErr(7)

Err(0)
Err(1)
Err(2)
Err(3)
Err(4)
Err(5)
Err(6)
Err(7)

plt.hlines(9.5, 170, 535, colors = 'gray', linestyles = 'dashed', linewidth = 0.5)
plt.hlines(17.5, 170, 535, colors = 'gray', linestyles = 'dashed', linewidth = 0.5)
plt.hlines(25.5, 170, 535, colors = 'gray', linestyles = 'dashed', linewidth = 0.5)
plt.xlim(170,535)
plt.ylim(1,34)

plt.text(GenderArray[0]+10,33,'186 ± 1', fontsize=8, verticalalignment='center')
plt.text(GenderArray[1]+10,32,'191 ± 2', fontsize=8, verticalalignment='center')
plt.text(GenderArray[2]+10,31,'212 ± 1', fontsize=8, verticalalignment='center')
plt.text(GenderArray[3]+10 ,30,'216 ± 2', fontsize=8, verticalalignment='center')
plt.text(GenderArray[4]+10 ,29,'359 ± 4 *', fontsize=8, verticalalignment='center')
plt.text(GenderArray[5]+10 ,28,'378 ± 5 *', fontsize=8, verticalalignment='center')
plt.text(GenderArray[6]+15 ,27,'465 ± 5 *', fontsize=8, verticalalignment='center')
plt.text(GenderArray[7]+15 ,26,'484 ± 7 *', fontsize=8, verticalalignment='center')

plt.text(GamesArray[0]+10 ,25,'185 ± 3 *', fontsize=8, verticalalignment='center')
plt.text(GamesArray[1]+10 ,24,'194 ± 4 *', fontsize=8, verticalalignment='center')
plt.text(GamesArray[2]+10 ,23,'211 ± 3 *', fontsize=8, verticalalignment='center')
plt.text(GamesArray[3]+10 ,22,'225 ± 4 *', fontsize=8, verticalalignment='center')
plt.text(GamesArray[4]+10 ,21,'350 ± 7 *', fontsize=8, verticalalignment='center')
plt.text(GamesArray[5]+10 ,20,'377 ± 8 *', fontsize=8, verticalalignment='center')
plt.text(GamesArray[6]+15 ,19,'450 ± 9 *', fontsize=8, verticalalignment='center')
plt.text(GamesArray[7]+15 ,18,'484 ± 11 *', fontsize=8, verticalalignment='center')

plt.text(SportsArray[0]+10 ,17,'189 ± 2', fontsize=8, verticalalignment='center')
plt.text(SportsArray[1]+10 ,16,'190 ± 2', fontsize=8, verticalalignment='center')
plt.text(SportsArray[2]+10 ,15,'215 ± 2', fontsize=8, verticalalignment='center')
plt.text(SportsArray[3]+10 ,14,'215 ± 2', fontsize=8, verticalalignment='center')
plt.text(SportsArray[4]+10 ,13,'361 ± 6', fontsize=8, verticalalignment='center')
plt.text(SportsArray[5]+10 ,12,'370 ± 6', fontsize=8, verticalalignment='center')
plt.text(SportsArray[6]+15 ,11,'470 ± 8', fontsize=8, verticalalignment='center')
plt.text(SportsArray[7]+15 ,10,'471 ± 8', fontsize=8, verticalalignment='center')

plt.text(MusicArray[0]+10 ,9,'190 ± 3', fontsize=8, verticalalignment='center')
plt.text(MusicArray[1]+10 ,8,'185 ± 2', fontsize=8, verticalalignment='center')
plt.text(MusicArray[2]+10 ,7,'215 ± 3', fontsize=8, verticalalignment='center')
plt.text(MusicArray[3]+10 ,6,'213 ± 2', fontsize=8, verticalalignment='center')
plt.text(MusicArray[4]+10 ,5,'364 ± 6', fontsize=8, verticalalignment='center')
plt.text(MusicArray[5]+10 ,4,'366 ± 5', fontsize=8, verticalalignment='center')
plt.text(MusicArray[6]+15 ,3,'468 ± 9', fontsize=8, verticalalignment='center')
plt.text(MusicArray[7]+15 ,2,'472 ± 8', fontsize=8, verticalalignment='center')


plt.xlabel('Average RT (ms)')

plt.yticks(np.arange(2,34, step=1), ('3 LED Low Music', '3 LED High Music', '2 LED Low Music', '2 LED High Music', '1 LED Low Music', '1 LED High Music', 'Audio Low Music',  'Audio High Music','3 LED Low Sports', '3 LED High Sports', '2 LED Low Sports', '2 LED High Sports', '1 LED Low Sports', '1 LED High Sports', 'Audio Low Sports',  'Audio High Sports', '3 LED Low Games', '3 LED High Games', '2 LED Low Games', '2 LED High Games', '1 LED Low Games', '1 LED High Games', 'Audio Low Games',  'Audio High Games',  '3 LED Female', '3 LED Male', '2 LED Female', '2 LED Male', '1 LED Female', '1 LED Male', 'Audio Female',  'Audio Male'))


#Gender but removing video games
RMaleNG = []
RFemaleNG = []
BMaleNG = []
BFemaleNG = []
TMaleNG = []
TFemaleNG = []
ThMaleNG = []
ThFemaleNG = []

for i in range(0,len(fRA)):
  if fGender[i] == 1 and fGames[i] != 4 and fGames[i] != 5:
    RMaleNG.append(fRA[i])
    BMaleNG.append(fBA[i])
    TMaleNG.append(fTA[i])
    ThMaleNG.append(fThA[i])

for i in range(0,len(fRA)):
  if fGender[i] == 2 and fGames[i] != 4 and fGames[i] != 5:
    RFemaleNG.append(fRA[i])
    BFemaleNG.append(fBA[i])
    TFemaleNG.append(fTA[i])
    ThFemaleNG.append(fThA[i])
      
RGendermeanNG = [np.mean(RMaleNG),np.mean(RFemaleNG)]
BGendermeanNG = [np.mean(BMaleNG),np.mean(BFemaleNG)]
TGendermeanNG = [np.mean(TMaleNG),np.mean(TFemaleNG)]
ThGendermeanNG = [np.mean(ThMaleNG),np.mean(ThFemaleNG)]

RGenderSDNG = [np.std(RMaleNG),np.std(RFemaleNG)]
BGenderSDNG = [np.std(BMaleNG),np.std(BFemaleNG)]
TGenderSDNG = [np.std(TMaleNG),np.std(TFemaleNG)]
ThGenderSDNG = [np.std(ThMaleNG),np.std(ThFemaleNG)]

print('Gender Male vs Female NO HIGH GAMES')

print('Audio RT Mean: ', BGendermeanNG)
print('1 LED RT Mean: ', RGendermeanNG)
print('2 LED RT Mean: ', TGendermeanNG)
print('3 LED RT Mean: ', ThGendermeanNG)

print('Audio RT SD: ', BGenderSDNG)
print('1 LED RT SD: ', RGenderSDNG)
print('2 LED RT SD: ', TGenderSDNG)
print('3 LED RT SD: ', ThGenderSDNG)

print('Audio RT p-value: ', stats.ttest_ind(BMaleNG, BFemaleNG)[1])
print('1 LED RT p-value: ', stats.ttest_ind(RMaleNG, RFemaleNG)[1])
print('2 LED RT p-value: ', stats.ttest_ind(TMaleNG, TFemaleNG)[1])
print('3 LED RT p-value: ', stats.ttest_ind(ThMaleNG, ThFemaleNG)[1])

print('N:', len(BMaleNG), len(BFemaleNG), len(RMaleNG), len(RFemaleNG), len(TMaleNG), len(TFemaleNG), len(ThMaleNG), len(ThFemaleNG))

