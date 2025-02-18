#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 11:33:04 2023

@author: kmccormack
"""

#%% Preamble

from numpy import (array, empty, ones, linspace, arange, transpose, matmul, argmax, argmin, argwhere,
                   sqrt, abs, exp, log, sin, cos, tan, arctan, arctan2, pi as π)

import math

from scipy.spatial.distance import pdist, squareform

import scipy.io as sio

import array as arr

import csv

import numpy as np

from bayesian_utilities import scatterplot_matrix, percentile

from numpy.random import default_rng
seed = 79524696  # this was selected to highlight the value of the theoretical model
my_rng = default_rng(seed)
std_norm = my_rng.standard_normal

import matplotlib.pyplot as plt

from numpy import (ndarray, array, empty, zeros, ones, ones_like, linspace, arange, hstack, meshgrid,
                   isfinite, inf, interp, quantile, maximum, ceil, sqrt, exp, log, tan, sin, cos, pi as π)

figsiz = (8, 5)

from os import listdir

#%% Generate Data

# MyPath='/Users/kmccormack/Dropbox (CSER)/Utah/FaultCharacterizationProject/UpdatedHazard/LANL_Faults'

# LD=listdir(MyPath)
# D='.DS_Store'
# while(D in LD):
#     LD.remove(D)
LD='/Users/kmccormack/CSER Dropbox/Kevin McCormack/Utah/FaultCharacterizationProject/UpdatedHazard/Paper/Depth_Migrated_MedianFilter_DipDeviation_AntTrack 31'

ele=7580 #feet above sea level

xdim = 51
ydim = 50
CFF_byFault=zeros((1,xdim,ydim))
Coords_byFault=zeros((1,1000,3))
X1i_byFault=zeros((1,xdim,ydim))
X2i_byFault=zeros((1,xdim,ydim))
yi_byFault=zeros((1,xdim*ydim,1))

ZM_byFault=zeros((1))


    
    
with open(LD) as f:
    
    rFault=f.readlines()

Fault=rFault[9:-1]
f.close()




XF = zeros((len(Fault),1))
YF = zeros((len(Fault),1))
ZF = zeros((len(Fault),1))


for i in range(0,len(Fault)):

    XF[i,0]=Fault[i].split(' ')[0]
    YF[i,0]=Fault[i].split(' ')[1]
    ZF[i,0]=Fault[i].split(' ')[2]
        
    
    
    # XX=np.zeros((len(Fault)))
    # YY=np.zeros((len(Fault)))
    # #% Try making the data variables for the Kriging
    # for j in range(0,len(Fault)):
    #     XX[j] = XF[j] - np.mean(XF)
    #     YY[j] = YF[j] - np.mean(YF)
    
    XX = XF - np.mean(XF)
    YY = YF - np.mean(YF)
    
    test_xy=np.concatenate((XX, YY),1)
    test_z = ZF+ele # Datum to switch away from TVDss
    
    
    
    
    
#% Krige


from pyregress import Noise, SquareExp, LogNormal, GPI


K = Noise(w=LogNormal(guess=.3, std=0.25)) + SquareExp(w=300, l=[LogNormal(guess=.3, std=0.25),
                                                                   LogNormal(guess=.3, std=0.25)])

D = pdist(test_xy)
D = squareform(D)
N, [I_row, I_col] = np.nanmax(D), np.unravel_index( np.argmax(D), D.shape)

# theta=-np.arctan2(abs(test_xy[I_row,1]-test_xy[I_col,1]),abs(test_xy[I_row,0]-test_xy[I_col,0]))
theta=-np.arctan2(abs(test_xy[I_row,0]-test_xy[I_col,0]),abs(test_xy[I_row,1]-test_xy[I_col,1]))

rot=np.array([[np.cos(theta), -np.sin(theta)], 
                     [np.sin(theta),  np.cos(theta)]])

KrigXY=np.zeros((len(test_xy),2))
for i in range(0,len(test_xy)):
    KrigXY[i,:]=matmul(rot,test_xy[i,:])

# fig = plt.figure()
# plt.scatter(KrigXY[:,0], KrigXY[:,1], s=20)

mxxK=np.max(KrigXY[:,0])
mnxK=np.min(KrigXY[:,0])
mxyK=np.max(KrigXY[:,1])
mnyK=np.min(KrigXY[:,1])

# K = Noise(w=100) + SquareExp(w=100, l=.3)
testGPI = GPI(KrigXY, test_z, K, explicit_basis=[0, 1, 2], Xscaling='range', optimize='v')
print('Optimized value of the hyper-parameters:', testGPI.kernel.get_φ())


# x1i = linspace(int(min(test_xy[:,0])), int(max(test_xy[:,0])), xdim) 
# x2i = linspace(int(min(test_xy[:,1])), int(max(test_xy[:,1])), ydim)     
x1i = linspace(int(mnxK), int(mxxK), xdim) 
x2i = linspace(int(mnyK), int(mxyK), ydim)     

X1i, X2i = meshgrid(x1i, x2i, indexing='ij')
Xi = hstack([X1i.reshape((-1,1)), X2i.reshape((-1,1))])
yi, Yi_grad = testGPI(Xi, grad=True)
yi_grad = Yi_grad.reshape(X1i.shape + (2,))

# fig = plt.figure(figsize=figsiz)
# ax = fig.add_subplot(projection='3d')
# ax.view_init(10,150)
# # ax.scatter(test_xy[:,0], test_xy[:,1], -test_z[:,0], s=20)
# ax.scatter(KrigXY[:,0], KrigXY[:,1], -test_z[:,0], s=20)
# ax.plot_surface(X1i, X2i, -yi.reshape(X1i.shape), alpha=.6, color='red')
# ax.set_title('Data & Kriged Surface')

rot_return=np.array([[np.cos(-theta), -np.sin(-theta)], 
                     [np.sin(-theta),  np.cos(-theta)]])

rot_krig=np.array([X1i.reshape(xdim*ydim,1),X2i.reshape(xdim*ydim,1)])

KrigXY_return=np.zeros((xdim*ydim,2))
for i in range(0,xdim*ydim):
    KrigXY_return[i,:]=matmul(rot_return,rot_krig[:,i,0])

SurfX_return=KrigXY_return[:,0].reshape(xdim,ydim)
SurfY_return=KrigXY_return[:,1].reshape(xdim,ydim)

yi_grad_return=np.zeros((xdim,ydim,2))
for i in range(0,xdim):
    for j in range(0,ydim):
        yi_grad_return[i,j,:]=matmul(rot_return,yi_grad[i,j,:])
 

fig = plt.figure(figsize=figsiz)
ax = fig.add_subplot(projection='3d')
ax.view_init(10,150)
# ax.scatter(test_xy[:,0], test_xy[:,1], -test_z[:,0], s=20)
ax.scatter(test_xy[:,0], test_xy[:,1], -test_z[:,0], s=20)
ax.plot_surface(SurfX_return, SurfY_return, -yi.reshape(X1i.shape), alpha=.6, color='red')
ax.set_title('Data & Kriged Surface')

strike=zeros((xdim,ydim))
dip = zeros((xdim,ydim))
#% Get Tangent Data
for kk in range(0,xdim):
    for kkk in range(0,ydim):
        # if yi[ydim*kk+kkk] > 7287 and yi[ydim*kk+kkk] < 13384:
        if yi[ydim*kk+kkk] > 2000 and yi[ydim*kk+kkk] < 13384:
            strike[kk,kkk] = (arctan2(-yi_grad_return[kk,kkk,1] , yi_grad_return[kk,kkk,0]) * 180 / π) #Check to make sure the strike is coming in correctly
           # print('Strike min and max: ', strike.min(), strike.max())
            
            dip[kk,kkk]    = arctan(sqrt(yi_grad_return[kk,kkk,0]**2 + yi_grad_return[kk,kkk,1]**2)) * 180 / π
           # print('Dip min and max   : ', dip.min(), dip.max())

# fig = plt.figure(figsize=figsiz)
# ax = fig.add_subplot(projection='3d')
# ax.plot_surface(X1i, X2i, strike, alpha=.6, color='purple')
# ax.plot_surface(X1i, X2i, dip, alpha=.6, color='yellow')
# ax.set_title('Strike and Dip')


#% Define GeoMech model

def Geomech(a, b, c, S1, S2, S3, Pp, mu, s, d, depth):
    dip = math.radians(d)
    strike = math.radians(s)
    S =    array([[S1, 0, 0],
                  [0, S2, 0],
                  [0, 0, S3]], dtype=object)
    
    R1=   array([[cos(a)*cos(b), sin(a)*cos(b), -sin(b)],
                 [cos(a)*sin(b)*sin(c)-sin(a)*cos(c), sin(a)*sin(b)*sin(c)+cos(a)*cos(c), cos(b)*sin(c)],
                 [cos(a)*sin(b)*cos(c)+sin(a)*sin(c), sin(a)*sin(b)*cos(c)-cos(a)*sin(c), cos(b)*cos(c)]],dtype=object)
    
    Sg=matmul(matmul(transpose(R1),S),R1)
    
    R2=   array([[cos(strike), sin(strike), 0],
                 [sin(strike)*cos(dip), -cos(strike)*cos(dip), -sin(dip)],
                 [-sin(strike)*sin(dip), cos(strike)*sin(dip), -cos(dip)]], dtype=object)
    
    Sf=matmul(matmul(R2,Sg),transpose(R2))
    
    Sn=Sf[2,2]
    
    if Sf[2,1] > 0:
        rake=arctan(Sf[2,1]/Sf[2,0])
    elif Sf[2,1] < 0 and Sf[2,0] > 0:
        rake=3.141592653589793-arctan(Sf[2,1]/(-Sf[2,0]))
    else:
        rake=arctan(-Sf[2,1]/(-Sf[2,0]))-3.141592653589793
        
    R3=   array([[cos(rake), sin(rake), 0],
                 [-sin(rake),cos(rake),0],
                 [0,0,1]], dtype=object)
    
    Sr=matmul(matmul(R3,Sf),transpose(R3))
    
    tau=Sr[2,0]
    Sn_eff=Sn-Pp
    CFF=abs(tau)-mu*Sn_eff
    
    return CFF





#% Inputs for the GeoMch Model

# Generate Stresses (SCITS)

TomDat=sio.loadmat('/Users/kmccormack/CSER Dropbox/Kevin McCormack/Utah/FAultCharacterizationProject/UpdatedHazard/StressForTom/Data.mat')

Aphi = 1.07
TomPp = TomDat["Data"][:,28]
TomSv = TomDat["Data"][:,25]
TomMu = TomDat["Data"][:,15]
TomDEP = TomDat["Data"][:,0]
SHmax_azi = 35
#!!!!!!!
# a = SHmax_azi+270
# b = -0
# c = 90
#!!!!!!!

TomMu_=((np.tan(TomMu*np.pi/180)**2+1)**(1/2)+np.tan(TomMu*np.pi/180))**2

S1_SCITS=((1/(2-Aphi)*TomSv-(1/(2-Aphi))*TomPp+TomPp+TomPp*(1/(TomMu_*(2-Aphi))-1/TomMu_))/(1+1/(TomMu_*(2-Aphi))-1/TomMu_))
S3_SCITS=(S1_SCITS-TomPp)/TomMu_+TomPp


# EarthM = np.empty((17800,5))
# cnt1=-1
# with open("/Users/kmccormack/Dropbox (CSER)/Utah/FaultCharacterizationProject/UpdatedHazard/TableTrim.csv") as f1:
#     reader1 = csv.reader(f1)
#     for row in reader1:
#        cnt1+=1
#        EarthM[cnt1,]=[float(iiii) for iiii in row]

# DEP = EarthM[:,0]
# sDEP=arr.array('f',DEP)

sTomDEP = arr.array('f',TomDEP)

n_sp = 1000
coulff = zeros([len(x1i),len(x2i),n_sp])
coulff_max = strike * 0
coulff_min = strike * 0
coul_samp_ij = zeros([len(x1i), len(x2i), n_sp])
cff_dist_one = zeros([n_sp])

for kk in range(0,xdim):
    for kkk in range(0,ydim):
        if strike[kk,kkk] != 0:

            Zref=np.round(yi[ydim*kk+kkk])
            # Geomechanics Model - These need to become depth-dependent
            a_ = math.radians(35) #305
            b  = math.radians(0)  #-90
            c  = math.radians(90) #0  
            S1_= S1_SCITS[sTomDEP.index(Zref)]
            S2_= TomSv[sTomDEP.index(Zref)]
            S3_= S3_SCITS[sTomDEP.index(Zref)]
            μ_ = np.tan(TomMu[sTomDEP.index(Zref)]*np.pi/180)
            Pp_= TomPp[sTomDEP.index(Zref)]
            
            for kkkk in range(0,n_sp):
                ### sampling the geomechanical model
                # n_sp = 300
                # a=np.random.normal(math.radians(305), math.radians(10), n_sp)
                a  = a_ + math.radians(10) * std_norm(n_sp)
                # S1=np.random.normal(9087, 363, n_sp)
                S1 = S1_ + S1_*0.1 * std_norm(n_sp)
                # S2=np.random.normal(5580, 558, n_sp)
                S2 = S2_ + S2_*0.1 * std_norm(n_sp)
                # S3=np.random.normal(5242, 524, n_sp)
                S3 = S3_ + S3_*0.1 * std_norm(n_sp)
                # mu=np.random.normal(0.6, 0.09, n_sp)
                μ = μ_ + μ_*0.1 * std_norm(n_sp)
                # Pp=np.random.normal(3502, 140, n_sp)
                Pp = Pp_ + Pp_*0.1 * std_norm(n_sp)
                
                strike_ = strike[kk,kkk] + strike[kk,kkk]*0.1 * std_norm(n_sp)
                dip_ = dip[kk,kkk] + dip[kk,kkk]*0.1 * std_norm(n_sp)
                
                
                if S1[kkkk] > S2[kkkk]:
                
                    coulff[kk,kkk,kkkk] = Geomech(a[kkkk], b, c, S1[kkkk], S2[kkkk], S3[kkkk], Pp[kkkk], μ[kkkk], strike_[kkkk], dip_[kkkk], yi.reshape(X1i.shape)[kk,kkk])
                else:
                    coulff[kk,kkk,kkkk] = Geomech(305, -90, 0, S2[kkkk], S1[kkkk], S3[kkkk], Pp[kkkk], μ[kkkk], strike_[kkkk], dip_[kkkk], yi.reshape(X1i.shape)[kk,kkk])
    print(kk)



KrigDep=yi.reshape(xdim,ydim)

OutputForMATLAB={}
OutputForMATLAB['FaultXY']=test_xy
OutputForMATLAB['FaultZ']=test_z
OutputForMATLAB['KrigXY']=KrigXY_return
OutputForMATLAB['KrigZ']=KrigDep
OutputForMATLAB['CFF']=coulff



# ={FaultXY: test_xy, FaultZ: test_z, KrigXY: KrigXY_return, KrigZ: KrigDep, CFF: coulff}

sio.savemat('/Users/kmccormack/CSER Dropbox/Kevin McCormack/Utah/FaultCharacterizationProject/UpdatedHazard/Paper/MATLAB_Input_SCITS.mat',OutputForMATLAB)

# if np.sum(np.sum(coulff,0),0) == 0:
#     continue

# Rdep = round(test_z)

# ZM = np.round(np.mean(test_z))
# ZM_byFault[ii]=ZM

# # for i in range(0,len(Fault)):
# #     if ZF[i] > 7287 and ZF[i] < 13384:
        
# # if ZM < 7287 or ZM > 13384:
# #     continue                
        
        
# # Geomechanics Model - These need to become depth-dependent
# a_ = math.radians(305)
# b  = math.radians(-90)
# c  = math.radians(0)
# S1_= EarthM[sDEP.index(ZM),1]
# S2_= EarthM[sDEP.index(ZM),4]*ZM
# S3_= EarthM[sDEP.index(ZM),3]
# μ_ = 0.6
# Pp_= EarthM[sDEP.index(ZM),2]

# ### sampling the geomechanical model
# n_sp = 300
# # a=np.random.normal(math.radians(305), math.radians(10), n_sp)
# a  = a_ + math.radians(10) * std_norm(n_sp)
# # S1=np.random.normal(9087, 363, n_sp)
# S1 = S1_ + 363 * std_norm(n_sp)
# # S2=np.random.normal(5580, 558, n_sp)
# S2 = S2_ + 558 * std_norm(n_sp)
# # S3=np.random.normal(5242, 524, n_sp)
# S3 = S3_ + 524 * std_norm(n_sp)
# # mu=np.random.normal(0.6, 0.09, n_sp)
# μ = μ_ + 0.09 * std_norm(n_sp)
# # Pp=np.random.normal(3502, 140, n_sp)
# Pp = Pp_ + 140 * std_norm(n_sp)

#-----------------#

# #% Create CFF dist for each Kriged Point


# # for i in range(len(x1i)):
# #     for j in range(len(x2i)):

# samp_4 = argmax(coulff.reshape(-1, 1))
# # samp_4 = argmin(coulff.reshape(-1, 1))
# # samp_4 = 255

# samp_4_pair = argwhere(coulff == coulff.reshape(-1, 1)[samp_4])
# #print('Maximum cff for kriged surface @:', samp_4,'& ', samp_4_pair, ' is ', coulff.reshape(-1, 1)[samp_4])

# for i in range(len(x1i)):
#     for j in range(len(x2i)):
#         for n in range(n_sp):
#             coul_samp_ij[i,j,n] = Geomech(a[n], b, c, S1[n], S2[n], S3[n], Pp[n], μ[n], strike[i,j], dip[i,j], yi.reshape(X1i.shape)[i,j])
#             if i == samp_4_pair[0][0] and j == samp_4_pair[0][1]:
#                 cff_dist_one[n] = coul_samp_ij[i,j,n]
#         coulff_max[i,j] = percentile(97.5, coul_samp_ij[i,j,:], None)
#         coulff_min[i,j] = percentile(2.5, coul_samp_ij[i,j,:], None)
# cff_dist = coul_samp_ij.reshape((-1))

# CFF_byFault[1,:,:]=coulff
# Coords_byFault[1,0:len(XF),:]=np.concatenate((XF,YF,ZF),1)
# X1i_byFault[1,:,:]=X1i+np.mean(XF)
# X2i_byFault[1,:,:]=X2i+np.mean(YF)
# yi_byFault[1,:,:]=yi

# # with open('/Users/kmccormack/Dropbox (CSER)/Utah/FaultCharacterizationProject/UpdatedHazard/ExportLANLNoZero/Fault'+str(ii)+'.csv','w') as csvfile:
# #     CF=csv.writer(csvfile)
# #     for i in range(0,xdim*ydim):
# #         if coulff.reshape(xdim*ydim)[i] != 0:
# #             CF.writerow([SurfX_return.reshape(xdim*ydim)[i]+np.mean(XF), SurfY_return.reshape(xdim*ydim)[i]+np.mean(YF), float(yi[i])-ele, coulff.reshape(xdim*ydim)[i]])
    
#--------------#


# #%% Plot the whole field
# fig = plt.figure(figsize=figsiz)
# ax = fig.add_subplot(projection='3d')

# minCFF=np.min(np.min(np.min(CFF_byFault,0),0),0) #taking out the crazy fault
# maxCFF=np.max(np.max(np.max(CFF_byFault,0),0),0)


# yi_forPlot=zeros((len(LD),xdim,ydim))

# for ii in range(0,len(LD)):
#     if CFF_byFault[ii,0,0] != 0:# and ii != 31  and ii != 5:# and ii != 27: #there's a ridiculous fault on position 31 and 5; 27 is "over-kriged"
#         yi_forPlot[ii,:,:]=yi_byFault[ii,:,:].reshape(X1i.shape)
#         for jj in range(0,51):
#             for kk in range(0,50):
#                 if CFF_byFault[ii,jj,kk] != 0:
#                     col=(CFF_byFault[ii,jj,kk]-minCFF)/(maxCFF-minCFF)
#                     ax.scatter3D(X1i_byFault[ii,jj,kk], X2i_byFault[ii,jj,kk], yi_forPlot[ii,jj,kk], alpha=.8, facecolors=[[col, 1-col, 0]])
# # ax.plot_surface(X1i, X2i, coulff_max, alpha=.3, color='blue')
# # ax.plot_surface(X1i, X2i, coulff_min, alpha=.3, color='blue')
#     print(ii)
    
# ax.invert_zaxis()
# ax.view_init(20, 325)
# ax.set_zlabel('Depth (ft)')
# ax.set_xlabel('X (ft)')
# ax.set_ylabel('Y (ft)')

# plt.savefig('Faults_wCFF2.pdf')