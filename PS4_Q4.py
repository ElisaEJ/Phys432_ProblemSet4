#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 22:41:40 2020

@author: Elisa
"""

'''This code will solve the Advection/Diffusion equation using the 
Lax-Friedrichs method. I use the implicit method.  The implementation of this code is greatly 
inspired from my own code written in PS4_Q3 where I solve the simpler 
case of the advection equation.  '''

import numpy as np
import matplotlib.pyplot as plt

''' In this code, I use D1 and D2 for two different diffusion
 coefficients.   ''' 

''' Define initial parameters of the problem - according to Q3 '''
u = -0.1 #according to the problem instructions, this is the speed we should use.  
D1 = .1 #diffusion coefficient 
D2 = .5

delx = 1 #spatial increment 
delt = 1 #time increment 

#Play around with the increments to see what works 

jmax = 50 #length of array
nmax = 500 #maximum number of time steps


''' Setting up the plot: '''
#define the x axis of our plot 

x = np.asarray(np.arange(jmax))*delx

#initializing the plot: 
plt.ion()
fig,axes = plt.subplots(1,2)
axes[0].set_title('D Coeff =' + str(D1))
axes[1].set_title('D Coeff =' + str(D2))


'''Plot the first time steps: '''
#initialize the array of numbers 
f1 = np.copy(x)*1./jmax #each corresponds to different coef. 
f2 = np.copy(x)*1./jmax

plt1, = axes[0].plot(x, f1, 'ro')
plt2, = axes[1].plot(x, f2, 'bo')

for ax in axes: 
    ax.set_xlim([0,jmax])
    ax.set_ylim([0,1.5])

fig.canvas.draw()

'''Evolving f1 and f2 according to the advection equation '''

n = 0 #will label the time steps 

for n in range (0,nmax):        
    #update the value of f1 and f2
    #using boundary counditions where 1st and last values don't evolve
    
    #First, update grid with the diffusion term using the implicit method
    
    A1 = np.eye(jmax)*(1.0+2.0*(D1*delt/(delx**2))) + np.eye(jmax, k=1) * (-D1*delt/(delx**2))  + np.eye(jmax, k=-1) * (-D1*delt/(delx**2)) 
    A2 = np.eye(jmax)*(1.0+2.0*(D2*delt/(delx**2))) + np.eye(jmax, k=1) * (-D2*delt/(delx**2))  + np.eye(jmax, k=-1) * (-D2*delt/(delx**2))
    
    #apply the boundary conditions on A: 
    A1[0][0] = 1 
    A1[0][1] = 0 
    A1[-1][-1] = 1
    A1[-1][-2] = 0 #ensures that we have the hard surface at both sides of the grid
    
    A2[0][0] = 1 
    A2[0][1] = 0 
    A2[-1][-1] = 1
    A2[-1][-2] = 0
    
   
    
    #now solve f1 and f2: 
    
    f1 = np.linalg.solve(A1,f1)
    f2 = np.linalg.solve(A2,f2)
    
    #then update the grid with the advection term (from Q3)
    f1[1:jmax-1] = (0.5)*(f1[2:] + f1[:jmax-2]) - (u*delt/(2*delx)) *(f1[2:] - f1[:jmax-2])
    f2[1:jmax-1] = (0.5)*(f2[2:] + f2[:jmax-2]) - (u*delt/(2*delx)) *(f2[2:] - f2[:jmax-2])
    
    #plot the updated value:
    plt1.set_ydata(f1)
    plt2.set_ydata(f2)
    
    fig.canvas.draw()
    plt.pause(0.0000001)