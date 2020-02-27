#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 11:54:08 2020

@author: Elisa

Fluids Course 
Solution to Problem #3 in Assignment 4 
"""

import numpy as np
import matplotlib.pyplot as plt

'''All variables with "1": 
        Solve the advection equation using the FTCT Method 
        
   All variables with "2": 
       Solve the Lax-Friedrichs method'''


''' Define initial parameters of the problem    '''
u = -0.1 #according to the problem instructions, this is the speed we should use.  

delx = 1 #spatial increment 
delt = 1 #time increment 

#Play around with the increments to see what works 

jmax = 50 #length of array
nmax = 500 #maximum number of time steps


''' Setting up the plot: '''
#define the x axis of our plot 
x1 = np.asarray(np.arange(jmax))*delx
x2 = np.asarray(np.arange(jmax))*delx

#initializing the plot: 
plt.ion()
fig,axes = plt.subplots(1,2)
axes[0].set_title('FTCS')
axes[1].set_title('Lax-Friedrichs')


'''Plot the first time steps: '''
#initialize the array of numbers 
f1 = np.copy(x1)*1./jmax
f2 = np.copy(x1)*1./jmax

plt1, = axes[0].plot(x1, f1, 'ro')
plt2, = axes[1].plot(x2, f2, 'bo')

for ax in axes: 
    ax.set_xlim([0,jmax])
    ax.set_ylim([0,2])

fig.canvas.draw()

'''Evolving f1 and f2 according to the advection equation '''

n = 0 #will label the time steps 

for n in range (0,nmax):        
    #update the value of f1 and f2
    #using boundary counditions where 1st and last values don't evolve
    
    #FTCS: 
    f1[1:jmax-1] = f1[1:jmax-1] - (u*delt/(2*delx)) *(f1[2:] - f1[:jmax-2])
    
    #Lax-Friedrich:
    f2[1:jmax-1] = (0.5)*(f2[2:] + f2[:jmax-2]) - (u*delt/(2*delx)) *(f2[2:] - f2[:jmax-2])
    
    #plot the updated value:
    plt1.set_ydata(f1)
    plt2.set_ydata(f2)
    
    fig.canvas.draw()
    plt.pause(0.00001)
    
    

#