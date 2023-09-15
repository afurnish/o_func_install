# -*- coding: utf-8 -*-
""" Multiprocessing Functions

Created on Thu Feb  2 14:54:05 2023
@author: aafur
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
#attempt at a debug
## doesnt like being called func_something, so has been renamed w_d which worked before
def w_d(i,n,tim,name_number_zero,xxx,yyy,wd,UKWEST,df_time,figurepath,bounds):
#    for i in range(0, len(wd)):
       # for i in range(0,25):    
           #new change to stop matplotlib hogging memory
            plt.rcParams['image.caching'] = 'none'

            
            name = n + tim + str(name_number_zero[i]) + ".png"
            fig5, ax = plt.subplots()
            fig5.set_figheight(22)
            fig5.set_figwidth(10)
            plt.tricontourf(xxx, yyy, \
                wd[i],levels= np.linspace( bounds[0][0],bounds[0][1],bounds[0][2] \
                ),cmap=cm.jet,extend='both' )
            cbar= plt.colorbar()
            UKWEST.plot(ax=ax, color="white")
            loc = np.linspace( bounds[1][0],bounds[1][1],bounds[1][2] )
            cbar.set_ticks(loc)
            cbar.set_label("water depth (m)", labelpad=+1)
            #plt.clim(0,65)
            ax.set_xlim( bounds[2][0],bounds[2][1] )
            ax.set_ylim( bounds[3][0],bounds[3][1] )
           
            plt.title('timestep = ' + str(df_time['d_index'][i]) )
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
           
      
           
            fig5.savefig(figurepath + name, bbox_inches='tight')
            #fig5.clear()   # for computaional effectiveness clear the figure from memory
            #plt.close()
            
            #changes to speed it up 
            plt.close(fig5)
            plt.close('all')
            del fig5, ax, cbar  # Delete the variables to free up memory

            
def interpolate(val,ukc3,sh,tree,df_primea,p,z_interpolated_time):
    
    ukc3_sh = ukc3.sossheig[val].values
    stacked = np.dstack([ukc3_sh.ravel()])[0]
    z_interpolated = []
    for i in range((sh.values.shape[1])):
        distances, indices = tree.query([df_primea.x[i], df_primea.y[i]], k=4)
        weights = 1 / distances**p
        z_interpolated.append(np.sum(weights * np.array(stacked)[indices]) / np.sum(weights))
    z_interpolated = np.array(z_interpolated)
    
    z_interpolated = np.array(z_interpolated)
    #z_interpolated_time = np.vstack((z_interpolated_time, np.expand_dims(z_interpolated, axis=0)))
    z_interpolated_time[val,:] = z_interpolated
              