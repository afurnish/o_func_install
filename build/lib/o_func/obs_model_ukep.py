###############################################################################
# Generic obs vs. model comparison code for UKEP plotting routines
###############################################################################
import numpy as np
import datetime as dt
from scipy import stats
import matplotlib.pyplot as plt
from def_plot_vars import *
from plot_ukep import *
from read_ukep_data import set_time, find_xy_coord
from matplotlib import dates
import matplotlib.dates as mdates
import pandas as pd
import itertools
from matplotlib.ticker import FuncFormatter

###############################################################################
## Calculate 'scatter index' statistic
def calc_sind(diff, obs, model):

    if np.var(obs) == 0:
        si = -9999.
    else:
        si = np.nanvar(model) / np.nanvar(obs)
##  si = ( np.nanvar(diff) / np.nanvar(obs) )**0.5
    if (si == 0.0): si=1.e-15
    return si

###############################################################################
## Calculate rmse statistic
def calc_rmse(diff, obs, model):

    rmse = (np.nansum((diff)**2)/np.size(np.where(np.isfinite(diff))))**0.5
    if (rmse == 0.0): rmse=1.e-15
    return rmse

###############################################################################
## Calculate bias statistic
def calc_bias(diff, obs, model):
   
    bias = np.nanmean(diff)          
    if (bias == 0.0): bias=1.e-15
    return bias

###############################################################################
## Calculate normalised bias statistic
def calc_nbias(diff, obs, model):
    
    nbias = np.nanmean(diff)/np.nanmean(obs)          
    if (nbias == 0.0 or np.nanmean(obs) == 0.0): nbias=1.e-15
    return nbias

## Calculate Nash-Sutcliffe score
def calc_ns(diff, obs, mod):

    obsmean=np.nanmean(obs)
    num=np.nansum((obs-mod)**2)
    den=np.nansum((obs-obsmean)**2)
    if den != 0.0: 
        ns=1.0-num/den
    else:
        ns=-999999.

    return ns

############################################################################## 
############################################################################## 
## Write timeseries data to text file
def write_data_to_file(outname, model_time, obs_m_data, out, val_len, model):
    
    print(('WRITING DATA TO FILE: ', outname+'.txt'))
    dmt_out=[]
    # convert from dt to decimal year
    for dmt in model_time: ##.astype(dt.datetime):
        year_part = float((dmt - dt.datetime(dmt.year, month=1, day=1)).days) + dmt.hour/24.
        year_length = float((dt.datetime(dmt.year+1, month=1, day=1) - dt.datetime(dmt.year, month=1, day=1)).days)
        dmt_out.append(dmt.year + float(year_part/year_length))
    # write output to file (>75% data availability)
    if np.size(dmt_out) > val_len:
        # write header
        hdr="  TIME  |   OBS    |"+str(model)
        # write data
        txtarr = np.around(np.vstack([dmt_out, obs_m_data, out.transpose()]).transpose(), decimals=5)
        np.savetxt(outname+'.txt',txtarr,delimiter=' ',fmt="%10.5f",header=hdr) 

    return

############################################################################## 
############################################################################## 
def to_percent(y, position):

   s = str(100 * y)
   return s + '%'
#   if matplotlib.rcParams['text.usetex'] is True:
#           return s + r'$\%$'
#   else:
#      return s + '%'

############################################################################## 
##############################################################################
# Collapse time series to mean diurnal cycle
def to_diurnal(intime, indata):
    
    dseries = pd.Series(indata, index=intime).to_frame()
    dseries['TimeB'] = dseries.index.map(lambda x: x.strftime("%H:%M"))
    diurnal_data = dseries[1:].groupby('TimeB').mean()
    diurnal_data.index = pd.to_datetime(diurnal_data.index.astype(str))

    return diurnal_data

############################################################################## 
##############################################################################
# Run Doodson filter on hourly data
def DoodsonX0(series):


    ## Doodson filter
    x0filt = np.array([1,0,1,0,0,1,0,1,1,0,2,0,1,1,0,2,1,1,2,0,2,1,1,2,0,1,1,0,2,0,1,1,0,1,0,0,1,0,1]) / 30.

    detided = np.zeros(len(series))
    srg = series - np.nanmean(series)

    for lp in range(19,len(srg)-19):
            detided[lp] = np.sum( srg[lp-19:lp+20] * x0filt )

    # persist tail series
    detided[0:19] = np.nan 
    detided[len(srg)-19:] = np.nan 

    ### Demerliac filter
    x0filt = np.array([1,3,8,15,21,32,45,55,72,91,105,128,153,171,200,231,253,288,325,351,392,435,465,512,558,586,624,658,678,704,726,738,752,762,766,768,766,762,752,738,726,704,678,658,624,586,558,512,465,435,392,351,325,288,253,231,200,171,153,128,105,91,72,55,45,32,21,15,8,3,1])/24576.0
   
    detided = np.zeros(len(series))
    srg = series - np.nanmean(series)
    for lp in range(35,len(srg)-35):
         detided[lp] = np.sum( srg[lp-35:lp+36] * x0filt )
    
    detided[0:35] = np.nan
    detided[len(srg)-35:] = np.nan

    return detided

############################################################################
# Compute tidal maxima/minima - e.g. for determining skew surge
def max_tide(sshtime,ssh,tide_max=False):

    pkhw_tide=np.array([])
    pkhw_tide_time=[]
    
    winwidth=11
    halfwin = np.int(winwidth / 2)
    lptide=halfwin

    while lptide <= len(ssh)-halfwin:
        
        take_point=False
#        if min and ssh[lptide] == np.nanmin(ssh[lptide-halfwin:lptide+halfwin]): 
#            take_point=True
        if tide_max: 
            if ssh[lptide] == np.nanmax(ssh[lptide-halfwin:lptide+halfwin]): 
                take_point=True  
        else:
            if (ssh[lptide] == np.min(ssh[lptide-halfwin:lptide+halfwin])) or (ssh[lptide] == np.max(ssh[lptide-halfwin:lptide+halfwin])): take_point=True     
            
        if take_point:
            pkhw_tide = np.append(pkhw_tide, ssh[lptide])
            pkhw_tide_time.append(sshtime[lptide])
            lptide = lptide + halfwin

            #print('TP: ', pkhw_tide_time[-1], pkhw_tide[-1])

        else:
            lptide = lptide + 1
            
    return pkhw_tide_time, pkhw_tide

############################################################################## 
############################################################################## 
## Compute categorical statistics (hit, miss, FA etc)
def categorical_stats(self,all_obs,all_data):
    
    # define observed (visibility) thresholds
    thresh_values = [10000.,5000.,1000.,200.,100.,50.]

    print('CALCULATING CATEGORICAL STATS')
    print(('Writing to file: ', self.savename+'_CatStats_'+self.zoom+'.txt'))
    f = open(self.savename+'_CatStats_'+self.zoom+'.txt','w')
    for i,suite in enumerate(self.suite):
        master_obs = np.ravel(all_obs[:])
        master_mod = np.ravel(all_data[i][:])
        flatmod = np.array(list(itertools.chain.from_iterable(all_data[i])), dtype=float)
        flatobs = np.array(list(itertools.chain.from_iterable(all_obs)), dtype=float)
        flatmod = flatmod + flatobs   # remove 'MOD - OBS'        
        
        val=np.where(flatobs >= 0.)[0]
        f.write(self.model[i])
        for thresh in thresh_values:
            hits = len(np.where((flatobs[val] <= thresh) & (flatmod[val] <= thresh))[0])
            miss = len(np.where((flatobs[val] <= thresh) & (flatmod[val] > thresh))[0])
            falm = len(np.where((flatobs[val] > thresh) & (flatmod[val] <= thresh))[0])
            rejec = len(np.where((flatobs[val] > thresh) & (flatmod[val] > thresh))[0])
            tot = hits + miss + falm + rejec
            
            pod =-999.99; far=-999.99
            if hits+miss > 0: pod=float(hits)/float(hits+miss)
            if hits+falm > 0: far=float(falm)/float(hits+falm)
            print(('*** ', self.model[i], ' THRESH: ', thresh, ' ***', 'POD = {:.2f}'.format(pod), ' FAR = {:.2f}'.format(far), ' N = ',str(hits+miss), ' [',str(hits+falm),']'))
            print((hits, miss, falm, rejec, tot))
            f.write(' THRESH: '+str(thresh)+' ***'+' POD = {:.2f}'.format(pod)+' FAR = {:.2f}'.format(far)+' N = '+str(hits+miss)+' ['+str(hits+falm)+'] \n')
        f.write('********** \n')       
            
############################################################################## 
############################################################################## 
## Plot combined plot of all lines
def plot_combined(self, aow, all_time, all_obs, all_data, xlabel, ylabel):
    
    fx=9
    if self.nday > 15: fx=15
    print((self.savename))
    allmean=[]
    allstd=[]
    allrms=[]
    for i,suite in enumerate(self.suite):
        allmean.append([])
        allstd.append([])
        allrms.append([])
        plt.figure(figsize=(fx,5))
        for r in range(0,len(all_time)):
            plt.plot(all_time[r],all_data[i][r],color=self.col[i])
        plt.title(self.label[i]+'      ['+self.zoom+']', fontsize=20)
        plt.xlabel('Date '+xlabel, fontsize=20)
        plt.ylabel(ylabel, fontsize=20)
        plt.tick_params(axis='both', which='major', labelsize=18)        
        save_close_show(self.savename+'_ALL_'+self.model[i]+'_'+self.zoom) 

        master_obs = np.ravel(all_obs[:])
        master_mod = np.ravel(all_data[i][:])
        flatmod = np.array(list(itertools.chain.from_iterable(all_data[i])), dtype=float)
        flatobs = np.array(list(itertools.chain.from_iterable(all_obs)), dtype=float)
        flatmod = flatmod + flatobs   # remove 'MOD - OBS'
        plt.figure(figsize=(8,5))
        plt.hexbin(flatobs, flatmod, gridsize=[100,100], mincnt=1)
        plt.grid()
        plt.colorbar()
        plt.axis('equal')
        ymin = np.floor(np.nanmin([np.nanmin(flatobs),np.nanmin(flatmod)]))
        ymax = np.ceil(np.nanmax([np.nanmax(flatobs),np.nanmax(flatmod)]))
        plt.plot([ymin,ymax],[ymin,ymax],'k--')
        plt.ylim(ymin,ymax); plt.xlim(ymin,ymax)
        plt.ylabel('Model output '+ylabel+' '+self.model[i], fontsize=20)
        plt.xlabel('Observations '+ylabel, fontsize=20)
        plt.yscale('log')
        plt.xscale('log')

        slope, inter, r_val, p_val, stderr = stats.linregress(flatobs, flatmod)
        rmse = calc_rmse(flatmod-flatobs, flatobs, flatmod)
        bias = calc_bias(flatmod-flatobs, flatobs, flatmod)
        tx = ymin+0.01*(ymax-ymin); ty =ymin+0.85*(ymax-ymin)
        #plt.text(tx,ty,'r2={:.2f}'.format(r_val**2),fontsize=18)
        tx = ymin+0.01*(ymax-ymin); ty =ymin+0.75*(ymax-ymin)
        #plt.text(tx,ty,'RMSE={:.2f}'.format(rmse),fontsize=18)
        tx = ymin+0.01*(ymax-ymin); ty =ymin+0.65*(ymax-ymin)
        #plt.text(tx,ty,'Bias={:.2f}'.format(bias),fontsize=18)
        tx = ymin+0.01*(ymax-ymin); ty =ymin+0.95*(ymax-ymin)
        #plt.text(tx,ty,self.model[i],fontsize=18)
        
        save_close_show(self.savename+'_allSCATlog_'+self.model[i]+'_'+self.zoom)

        # use pandas library to sort timeseries
        smean=[]; smax=[]; smin=[]; s_new=[]; sstd=[]; s95p=[]; srms=[]
        for r in range(0,len(all_time)):
            series=pd.Series(all_data[i][r],index=all_time[r])
            series=series.groupby(series.index).first()
            s_new.append(series)
        frame=pd.DataFrame(s_new)
        all_s=frame.T.sort_index(ascending=True)
        val=all_s.values
#####        val[np.abs(val)>10.]=np.nan      #### QC....
        for me in range(0,len(val)):
            smean.append(np.nanmean(val[me]))
            smax.append(np.nanmax(val[me]))
            smin.append(np.nanmin(val[me]))
            sstd.append(np.nanstd(val[me]))
            s95p.append(np.percentile(val[me],95))
            srms.append(np.nanmean(val[me]**2.)**0.5)
            
        time=all_s.index.values
        zero=np.zeros(len(smean))

        plt.figure(figsize=(fx,5))
        plt.title(self.label[i]+'      ['+self.zoom+']', fontsize=20)
        plt.plot(time,smean,color=self.col[i],linestyle='--',linewidth=2,label='MODEL-OBS')
        plt.plot(time,smin,color=self.col[i])
        plt.plot(time,smax,color=self.col[i])
        #plt.fill_between(time,smax,smin,facecolor=self.col[i],alpha=0.2,linewidth=0.1,interpolate=True)
        plt.plot(time,zero,'k',linestyle='--')
        plt.xlabel('Date '+xlabel, fontsize=20)
        plt.ylabel(ylabel,fontsize=20)
        plt.ylim(-5.,5.)
        save_close_show(self.savename+'_MMM_'+self.model[i]+'_'+self.zoom)

        plt.figure(figsize=(fx,5))
        plt.title(self.label[i]+'      ['+self.zoom+']', fontsize=20)
        plt.plot(time,smean,color=self.col[i],linestyle='-',linewidth=3,label='MODEL-OBS')
        s1=np.asarray(smean)-np.asarray(sstd)
        s2=np.asarray(smean)+np.asarray(sstd)
        plt.plot(time,s1,color=self.col[i])
        plt.plot(time,s2,color=self.col[i])
        plt.fill_between(time,s1,s2,facecolor=self.col[i],alpha=0.4,linewidth=0.1,interpolate=True)
        plt.plot(time,smin,color=self.col[i])
        plt.plot(time,smax,color=self.col[i])
        plt.fill_between(time,s2,smax,facecolor=self.col[i],alpha=0.1,linewidth=0.1,interpolate=True)
        plt.fill_between(time,smin,s1,facecolor=self.col[i],alpha=0.1,linewidth=0.1,interpolate=True)
        plt.plot(time,zero,'k',linestyle='--')
        if fx >= 1: plt.gca().xaxis.set_major_formatter(dates.DateFormatter('%d')) # %b'))
        if fx<1:plt.gca().xaxis.set_major_formatter(dates.DateFormatter('%H:%M'))
        if fx >= 1: plt.xlabel('Date '+xlabel, fontsize=20)
        if fx<1: plt.xlabel('Hour of day '+xlabel, fontsize=20)
        plt.ylabel(ylabel,fontsize=20)
        plt.ylim(-5.,5.)
        save_close_show(self.savename+'_MSS_'+self.model[i]+'_'+self.zoom)

        #plt.figure(figsize=(fx,5))
        #plt.title(self.label[i]+'      ['+self.zoom+']')
        #plt.plot(time,smean,color=self.col[i],linestyle='--',linewidth=2,label='MODEL-OBS')
        #s1=np.asarray(smean)-np.asarray(s95p)
        #s2=np.asarray(smean)+np.asarray(s95p)
        #plt.plot(time,s1,color=self.col[i])
        #plt.plot(time,s2,color=self.col[i])
        #plt.fill_between(time,s1,s2,facecolor=self.col[i],alpha=0.2,linewidth=0.1,interpolate=True)
        #plt.plot(time,zero,'gray',linestyle='--')
        #plt.xlabel('Date '+xlabel, fontsize=12)
        #plt.ylabel(ylabel,fontsize=16)
        #plt.ylim(-5.,5.)
        #save_close_show(self.savename+'_M9P_'+self.model[i]+'_'+self.zoom+'.png')

        allmean[i].append(smean)
        allstd[i].append(sstd)
        allrms[i].append(srms)

    ## Cumulative histogram
    plt.figure(figsize=(7,5))
    plt.title('Cumulative bias ['+self.zoom+']')
    for i,suite in enumerate(self.suite):
        flatlist = list(itertools.chain.from_iterable(all_data[i]))
#   b1=np.sort(np.abs((np.array(flatlist, dtype=float))))
        b1=np.sort(np.array(flatlist, dtype=float))
        nbins=np.arange(1,len(b1)+1)/np.float(len(b1))
        plt.step(nbins,b1, label=self.label[i], color=self.col[i], linewidth=3)
        plt.plot(nbins[-1],b1[-1],'o',color=self.col[i], markersize=15)
        plt.plot(nbins[0],b1[0],'o',color=self.col[i], markersize=15)
    plt.plot([0.,1.],[0.,0.],'gray',linestyle='--')
    plt.gca().xaxis.set_major_formatter(FuncFormatter(to_percent))
    plt.xlim(0.,1.)
    plt.yscale('symlog')
    plt.ylabel(ylabel,fontsize=20)
    plt.legend(fontsize=14,loc='best')
    plt.grid(True)
    save_close_show(self.savename+'_AbsBiasHist_'+self.zoom)
   
    ## Histogram
    plt.figure(figsize=(7,5))
    plt.title('Bias ('+xlabel+') ['+self.zoom+']')
    for i,suite in enumerate(self.suite):
        flatlist = list(itertools.chain.from_iterable(all_data[i]))
        b1=np.array(flatlist, dtype=float)
        b1 = b1[np.isfinite(b1)]
        nbin=250
        if aow=='ocn': nbin=1000
        bins = np.nanmin(b1) + (np.arange(nbin)/(1.0*nbin))*(np.nanmax(b1)-np.nanmin(b1))
        plt.hist(b1,bins=bins,label=self.label[i], color=self.col[i],histtype='step',linewidth=2)
    plt.xlabel(ylabel,fontsize=20)
    if not aow=='wav':
        plt.xlim(-10.,10.)
    else:
        plt.xlim(-4.,4.)
        plt.xlim(-8.,8.)
    plt.grid(True)
    plt.legend(fontsize=14,loc='best')
    save_close_show(self.savename+'_BiasHist_'+self.zoom)
   
    ## Time series of average means
    plt.figure(figsize=(fx,5))
    plt.title('Average bias ['+self.zoom+']')
    for i,suite in enumerate(self.suite):
       print((np.size(time),np.shape(time)))
       print((np.size(allmean[i]), np.shape(allmean[i]), np.shape(allmean[i][0])))
       plt.plot(time,allmean[i][0],color=self.col[i],linestyle='-',linewidth=3,label=self.label[i])
       #plt.plot(time,smin,color=self.col[i])
       #plt.plot(time,smax,color=self.col[i])
       #plt.fill_between(time,smax,smin,facecolor=self.col[i],alpha=0.2,linewidth=0.1,interpolate=True)
       plt.plot(time,zero,'k',linestyle='--',linewidth=1)
       plt.xlabel('Date '+xlabel, fontsize=20)
       plt.ylabel(ylabel,fontsize=20)
       if fx >= 1: plt.gca().xaxis.set_major_formatter(dates.DateFormatter('%d'))
       if fx < 1: plt.gca().xaxis.set_major_formatter(dates.DateFormatter('%H:%M')) 
       #plt.ylim(-2.0,2.0)
       plt.legend(fontsize=14,loc='lower left')
    save_close_show(self.savename+'_MMMall_'+self.zoom)


    plt.figure(figsize=(fx,5))
    plt.title('Relative |average bias| ['+self.zoom+']')
    for i,suite in enumerate(self.suite):
       print((np.size(allmean[i]), np.shape(allmean[i]), np.shape(allmean[i][0])))
       if i>0:
           plt.plot(time,np.abs(np.asarray(allmean[i][0]))-np.abs(np.asarray(allmean[0][0])),color=self.col[i],linestyle='-',linewidth=3,label=self.label[i]+'-'+self.model[0])
       plt.plot(time,zero,'k',linestyle='--',linewidth=1)
       plt.xlabel('Date '+xlabel, fontsize=20)
       plt.ylabel(ylabel,fontsize=20)
       if fx >= 1: plt.gca().xaxis.set_major_formatter(dates.DateFormatter('%d %b'))
       if fx < 1: plt.gca().xaxis.set_major_formatter(dates.DateFormatter('%H:%M'))
       #plt.ylim(-1.0,1.0)
       plt.legend(fontsize=14,loc='best')
    save_close_show(self.savename+'_MMMrel_'+self.zoom)

    plt.figure(figsize=(fx,5))
    plt.title('Standard deviation of bias ['+self.zoom+']')
    for i,suite in enumerate(self.suite):
       plt.plot(time,allstd[i][0],color=self.col[i],linestyle='-',linewidth=2,label=self.label[i])
       plt.plot(time,zero,'k',linestyle='--',linewidth=1)
    plt.xlabel('Date '+xlabel, fontsize=20)
    plt.ylabel(ylabel,fontsize=20)
       #plt.ylim(-2.0,2.0)
    plt.legend(fontsize=14,loc='best')
    save_close_show(self.savename+'_SSSall_'+self.zoom)

    plt.figure(figsize=(fx,5))
    plt.title('Root mean square error ['+self.zoom+']')
    for i,suite in enumerate(self.suite):
       plt.plot(time,allrms[i][0],color=self.col[i],linestyle='-',linewidth=2,label=self.label[i])
       plt.plot(time,zero,'k',linestyle='--',linewidth=1)
    plt.xlabel('Date '+xlabel, fontsize=20)
    plt.ylabel(ylabel,fontsize=20)
      #plt.ylim(-2.0,2.0)
    plt.legend(fontsize=14,loc='best')
    save_close_show(self.savename+'_RMSall_'+self.zoom)


    #fig = plt.figure(figsize=(fx,5))
    #ax = fig.add_subplot(111)
    #plt.title('Relative standard deviation of bias ['+self.zoom+']')
    #for i,suite in enumerate(self.suite):
    #   if i>0:
    #       ax.plot(time,np.asarray(allstd[i][0])-np.asarray(allstd[0][0]),color=self.col[i],linestyle='-',linewidth=2,label=self.model[i]+'-'+self.model[0])
    #   ax.plot(time,zero,'k',linestyle='--',linewidth=1)
    #   plt.xlabel('Date '+xlabel, fontsize=16)
    #   plt.ylabel(ylabel,fontsize=16)
    #   dfmt = dates.DateFormatter('%d %b') 
    #   plt.xaxis.set_major_formatter(dfmt)
       #plt.ylim(-1.0,1.0)
    #   plt.legend(fontsize=10,loc='best')
    #save_close_show(self.savename+'_SSSrel_'+self.zoom+'.png')


############################################################################## 
# Find minimum location for 'best stat'
def check_stats(map_stats, stat_name, indx, stat_value, min_value, max_test=False):
    
    pm=1.0
    if max_test: pm=-1.
    if np.isfinite(stat_value) & (pm*np.abs(stat_value) < pm*min_value[stat_name]):
        map_stats[stat_name][-1] = indx
        min_value[stat_name] = stat_value

    return map_stats, min_value

############################################################################## 
# Test significance and plot relative stats 
def compare_stats(self, map_lon, map_lat, map_num, map_stats, stat_name, mod_num, ref_num, title, savename):


    print(('Compare_stats: ', mod_num, ref_num, stat_name))
    vmin=0.1; vmax=2.0; cmap='PRGn_r'
    if stat_name=='rmse_abs':
        vmin=0.1; vmax=2.0; cmap='YlOrRd'
    if stat_name=='ns_abs':
        vmin=-1.; vmax=1.; cmap=plt.get_cmap('jet',25)
    if stat_name=='nbias_abs':
        vmin=-1.; vmax=1.; cmap=plt.get_cmap('RdYlBu_r',25)
    if np.size(np.shape(map_stats[stat_name])) > 1:
        
        # T-test for significance
        stat_label=stat_name.split('_abs')[0].upper()
        print(('** t-test REL '+stat_label, stats.ttest_rel(a=map_stats[stat_name][:,mod_num], b=map_stats[stat_name][:,ref_num])))

        # Plot absolute statistics values on map
        name=savename+"_map_"+stat_name
        full_title=title+' '+stat_label
        plot_scattermap(self,map_lon,map_lat,map_stats[stat_name][:,mod_num],map_num, savename=name,title=full_title,vmin=vmin,vmax=vmax,cmap=cmap)

        # Plot relative statistics to ref_num results on map
        name=name.replace("abs","rel")
        full_title=title+' Relative '+stat_label+' to '+self.label[ref_num]+' (%)'
        rel_stats =100.*(map_stats[stat_name][:,mod_num] - map_stats[stat_name][:,ref_num])/map_stats[stat_name][:,ref_num]
        plot_scattermap(self,map_lon,map_lat,rel_stats,map_num,savename=name,title=full_title,vmin=-25.,vmax=25.,cmap='PRGn_r')

############################################################################## 
# Matchup timeseries to common time stamps
def match_time(indata, intime, newtime, plot_freq=None, inst=True):

    # Interpolate indata(intime) to new time series as fn(newtime)
    intime2, inarg = np.unique(np.asarray(intime),return_index=True)
    out = pd.Series(index=np.asarray(newtime).astype(dt.datetime))
    in_data = pd.Series(indata[inarg], index=intime2.astype(dt.datetime))

    if plot_freq:   ## amend timing to compute daily means centred 0900-0900
        if plot_freq=='waterday': in_data = pd.Series(indata[inarg], index=intime2.astype(dt.datetime)).shift(-9)

    offset=newtime[0].minute
    x = in_data.combine_first(out) #.interpolate('time',limit=1)
    x = x.resample('60s').interpolate('time')
    output= x[(x.index >= newtime[0]) & (x.index <= newtime[-1])]

    if inst:      #treatment of instantaneous inputs - compute means
        ##output = output.resample('H', kind='timestamp',loffset=pd.Timedelta(str(offset)+'Min')).mean() #ffill(limit=1) #mean()    ## SURGE
        output = output.resample('H', kind='timestamp',loffset=pd.Timedelta(str(offset)+'Min')).ffill()  
    else:         #treatment of mean inputs - preserve means
#        output = output.resample('H', kind='timestamp',loffset=pd.Timedelta('30Min')).ffill()
    #    output = output.resample('H', kind='timestamp').ffill() #WAS:.ffill(limit=1)
        output = output.resample('H', kind='timestamp').ffill(limit=1)

    # Refine time series to required sampling frequency
    if plot_freq:
        if plot_freq=='daily': output=output.resample('D').mean()
        if plot_freq=='waterday': output=output.resample('D',loffset=pd.Timedelta('9h')).mean()
        if plot_freq=='hourly': output=output.resample('H').mean()
        if plot_freq=='diurnal': output=to_diurnal(output.index,output.values)
       
    return output

############################################################################## 
###############################################################################
## Match up model output with observations and call stats/plotting routines
def obs_model_match(self, aow, savename=None, output_tseries=False, 
                    plottype=None, scatter=False, mapstats=False, 
                    recip=False, dir=False, plot_freq='hourly', apply_qc=False, 
                    onlymod=False, ngrids=1):
    
    print(('RUNNING obs_model_match...', aow))
    print((self.title))
    print((self.savename))
    
    # length of obs data considered valid for plotting tseries
    if plot_freq=='hourly': val_len = 0.75*(self.nday*24)
    if plot_freq=='daily' or plot_freq=='waterday': val_len = 0.75*self.nday
    if plot_freq=='diurnal': val_len = 24

    # fix titles/filenames if only plotting 1 day
    if np.size(self.title)>1: self.title=self.title[0]
    if np.size(self.savename)>1: self.savename=self.savename[0]

    # set other plot labels
    if plottype=='doodson': self.title='Demerliac filtered \n'+str(self.title)
    if plottype=='residual': self.title='Residual \n'+str(self.title)
    tend_label=dt.datetime.strftime(dt.datetime.strptime(self.date,'%Y%m%d') + 
                                    dt.timedelta(days=self.nday),'%Y%m%d')
    tlabel = self.date+' - '+tend_label
    
    # produce matched outputs if mapstats, 'omb' plottype or scatter
    matchup = False
    if plottype=='omb' or mapstats or scatter:
        matchup = True
    print(('MATCHUP ', matchup))

###############################################################################
## initialise master OmB arrays
###############################################################################
    if matchup:
        all_time=[]
        all_data=[]
        all_obs=[]
        for i in self.suite:
            all_data.append([])
            
###############################################################################
## initialise 'best' stats lists
###############################################################################
    if matchup:
        map_lat=[]; map_lon=[]; map_num=[]
        map_stats={'rmse': [], 'bias': [], 'nbias': [], 'sind': [], 'ns': [], 
                   'rsq': [], 'rmse_abs': [], 'bias_abs': [], 'nbias_abs': [], 
                   'sind_abs': [], 'ns_abs': [], 'rsq_abs': []} 
    
################################################################################
## Loop over all available obs sites
################################################################################
    
#    for site in (["AberdeenTG","MillportTG","HeyshamTG","LerwickTG"]): #L4"]):   #,"HornseaWaver"]):
#    for site in (['23013','23091','23092','23093','23094','23095','23099','23451','23452','23453','23454','23456','23459','23460','23491','23492']):
    for site in self.all_sites[:]:    

        # Dictionaries to track min/max stat value across model runs
        best_stat={'rmse': 1.e15, 'bias': 1.e15, 'nbias': 1.e15, 'sind': -1.e15, 'ns': 1.e15, 'rsq':-1.e15} 
        stat_val={'rmse': [], 'bias': [], 'nbias': [], 'sind': [], 'ns': [], 'rsq': []} 
        for stat in stat_val:
            stat_val[stat]=np.zeros(np.size(self.suite))
        sig_ttest=np.zeros(np.size(self.suite))
        if plottype=="residual": mod_skew=[]
        
        # Set up observation data 
        if onlymod:   
            obs_data = np.zeros(5)
        else:
            obs_data = self.obs[self.obsite==site]
        print((site, '(Number of obs: ', np.size(obs_data), ')'))

        ## Find matching LAT,LON from suite      
        lllon, lllat, urlon, urlat = define_zoom(self.zoom)
        obx = self.oblon[self.obsite == site][0]
        oby = self.oblat[self.obsite == site][0]

        if aow=='wav' or aow=='ocn':
            mask=self.mapsta
        else:
            mask=None
        ix, iy, lonl, latl = find_xy_coord(self.lon, self.lat, obx, oby, self.data[-1][0], aow, mask=mask)

        # Check whether anything valid to plot
        plot = False
        print(('TESTING SITE : ', site, '[MAX/MIN obs ', np.nanmax(obs_data), np.nanmin(obs_data), np.size(obs_data), ']', obx, oby, ix, iy, np.max(self.data[-1][0].data[:,iy,ix]), np.min(self.data[-1][0].data[:,iy,ix])))
        if site != "BLACKLIST" and site != "None" and np.nanmax(obs_data) > -9000.:
            if np.size(obs_data) > 3:
                if ix > 0+ngrids and iy > 0+ngrids and obx > lllon and obx < urlon and oby > lllat and oby < urlat:
                    if np.isfinite(np.max(self.data[-1][0].data[:,iy,ix])): plot=True
                    #if np.nanmax(obs_data) > 0. and np.nanmin(obs_data) == np.max(obs_data): plot=False    # Simple obs QC
            if plot: 
                print(('PLOTTING SITE : ', site, '[MAX/MIN obs ', np.nanmax(obs_data), np.nanmin(obs_data), np.size(obs_data), ']'))
  
#######################################################################################################              
## create observation timeseries
        if plot:
            if plottype == 'anomaly': obs_data = obs_data - obs_data[0] #np.mean(obs_data)
            if not onlymod:
                                
                # Extract valid observations for selected location
                obs_data = obs_data[np.argsort(self.obtime[self.obsite==site])]
                obs_time = np.sort(self.obtime[self.obsite==site])
                valid_ob = np.where(np.isfinite(obs_data))
                obs_data = obs_data[valid_ob]
                obs_time = obs_time[valid_ob]
                
                # Set up time-matched observation timeseries
                print(('OBS_IN', obs_time[0], obs_time[-1]))
                obs_data2 = match_time(obs_data,obs_time,self.ptime,plot_freq)
                print(('OBS_DATA ', obs_data2.index[0], obs_data2.index[-1]))

                # If required, extract additional site-dependent metrics
                if plottype=='residual': 
                    obs_astro_time = self.astro_time[self.astro_site==site]
                    obs_astro_tide = self.astro_tide[self.astro_site==site]
                    if np.size(obs_astro_time) > 0:
                        astro_data = match_time(obs_astro_tide, obs_astro_time,self.ptime,plot_freq)

###############################################################################
## plot observation timeseries (if required)
            if output_tseries:
                if self.nday > 25:
                   plt.figure(figsize=(15,7))
                else:
                   plt.figure(figsize=(10,8))
                obllstr='Lat: {:.4}'.format(str(oby))+' Lon: {:.4}'.format(str(obx))
                if not onlymod: 
           
                   # Plot observations
                   if plottype=='omb':
                       plt.plot(obs_time.astype(dt.datetime), obs_data-obs_data[0], label=site+' OBS: '+obllstr+' (Mean: {:5.1f}'.format(np.nanmean(obs_data))+')', color='black',marker='+', lw=3)
                   elif plottype == 'residual':

                       obresid = self.obssr[(self.obsite==site) & np.isfinite(self.obssr)] 
                       #obresid = obresid - np.nanmean(obresid)
                       plt.plot(obs_time.astype(dt.datetime), obresid, label=site+' OBS: '+obllstr+' (Mean: {:5.1f}'.format(np.nanmean(obs_data))+')', color='black', lw=3,linestyle='-')

                   elif plottype == 'doodson':
                       pass
                   else:
                       oblabel='OBS: '+obllstr
                       if plot_freq=='diurnal': oblabel=oblabel+' (diurnal mean)'
                       if plot_freq=='waterday': oblabel=oblabel+' (waterday)'
                       print(('OBS_len', len(obs_time[obs_data > -900.])))
                       if plot_freq != 'waterday':
                          plt.plot(obs_time[obs_data > -900.].astype(dt.datetime), obs_data[obs_data > -900.], 'ko', label=oblabel, lw=4)
#                       if len(obs_time[obs_data > -900.]) > val_len: plt.plot(obs_time[obs_data > -900.].astype(dt.datetime), obs_data[obs_data > -900.], label=oblabel, color='black', lw=4)
#                       plt.plot(obs_time, obs_data2.values, 'ko', lw=4)
                       #if len(obs_time[obs_data > -900.]) > val_len: 
                       plt.plot(obs_data2.index, obs_data2.values, 'k-', lw=3)

###############################################################################
## create model timeseries - loop over all available model data        
            max_out = -500.; min_out = 500.
            for i,suite in enumerate(self.suite):
            
                if aow=='wav' or aow=='ocn':
                    if suite=='mi-aq602': 
                        mask=self.mapsta_am7
                        ix, iy, lonl, latl = find_xy_coord(self.lon, self.lat, obx, oby, self.data[i][0], aow, mask=mask)           
                    else:
                        mask=self.mapsta
                else:
                    mask=None
                
                if aow=='atm' and i==0 and np.size(self.data[i][0][0,0,:].data) > 1000: 
                    ix, iy, lonl, latl = find_xy_coord(self.lonO, self.latO, obx, oby, self.data[i][0], 'ocn', mask=mask)    # GLOBAL FLX FIX
                else:
                    ix, iy, lonl, latl = find_xy_coord(self.lon, self.lat, obx, oby, self.data[i][0], aow, mask=mask)   
                print(('IX/IY ', ix, iy, latl, lonl))

                # Extract model data
                model_cube=self.data[i]
                model_data = []
                if plottype == 'anomaly': dat0 = model_cube[0][0,iy,ix]
                print(('SIZE_CUBE ', np.size(model_cube)))

                # Loop over input days, consider nearby grids
                for ic,cube in enumerate(model_cube):            
            
                    if dir:
                        outx = cube[:,iy,ix].data
                        outy = self.datay[i][ic][:,iy,ix].data
                        wdir = 270.-(np.arctan2(outy,outx)*180./np.pi)
                        wdir[wdir > 360.] = wdir[wdir > 360] - 360.
                        outdir = cube[:,iy,ix].copy()
                        outdir.data = wdir
                        model_data.append(outdir)
                    else:

                        pr = np.int((ngrids-1)/2)
                        if np.size(np.shape(cube.data)) == 3:
                            local_data=cube[:,iy-pr:iy+pr+1,ix-pr:ix+pr+1]
                        else:
                            local_data=cube[iy-pr:iy+pr+1,ix-pr:ix+pr+1]
              
                        try:
                            col1 = 'grid_longitude'; col2 = 'grid_latitude'
                            local_mean=local_data.collapsed([col1,col2],iris.analysis.MEAN)
                        except:
                            col1 = 'longitude'; col2 = 'latitude'
                            local_mean=local_data.collapsed([col1,col2],iris.analysis.MEAN)
                        local_mean=local_data.collapsed([col1,col2],iris.analysis.MEAN)       
                        if plot_freq=='waterday': local_mean=local_data.collapsed([col1,col2],iris.analysis.MAX) 
                        if (aow=='ocn' or aow=='wav') and np.min(local_mean) == 0.0:
                            local_mean.data=np.nanmean(local_data.data, axis=(1,2))   
              
                        model_data.append(local_mean)

###############################################################################
## loop over daily input data                        

                mod_out = []
                model_time = np.asarray(self.ptime)
                for iday,cube in enumerate(model_data):
                    mod_out.append(cube.data)
                mod_out = np.ravel(mod_out)

                # Extract reference data for residual analysis
                if plottype=='residual':
                    #tideonly_all = self.tideonly['tideonly'][site]

                    # TEST - read in tideonly run as model0, + add inst=False in match_time
                    if i==0:
                       tideonly_all = mod_out
                       self.tideonly_time=model_time
                       tideonly = pd.Series(mod_out, index=np.asarray(model_time).astype(dt.datetime))

                    if not onlymod:
                       surge_model = self.tideonly['surgerun'][site]
                        #tideonly = match_time(tideonly_all,self.tideonly_time,self.ptime,plot_freq,inst=False)
                       surge_model = match_time(surge_model,self.tideonly_time,self.ptime,plot_freq)

                       if i==0:
                            plt.plot(self.tideonly_time, surge_model, label='NEMO4_surge', color='k',linestyle='-') 
   
                # Apply tseries-relevant transformations                 
                if recip: mod_out = 1.0/mod_out    # (e.g. output wave period)
                if plottype == 'anomaly': mod_out = mod_out - mod_out[0] #np.mean(mod_out)
                if apply_qc: qcthresh = np.std(mod_out) * 5.0
                
###############################################################################
## extract matching timeseries and compute matched statistics
                
                # Define model data at required sampling frequency#
                label = self.label[i]#+': Lat: {:.3f}'.format(latl)+' Lon: {:.3f}'.format(lonl)
                
                if plot_freq != 'hourly':
                   model_data2 = match_time(mod_out,model_time,self.ptime,plot_freq,inst=False)
                else:
                   model_data2 = pd.Series(mod_out, index=np.asarray(model_time).astype(dt.datetime))
                print(('MODEL: ', model_data2.index[0], model_data2.index[-1]))

                # Compute residual relative to reference
                if not onlymod and plottype == 'residual':
                    offset=np.nanmean(obresid)-np.nanmean(model_data2.values)
                    print((site, 'OFFSET: ', offset))


                    # Compute skew surge
                    if i==0 and np.size(obs_astro_tide)>1:
                        Amx_time, Amx_tide = max_tide(astro_data.index,astro_data.values-np.nanmean(astro_data.values),tide_max=True)
                        Omx_time, Omx_tide = max_tide(obs_data2.index,obs_data2.values-np.nanmean(obs_data2.values),tide_max=True)

                        ob_skew=np.zeros(np.size(obs_data2.values))
                        if np.size(Omx_tide) == np.size(Amx_tide):
                            ob_skew = Omx_tide-Amx_tide
                            plt.plot(Amx_time, ob_skew,'ko')
                                                        
                    if np.size(Amx_time)>1:
                        Rmx_time, Rmx_tide = max_tide(model_data2.index,tideonly-np.nanmean(tideonly),tide_max=True)
                        Mmx_time, Mmx_tide = max_tide(model_data2.index,model_data2.values-np.nanmean(model_data2.values),tide_max=True)
                        mod_skew.append(np.zeros(np.size(model_data2.values)))
                        if np.size(Mmx_tide) == np.size(Rmx_tide):
                            mod_skew[-1] = Mmx_tide-Rmx_tide
                            plt.plot(Rmx_time, mod_skew[-1],'o',color=self.col[i])

                # Compute obs and model differences
                model_time_data = model_data2.index
                model_time = model_time.astype(dt.datetime)
                mod_m_data = np.ravel(model_data2.values)
    
                if onlymod and plottype=='residual':
                   mod_m_data = mod_m_data - tideonly.values

                if not onlymod and matchup:

                    ## CALCULATE MATCHUP
                    obs_m_data = np.ravel(obs_data2.values)
                    print(('N matchup: ',np.size(obs_m_data),np.size(mod_m_data)))
                    print(('N matchup: ',np.shape(obs_m_data),np.shape(mod_m_data)))

                    # Apply low-pass filter, if required
                    # Compute DoodsonX0 filter for SSH hourly model/obs data
                    if plottype == 'doodson':
                        obs_m_data = DoodsonX0(obs_m_data)
                        mod_m_data = DoodsonX0(mod_m_data)

                    # Compute residual, if required
                    if plottype == 'residual':
                        obs_m_data = obs_data2.values - astro_data.values
                        mod_m_data = model_data2.values - tideonly #+ offset

                    # Calculate matched (model - obs) statistics
                    dmo = mod_m_data - obs_m_data
                    if dir:
                        dmo[dmo > 180.] = dmo[dmo > 180.] - 360.
                        dmo[dmo < -180.] = dmo[dmo < -180.] + 360.

                    # Apply QC checks
                    if apply_qc: dmo[np.abs(dmo) > qcthresh] = np.nan   
                    #if apply_qc: dmo[np.abs(dmo) > 6.] = np.nan
                    
                    # Append to master arrays 
                    # Only consider sites with >VAL_LEN% data coverage in period
                    if np.size(dmo[dmo > -900.]) > val_len:
                        if i==0:
                           all_time.append(np.array(model_data2.index))
                           all_obs.append(np.array(obs_m_data))
                        all_data[i].append(dmo)

###############################################################################
#  Compute model - obs statistics
                        slope, inter, r_val, p_val, stderr = stats.linregress(obs_m_data, mod_m_data)
                        if np.isnan(r_val): r_val = -999.
                        stat_val['rsq'][i] = r_val**2
                        stat_val['bias'][i] = calc_bias(dmo, obs_m_data, mod_m_data)
                        label=label+'; MD: {:.2f}'.format(stat_val['bias'][i])
                        stat_val['rmse'][i] = calc_rmse(dmo, obs_m_data, mod_m_data)
                        label=label+'; RMSD: {:.2f}'.format(stat_val['rmse'][i])
                        stat_val['nbias'][i] = calc_nbias(dmo, obs_m_data, mod_m_data)
                        #label=label+'; NBIAS: {:.2f}'.format(stat_val['nbias'][i])
                        stat_val['sind'][i] = calc_sind(dmo, obs_m_data, mod_m_data)
                        #label=label+'; SI: {:.2f}'.format(stat_val['sind'][i])
                        stat_val['ns'][i] = calc_ns(dmo, obs_m_data, mod_m_data)
                        #label=label+'; NS: {:.2f}'.format(stat_val['ns'][i])
                        print((suite, ': STATS ', label))
                        print('ss_lin'+site,latl, lonl, np.nanmax(obs_data), np.nanmin(obs_data), stat_val['bias'][i], stat_val['rmse'][i], stat_val['nbias'][i], stat_val['ns'][i])

                        # Test statistical significance of stats difference
                        if i==0:
                            b_arr = dmo
                        if i>0:
                            ttest=stats.ttest_rel(a=dmo, b=b_arr)
                            print(('** t-test REL bias', ttest))
                            if ttest[1] < 0.05: sig_ttest[i] = 1   # STAT. SIG.
                        
################################################################################
## select 'best' statistics
                        # extend collated stats arrays
                        if i==0:
                            map_lat.append(oby)
                            map_lon.append(obx)
                            map_num.append(np.size(dmo[dmo > -900.]))
                            for stat in map_stats:
                                if not 'abs' in stat:
                                    map_stats[stat].append(-1)
                        
                        # test stats relative to other model run, finding best
                        max_test=False
                        for stat in map_stats:
                            if not 'abs' in stat:
                                if stat=='sind' or stat=='ns' or stat=='rsq': max_test=True
                                check_stats(map_stats, stat, i, stat_val[stat][i], best_stat,max_test=max_test)

################################################################################
## plot model data timeseries

                if output_tseries:
                    
                    # plot time-matched filtered obs if low-pass filtered
                    if not onlymod and plottype == 'doodson' and i==0:
                       plt.plot(np.array(model_time_data), obs_m_data, label=site+' OBS: '+obllstr, color='black',marker='+', lw=1)
                    
                    # plot absolute/relative model time series
                    if plottype == 'omb': 
                        plt.plot(model_data2.index, dmo, label=label, lw=2, color=self.col[i], linestyle=self.lst[i])
#           elif plottype == 'diurnal':
#                        plt.plot(dcycle.index, dcycle.values, label=label, lw=2, color=self.col[i], linestyle=self.lst[i])
                    else:
                        #plt.plot(model_time,mod_out,color=self.col[i])
                        plt.plot(model_data2.index, mod_m_data, label=label, lw=3, color=self.col[i], linestyle=self.lst[i])

                # Set plot scales
                if plottype == 'omb' or plottype == 'doodson':  
                  if np.size(dmo) > 1 and np.max(dmo) > -900.:
                    max_out = np.max([np.max(dmo[dmo > -900.]), max_out])
                    min_out = np.min([np.min(dmo[dmo > -900.]), min_out])
                else:
                  if np.size(mod_m_data) > 1 and np.max(mod_m_data) > -900.:
                    max_out = np.max([np.max(mod_m_data[mod_m_data > -900.]), max_out])
                    min_out = np.min([np.min(mod_m_data[mod_m_data > -900.]), min_out])

                if not onlymod and matchup:
                    if i == 0: 
                        out = np.zeros((np.size(mod_m_data),np.size(self.suite)))
                    out[:,i] = mod_m_data 
  

## END OF LOOPING OVER MODEL RUNS
###############################################################################

## timeseries plot attributes
            plotstr=''
            if plottype == 'anomaly': 
                plotstr="ANOMALY"
            if plottype == 'omb':
                plotstr = 'MODEL - OBS'
            if not onlymod:
                if plottype=='omb':
                  ymin = np.nanmin([np.nanmin(obs_data-obs_data[0]),min_out])
                  ymax = np.nanmax([np.nanmax(obs_data-obs_data[0]),max_out])
                  #ymin = -3. #np.min(obs_data-obs_data[0])
                  #ymax = 3. #np.max(obs_data-obs_data[0])
                elif plottype=='doodson' or plottype=='residual':
                  ymin = np.nanmin([np.nanmin(obresid),np.nanmin(mod_m_data),-0.6])
                  ymax = np.nanmax([np.nanmax(obresid),np.nanmax(mod_m_data),0.6])
                  #ymin=-5
                  #ymax=5
                else:
                  ymin = np.nanmin([np.nanmin(obs_data),min_out])-0.2
                  ymax = np.nanmax([np.nanmax(obs_data),max_out])+0.2
            else:
                ymin=min_out; ymax=max_out
                id_obs=[]
            
            if output_tseries:
                if plottype != 'diurnal':
                    m1 = np.ravel(model_time)[0]
                    m2 = np.ravel(model_time)[-1]
                    plt.xlim([m1,m2])
#                if plottype=='doodson': plt.yscale('symlog') 
                if not onlymod: plt.ylim([ymin,ymax])
                if not onlymod and ymax > 10000.: 
                      plt.yscale('symlog')   #vis
                      plt.ylim(ymin=10.0)       #vis
                if np.size(self.model) <= 10: plt.legend(fontsize=14,loc='best')
                self.title = ''.join(self.title)
                ylabel=self.title+' '+plotstr
                plt.ylabel(ylabel,fontsize=20)
                xlabel=''
                if plottype == 'omb' or plottype == 'doodson': 
                    xlabel ='    ['+str(np.size(dmo[dmo > -900.]))+' match]'
                plt.xlabel('Date '+tlabel+xlabel,fontsize=18)
                plt.title(self.title+' at '+site,fontsize=20)
                plt.grid(True)
                if self.nday <= 5: plt.gca().xaxis.set_major_formatter(dates.DateFormatter('%H:%M'))
                if self.nday > 5: plt.gca().xaxis.set_major_formatter(dates.DateFormatter('%d')) # %b'))
                if plottype=='omb':
                   plt.plot([m1,m2],[0.,0.],'gray')

## save/show output plot
                self.savename = ''.join(self.savename)
                print(("savename:", self.savename))
                print(("plot:", plottype))
                print(("site:", site))
                outname=self.savename+'_'+plottype+site
                if plot_freq:
                    if plot_freq=='diurnal': outname=outname+plot_freq
                print(outname)
                save_close_show(outname)

####################################################################################################### 
## combine all outputs and results
            if not onlymod and matchup and np.size(dmo[dmo > -900.]) > val_len:

## write all data to file
                write_data_to_file(outname, model_data2.index, obs_m_data, out, val_len, self.model)
                
####################################################################################################### 
                # stat sig
                print(('SIG TEST', sig_ttest, map_stats['bias'][-1], map_stats['rmse'][-1]))
                if np.max(sig_ttest) > 0 and map_stats['bias'][-1]>-1 and map_stats['rmse'][-1]>-1: 
                    for stat in map_stats:
                        if not 'abs' in stat:
                            map_stats[stat][-1] = map_stats[stat][-1]+100

                for stat in map_stats:
                    if 'abs' in stat:
                        map_stats[stat].append(np.asarray(stat_val[stat.split('_abs')[0]][:]).copy())

## plot scatter plot and compute stats - show all model runs on same plot
                if scatter: 
                    plt.figure(figsize=(6*np.size(self.suite),7))
                    for i,suite in enumerate(self.suite):
                        label=self.label[i]+'; r2={:.2f}'.format(stat_val['rsq'][i])+' bias={:.3f}'.format(stat_val['bias'][i])
                    ## Point scatterplot
#                    plt.plot(obs_m_data, out[:,i],'*', color=self.col[i], label=label)
                    ## Hexbin scatterplot
                        plt.subplot(1,np.size(self.suite),i+1)
                        plt.hexbin( obs_m_data, out[:,i], gridsize=[100,100], mincnt=1,  label=label)
                    
## scatter plot attributes
                        plt.grid()
                        plt.colorbar()
                        plt.xlim([ymin-0.05,ymax+0.05])
                        plt.ylim([ymin-0.05,ymax+0.05])
                        plt.plot([ymin,ymax],[ymin,ymax],'k--')
                        if i==0: plt.ylabel('Model output '+self.title)
                        plt.xlabel('Observations '+self.title+' '+site+' ['+tlabel+']')
                        plt.legend(fontsize=14,loc='best')
                        plt.title(label, fontsize=20)
                #plt.legend(fontsize=10,loc='best')
                                    
                    outname=self.savename+plottype+'scatter_'+site
                    save_close_show(outname)
        
                    # Scatter plot skew surge
                    if plottype=='residual':
                        if np.size(ob_skew) == np.size(mod_skew[0]):
                            fig, ax = plt.subplots(figsize=(6,7))
                            for i,suite in enumerate(self.suite):
                                slope, inter, r_val, p_val, stderr = stats.linregress(ob_skew, mod_skew[i])
                                label=self.label[i]+'; r2={:1.2f}'.format(r_val**2)
                                plt.plot(ob_skew, mod_skew[i],'o', color=self.col[i],label=label)                        
                            plt.xlabel('Observed skew surge \n'+' '+site+' ['+tlabel+']')                     
                            plt.ylabel('Model skew surge '+self.model[i])
                            plt.grid()
                            smin = np.min([np.min(ob_skew), np.min(mod_skew)])
                            smax = np.max([np.max(ob_skew), np.max(mod_skew)])
                            smin = smin-0.05; smax=smax+0.05
                            plt.xlim([smin,smax])
                            plt.ylim([smin,smax])
                            plt.plot([smin,smax],[smin,smax],'k--')
                            plt.legend(fontsize=14, loc='best')
                            outname=self.savename+plottype+'skew_'+site
                            save_close_show(outname)
                            

##############################################################################
#### END OF LOOPING OVER ALL SITES

## after final site calculations, map 'best' suite
    if matchup and not onlymod: # and len(map_stats['rmse_abs'][:,0])>15:   # and self.plotobs:

        for stat in map_stats:
            map_stats[stat]=np.asarray(map_stats[stat])

        # This number indicates the model relative to which differences are calculated
        rs=0

        write_out=True
        for i,suite in enumerate(self.model):

            for stat in ['rmse_abs', 'bias_abs', 'nbias_abs', 'sind_abs', 'ns_abs', 'rsq_abs']:
                title=self.title+'\n '+tlabel+' '+self.label[i]
                savename=self.savename+plottype+'_'+self.label[i]
                print(title)
                compare_stats(self,map_lon,map_lat,map_num,map_stats,stat,i,rs,title,savename)
                print(("WRITE_OUT", write_out))
                print(np.size(map_stats[stat][:] <= 0.))
                print(("MAT?CH?UP", matchup))
#                if np.size(map_stats[stat][:] <= 0.): write_out=False

        ## OUTPUT COMBINED TIMESERIES MAP
        if matchup:

            if write_out:

                ## categorical stats
                if aow=='atm' and self.fog:
                    categorical_stats(self,all_obs,all_data)
                
                print('** plotting combined timeseries **')
                plot_combined(self,aow,all_time,all_obs,all_data,tlabel,ylabel)

        # Plot 'best suite' statistics
        for stat in ['rmse', 'bias', 'nbias', 'sind', 'ns', 'rsq']:
            name=self.savename+plottype+"_map_"+stat
            title=self.title+'\n '+tlabel+' "best suite", '+stat.upper()
            plot_scattermap(self,map_lon,map_lat,map_stats[stat],map_num,savename=name,title=title,best=True)

###############################################################################
