# -*- coding: utf-8 -*-
""" Estuarine Box Model Input maker

Created on Thu Jul  6 12:20:57 2023
@author: aafur
"""
#%%Dependecies
import os
import shutil
import pandas as pd
from o_functions.start import opsys2; start_path = opsys2()

#%%constants
path = start_path + 'modelling_DATA/EBM_PRIMEA/primea_models'
path_to_fortran_model = start_path + 'GitHub/python-oceanography/Delft 3D FM Suite 2019/ocean_functions/o_functions/cmcc_ebm_daily.F90'

estuary = '/estuary_in_'
ocean = '/ocean_in_'
Qr_perc = '/Qr_discharge_'
uv_tide = '/uvtide_'

#%%Functions

def make_files(path,name,river_vals,Qr_path):
    new_path = path + '/' + name
    if os.path.isdir(new_path) == False:
        os.mkdir(new_path)
        os.mkdir(new_path + '/output') #generate output directory
        print('Creating dir path, creating files...')
    else:
        print('Path already exists, overwriting files...')
    

    if os.path.isfile(new_path + ('/' + path_to_fortran_model.split('/')[-1])) == False:
        shutil.copy(path_to_fortran_model, new_path + ('/' + path_to_fortran_model.split('/')[-1]))

    def perc(Qr_path, new_path):
        ''' Function to make riverine input for CMCC model
        '''
        open(new_path + Qr_perc + name,'w').close()
        names = ['Time','Qr']
        d = pd.read_csv(Qr_path, names = names)
        river_path = new_path + ('/riverQr_' + Qr_path.split('/')[-1])
        if os.path.isfile(river_path) == False:
            shutil.copy(Qr_path, river_path)
        with open(new_path + Qr_perc + name, 'a') as f:
            f.writelines(str(i) + '\n' for i in d.Qr)
    
    def estuary_in(river_vals):
        '''
        #A_surf: cross section area at the river mouth in units of m^2
        #L_e: fixed lenght for the estuary box in units of m (deprecated)
        #W_m: estuary width at the river mouth in units of m
        #h: estuary depth at the river mouth in units of m 
        #h_l: depth of estuary upper or lower layer at the river mouth in units of m
        #Q_m: River multiannual average discharge m3/s
        '''
        open(new_path + estuary + name,'w').close() #make estuary file
        with open(new_path + estuary + name,'a') as f:
            f.writelines(str(i) + '\n' for i in river_vals) 
            f.write('\n')
            f.write(f"#A_surf: cross-section area at the river mouth in units of m^2\n#{river_vals[0]}\n")
            f.write(f"#L_e: fixed length for the estuary box in units of m (deprecated)\n#{river_vals[1]}\n")
            f.write(f"#W_m: estuary width at the river mouth in units of m\n#{river_vals[2]}\n")
            f.write(f"#h: estuary depth at the river mouth in units of m\n#{river_vals[3]}\n")
            f.write(f"#h_l: depth of estuary upper or lower layer at the river mouth in units of m\n#{river_vals[4]}\n")
            f.write(f"#Q_m: River multiannual average discharge m3/s\n#{river_vals[5]}\n")


    def ocean_in():
        open(new_path + ocean + name,'w').close()
    

    def uv_tide_maker():
        ''' This file should have columns from tidal prediction, they should all be 
        from one tidally forced location, as in the test case 
        column 1)Latitude  
        column 2)Longitude
        column 3)mm.dd.yyyy
        column 4)hh:mm:ss
        column 5)zonal transport
        column 6)meridional transport
        column 7)zonal velocity
        column 8)meridional velocity
        column 9)depth (m)
        '''
        open(new_path + uv_tide + name,'w').close()

    estuary_in(river_vals)
    uv_tide_maker()
    ocean_in()
    perc(Qr_path, new_path)

    return new_path

def netcdf_maker(new_path,year):
    ''' Netcdf files should exists for every timestep. Could produce hourly 
    results. 
    

    Returns
    -------
    None.

    '''
    nc_path = new_path + '/' + name + '_' +year
    if os.path.isdir(nc_path) == False:
        os.mkdir(nc_path)
    
def run_file_make(new_path,data, filenames): #nested function which calls the runfile. 
    filename = new_path + '/run_' + new_path.split('/')[-1] + '_ebm'
    path_to_blank = start_path + r'GitHub/python-oceanography/Delft 3D FM Suite 2019/ocean_functions/o_functions/run_cmcc_ebm.sh'
    shutil.copy(path_to_blank,filename)

    with open(filename, 'r') as file:
        lines = file.readlines()
        
    lines[20] = lines[20].replace('side=', f'side={data[0]}') #sides
    lines[23] = lines[23].replace('year=', f'year={data[1]}') #years
    lines[24] = lines[24].replace('days=', f'days={data[2]}') #days
    lines[25] = lines[25].replace('hours=', f'hours={data[3]}') #hours
    lines[26] = lines[26].replace('jday=', f'jday={data[4]}') #jday
    lines[29] = lines[29].replace('InFile_param=',f'InFile_param={filenames[0]}') #InfileParam
    lines[30] = lines[30].replace('InFile_ocean=',f'InFile_ocean={filenames[1]}') #InFile_ocean
    lines[31] = lines[31].replace('InFile_Qr=',f'InFile_Qr={filenames[2]}') #InFile_Qr
    lines[32] = lines[32].replace('InFile_vel_tide=',f'InFile_vel_tide={filenames[3]}') #InFile_vel_tide
    lines[33] = lines[33].replace('msk=',f'msk={data[5]}') #msk
    lines[34] = lines[34].replace('miss=',f'miss={data[6]}') #miss
    lines[41] = lines[41].replace('outFile=', f'outFile={filenames[4]}')
    
    

    #write the modified lines back to file
    with open(filename, 'w') as file:
        file.writelines(lines)
#%% Test Data Set to make everything work and run a model
if __name__ == '__main__':
    name = 'PRIMEA_TEST'
    
    side="west" #logical param on river mouth location
    year="2013"
    days="365"  #days of current year 
    hours="8760" #hours of current year
    jday="2458119.5" #jday of the day before the simulation start
    msk="1000" #upper threshold value of nemo LSM
    miss="-999" #runoff missing values
    
    data = [side,year,days,hours,jday,msk,miss]
    filenames = [estuary.replace('/','') + name,
                 ocean.replace('/','') + name,
                 Qr_perc.replace('/','') + name,
                 uv_tide.replace('/','') + name,
                 'cmcc_ebm_output.txt']
    river_vals = [24000,5000,800,30,15,40]
    
    Qr_path = start_path + r'modelling_DATA/kent_estuary_project/river_boundary_conditions/discharges/Dee.csv'
    
    model_path = make_files(path,name,river_vals,Qr_path)
    run_file_make(model_path,data, filenames)
    netcdf_maker(model_path,year)
    
    print('\nData for this run is located @:')
    print(model_path)
