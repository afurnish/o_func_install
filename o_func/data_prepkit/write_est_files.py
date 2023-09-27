#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" write_est_files
Created on Mon Sep 25 09:29:19 2023

@author: af
"""
import os

class est_file_maker_CMCC:
    def __init__(self):
        self.nemo_lines = [
            "9)nx !this is the 1st NEMO sea zonal point off the outlet point Po di Goro",
            "10)ny !this is the 1st NEMO sea meridional point off the outlet point Po di Goro",
            "11)up !this is the NEMO level corresponding to the top of EBM lowerlayer",
            "12)down !this is the NEMO level corresponding to the bottom of EBM lowerlayer",
            "13)top !this is the NEMO top level (deprecated)",
            "14)nl !num NEMO levels within EBM lowerlayer",
            "15)ne !num NEMO levels within EBM depth",
            "16)dx !buffer area of NEMO ROFI: zonal points around nx",
            "17)dy !buffer area of NEMO ROFI: meridional points around ny",
            "18)ngx !nemo mesh points in zonal  dir x",
            "19)ngy !nemo mesh points in meridional dir y",
            "20)ngz !nemo mesh  points in vertical dir z",
            "21)ngt !nemo output freq (daily files with 1 timestep)",
            "22)msk !land point value"
        ]
        
        self.est_lines = [
            "2)A_surf: cross section area at the river mouth in units of m^2",
            "3)L_e: fixed length for the estuary box in units of m (deprecated)",
            "4)W_m: estuary width at the river mouth in units of m",
            "5)h: estuary depth at the river mouth in units of m",
            "6)h_l: depth of estuary upper or lower layer at the river mouth in units of m",
            "7)Q_m: River multiannual average discharge m3/s",
        ]

    
    
    def write_layer(self):
        self.filename = self.est_name + '.est'
        with open (os.path.join(self.dir_path, self.filename), 'w') as f:
            f.write('#' * 35) ; f.write(self.filename); f.write('#' * 35)
            f.write ('\n')
            for i, val in enumerate(self.variables):
                f.write(val); f.write ('\n')
                if i == 6: # end of est data 
                    f.write('\n')
            f.write('\n')
            f.write('\n'.join(self.est_lines))
            f.write('\n'.join(self.nemo_lines))


    def write_est_file(self,directory, est_name, A_surf, L_e, W_m, h, h_l, Q_m, nx, 
                       ny, up, down, top, nl, ne, dx, dy, ngx, ngy, ngz, ngt, msk):
        self.dir_path = directory
        self.est_name = est_name
        self.A_surf = A_surf
        self.L_e = L_e
        self.W_m = W_m
        self.h = h
        self.h_l = h_l
        self.Q_m = Q_m
        self.nx = nx
        self.ny = ny
        self.up = up
        self.down = down
        self.top = top
        self.nl = nl
        self.ne = ne
        self.dx = dx
        self.dy = dy
        self.ngx = ngx
        self.ngy = ngy
        self.ngz = ngz
        self.ngt = ngt
        self.msk = msk
        
        self.variables=[self.A_surf,
                        self.L_e, 
                        self.W_m,
                        self.h, 
                        self.h_l,
                        self.Q_m,
                        self.nx,
                        self.ny,
                        self.up,
                        self.down,
                        self.top,
                        self.nl,
                        self.ne,
                        self.dx,
                        self.dy,
                        self.ngx,
                        self.ngy,
                        self.ngz,
                        self.ngt,
                        self.msk
                        ]
    
        self.write_layer()
        
    def read_change_est_file(self):
        file_to_read = os.path.join(self.dir_path, self.filename)
        
class est_file_maker_UKC4:
    def __init__(self):
        self.nemo_lines = [
            "9)nx !this is the 1st NEMO sea zonal point off the outlet point Po di Goro",
            "10)ny !this is the 1st NEMO sea meridional point off the outlet point Po di Goro",
            "11)up !this is the NEMO level corresponding to the top of EBM lowerlayer",
            "12)down !this is the NEMO level corresponding to the bottom of EBM lowerlayer",
            "13)top !this is the NEMO top level (deprecated)",
            "14)nl !num NEMO levels within EBM lowerlayer",
            "15)ne !num NEMO levels within EBM depth",
            "16)dx !buffer area of NEMO ROFI: zonal points around nx",
            "17)dy !buffer area of NEMO ROFI: meridional points around ny",
            "18)ngx !nemo mesh points in zonal  dir x",
            "19)ngy !nemo mesh points in meridional dir y",
            "20)ngz !nemo mesh  points in vertical dir z",
            "21)ngt !nemo output freq (daily files with 1 timestep)",
            "22)msk !land point value"
        ]
        
        self.est_lines = [
            "2)A_surf: cross section area at the river mouth in units of m^2",
            "3)L_e: fixed length for the estuary box in units of m (deprecated)",
            "4)W_m: estuary width at the river mouth in units of m",
            "5)h: estuary depth at the river mouth in units of m",
            "6)h_l: depth of estuary upper or lower layer at the river mouth in units of m",
            "7)Q_m: River multiannual average discharge m3/s",
        ]

    
    
    def write_layer(self):
        self.filename = self.est_name + '.est'
        with open (os.path.join(self.dir_path, self.filename), 'w') as f:
            f.write('#' * 35) ; f.write(self.filename); f.write('#' * 35)
            f.write ('\n')
            for i, val in enumerate(self.variables):
                f.write(val); f.write ('\n')
                if i == 6: # end of est data 
                    f.write('\n')
            f.write('\n')
            f.write('\n'.join(self.est_lines))
            f.write('\n'.join(self.nemo_lines))


    def write_est_file(self,directory, est_name, A_surf, L_e, W_m, h, h_l, Q_m, nx, 
                       ny, up, down, top, nl, ne, dx, dy, ngx, ngy, ngz, ngt, msk):
        self.dir_path = directory
        self.est_name = est_name
        self.A_surf = A_surf
        self.L_e = L_e
        self.W_m = W_m
        self.h = h
        self.h_l = h_l
        self.Q_m = Q_m
        self.nx = nx
        self.ny = ny
        self.up = up
        self.down = down
        self.top = top
        self.nl = nl
        self.ne = ne
        self.dx = dx
        self.dy = dy
        self.ngx = ngx
        self.ngy = ngy
        self.ngz = ngz
        self.ngt = ngt
        self.msk = msk
        
        self.variables=[self.A_surf,
                        self.L_e, 
                        self.W_m,
                        self.h, 
                        self.h_l,
                        self.Q_m,
                        self.nx,
                        self.ny,
                        self.up,
                        self.down,
                        self.top,
                        self.nl,
                        self.ne,
                        self.dx,
                        self.dy,
                        self.ngx,
                        self.ngy,
                        self.ngz,
                        self.ngt,
                        self.msk
                        ]
    
        self.write_layer()
        
    def read_change_est_file(self):
        file_to_read = os.path.join(self.dir_path, self.filename)
        
        
        
if __name__ == '__main__':
        
    #%%inputs
    #Name of estuary. 
    directory = 'ests'
    est_name  = 'pogoro'
    A_surf = '1000'
    L_e = '12000'
    W_m = '200'
    h = '5.0'
    h_l = '2.5'
    Q_m = '218.00'
    
    #maybe I can extrapolate this stuff. 
    nx = '11'
    ny = '19'
    up = '2'
    down = '2'
    top = '1'
    nl = '1'
    ne = '2'
    dx = '8'
    dy = '8'
    ngx = '49'
    ngy = '48'
    ngz = '141'
    ngt = '1'
    msk = '3000'
    
    est = est_file_maker_CMCC()
    
    est.write_est_file(directory,est_name, A_surf, L_e, W_m, h, h_l, Q_m, nx, 
                       ny, up, down, top, nl, ne, dx, dy, ngx, ngy, ngz, ngt, msk)
    
    
    
    
    
