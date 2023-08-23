#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import platform
import os
import datetime as dt  


def winc(i):
        if platform == "win32" or "Windows":
            b = i.replace('\\','/')
        else:
            b = i
        return b
    
def md(i):
    '''
    Function to generate a new directory path. Will check if directory exists and 
    then will create it if it doesnt exist or do nothing if it does. 

    Parameters
    ----------
    i : This should be the directory to be made. It should be a list of the components that go 
    into making the file as they are joined together with os.path.join

    Returns
    -------
    None.

    '''
    new_path = os.path.join(*i)
    
    if os.path.isdir(new_path) == False:
        #print(new_path)
        os.mkdir(new_path)
        
    return new_path
        
if __name__ == '__main__':
    from o_func import opsys3; start_path = opsys3()
    
    i = start_path, 'hello'
    md(i)
    
class Shutdown:
    def __init__(self, shutdown = 'n'):
        self.shutdown = shutdown
    
    def start_shutdown(self):
        def isNowInTimePeriod(startTime, endTime, nowTime): 
            if startTime < endTime: 
                return nowTime >= startTime and nowTime <= endTime 
            else: 
                #Over midnight: 
                return nowTime >= startTime or nowTime <= endTime 
    
        timey_wimey = isNowInTimePeriod(dt.time(20,30), dt.time(1,30), dt.datetime.now().time())
    
        if timey_wimey == True:
           shutdown = input("Do you wish to shutdown your computer afterwards? (y/n): ")
        else:
           shutdown = 'n'
           
        self.shutdown = shutdown
           
    def shutdown(self):
        if self.shutdown == 'y':
            os.system("shutdown /s /t 1")