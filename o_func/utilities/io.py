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
    from o_func import opsys; start_path = opsys()
    
    i = start_path, 'hello'
    md(i)
    
    
def compare_files(file1_path, file2_path):
    # Read the contents of both files
    with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2:
        content1 = file1.readlines()
        content2 = file2.readlines()

    # Compare the contents line by line
    for i, (line1, line2) in enumerate(zip(content1, content2)):
        if line1 != line2:
            print(f"Difference found at line {i + 1}:")
            print(f"File 1: {line1.strip()}")
            print(f"File 2: {line2.strip()}\n")

    # Check for extra lines in either file
    if len(content1) > len(content2):
        print(f"File 1 has {len(content1) - len(content2)} more lines.")
    elif len(content1) < len(content2):
        print(f"File 2 has {len(content2) - len(content1)} more lines.")
    else:
        print("Both files are identical.")
    
class Shutdown:
    def __init__(self):
        self.shutdown = 'n'
        
    @staticmethod
    def isNowInTimePeriod(startTime, endTime, nowTime): 
        if startTime < endTime: 
            return nowTime >= startTime and nowTime <= endTime 
        else: 
            #Over midnight: 
            return nowTime >= startTime or nowTime <= endTime 
 
    def start_shutdown(self):
        timey_wimey = self.isNowInTimePeriod(dt.time(20,30), dt.time(1,30), dt.datetime.now().time())
        
        if timey_wimey == True:
           self.shutdown = input("Do you wish to shutdown your computer afterwards? (y/n): ")
        else:
           self.shutdown = 'n'
       
           
    def kill(self):
        if self.shutdown == 'y':
            os.system("shutdown /s /t 1")