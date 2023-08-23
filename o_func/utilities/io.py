#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import platform
import os
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