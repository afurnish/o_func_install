# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 12:58:38 2023

@author: aafur
"""
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

import time

def printy_shit(i):
    print(i)
    

data = range(10000)  # List of image paths
start_time = time.time()

with ThreadPoolExecutor(max_workers=4
                        ) as executor:
    executor.map(printy_shit, data)
    
end_time = time.time()

print("ThreadPoolExecutor took {:.4f} seconds".format(end_time - start_time))


# start_time = time.time()

# with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
#     pool.map(printy_shit, data)

# end_time = time.time()

# print("Multiprocessing took {:.4f} seconds".format(end_time - start_time))