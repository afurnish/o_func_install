# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 13:00:04 2023

@author: aafur
"""
from o_func import opsys; start_path = opsys()
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.dates import MonthLocator, YearLocator, DateFormatter

def tide_gauge_loc():
    tgl = {'Heysham'  :{'x':-2.9311759, 'y':54.0345516},
           'Liverpool':{'x':-3.0168250, 'y':53.4307320},
           }
    return tgl


class Gauge:
    def __init__(self,file_path):
        self.file_path = file_path
    
    def read_gauge(self):
        # Read the data into a pandas DataFrame
        names = ['Cruise', 'Station','Type', 'Date','Time' ,'Longitude', 'Latitude', 'Bot_Depth', 'Depth', 'Pressure', 'Sea_pressure']
        self.df = pd.read_csv(self.file_path, delim_whitespace=True, skiprows=13, encoding='latin-1', names = names, index_col=False)  # Skip the first 15 rows as they contain metadata
        # Assuming df is your DataFrame with columns 'Date' and 'Time'
        self.df['Datetime'] = pd.to_datetime(self.df['Date'] + ' ' + self.df['Time'])
        self.df.set_index('Datetime', inplace=True)
        # Display the DataFrame
        print(self.df.head)
        
    def plot_gauge(self, path):
        print('depth is ',self.df['Depth'])
        # matplotlib.use('Qt5Agg')  # or 'TkAgg' or another interactive backend

        plt.figure(dpi=200)

        plt.plot(self.df.index, self.df['Depth'], linewidth=0.5)
        plt.ylabel('Depth (m)')
        
        ax = plt.gca()
        # ax.xaxis.set_major_locator(MonthLocator(interval=3))
        # ax.xaxis.set_minor_locator(YearLocator())
        # ax.xaxis.set_major_formatter(DateFormatter('%b %Y'))

        # Rotate x-axis labels for better visibility
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(path, bbox_inches='tight')

if __name__ == '__main__':
    main_path = start_path + 'Original_Data/SEARCH_river_data/2023-12-01_kent_estuary'
    gg = Gauge(main_path + '/205266_20231204_0846.txt')
    gg.read_gauge()
    gg.plot_gauge(main_path + '/plot.png')
    
    