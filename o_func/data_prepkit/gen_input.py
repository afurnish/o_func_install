# -*- coding: utf-8 -*-
""" Input generator for Delft 3d fm and EBM model grids 

Created on Wed Aug 16 17:33:40 2023
@author: aafur
"""
import matplotlib.pyplot as plt

class DataProcessor:
    def __init__(self, raw_data):
        self.raw_data = raw_data
        self.cleaned_data = None
        self.transformed_data = None
        self.analysis_results = None

    def clean_data(self):
        # Placeholder for data cleaning logic
        # This method could remove duplicates, handle missing values, etc.
        new_data = sorted(item for item in self.raw_data if item is not None)
        self.cleaned_data = new_data

    def transform_data(self):
        # Placeholder for data transformation logic
        # This method could perform data normalization, feature engineering, etc.
        self.transformed_data = self.cleaned_data
        print('printing self.cleaned_data: ',self.cleaned_data)
        print('printing type self.cleaned_data: ',type(self.cleaned_data))
        self.transformed_data2 = [i+10 for i in self.cleaned_data]

    def analyze_data(self):
        # Placeholder for data analysis logic
        # This method could perform statistical calculations, machine learning, etc.
        self.analysis_results = "Analysis results here"
        
    def print_out(self, dpi):
        fig, ax = plt.subplots(nrows=3,ncols=1, dpi = dpi)
        
        print('What are we doing here:')
        print('What are we doing here:', self.cleaned_data)
        print('What are we doing here:', self.transformed_data)
        print('What are we doing here:', self.analysis_results)
        ax[0].plot(self.cleaned_data)
        ax[1].plot(self.transformed_data)
        ax[2].plot(self.transformed_data2)
        
        y_min = min(min(self.cleaned_data), min(self.transformed_data), min(self.transformed_data2))
        y_max = max(max(self.cleaned_data), max(self.transformed_data), max(self.transformed_data2))

        for ax in ax:
            ax.set_ylim(y_min, y_max)
            
        plt.subplots_adjust(hspace=0.5)  # Adjust this value as needed

        #ax(3).plot(self.analysis_results)
            
    def process_and_print(self, dataset, dpi=100):
       # Creating an instance of the DataProcessor class
       dp = DataProcessor(dataset)
       
       # Cleaning, transforming, and analyzing the data
       dp.clean_data()
       dp.transform_data()
       dp.analyze_data()
       
       # Printing and plotting the processed data
       dp.print_out(dpi=dpi)
       
       # Accessing the processed data and analysis results
       print("Raw Data:", dataset)
       print("Cleaned Data:", dp.cleaned_data)
       print("Transformed Data:", dp.transformed_data)
       print("Transformed Data 2:", dp.transformed_data2)
       print("Analysis Results:", dp.analysis_results)

# Creating an instance of the DataProcessor class
raw_data = [1, 2, 3, 4, 5, 3, 2, None, 7, 6]
raw_data2 = list(range(50))
raw_data3 = [4,6,7,8]

# List of raw data sets
raw_data_sets = [
    [1, 2, 3, 4, 5, 3, 2, None, 7, 6],
    list(range(50)),
    [4, 6, 7, 8]
]

# Process and print each dataset
for dataset in raw_data_sets:
    data_processor = DataProcessor(dataset)
    data_processor.process_and_print(dataset, dpi=300)