# -*- coding: utf-8 -*-
"""
@author: sarthak
"""
# Importing the packages

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the data from statsmodels

data = sm.datasets.co2.load_pandas()
co2 = data.data
print(co2.head())

# Indexing the time series data 
# Important : When working with time series data we must always ensure that the 
# dates should be used as index

co2.index

# the dtype = datetime[ns] in the output confirms that the index is made of
# date stamp objects while length = 2284 and freq = 'W-SAT' shows that we have
# 2285 weekly date stamps starting on wednesday

y = co2['co2'].resample('MS').mean()
y.head(5)

# handling missing data: 
y.isnull().sum()

# The output gives 5 missing values which means 5 months of data is missing
# The missing values can be dropped, filled by fillna() command or rolling mean
y = y.fillna(y.bfill())

# Visualizing the time series data: 
y.plot(figsize = (15,6))
plt.show()

# Here some distinguishable features appear like seasonality and overall increasing 
# trend

# Statsmodels provides seasonal_decompose function to perform seasonal decomposition
# out of the box

from pylab import rcParams
rcParams['figure.figsize'] = 11,9

decomposition = sm.tsa.seasonal_decompose(y, model = 'additive')
fig = decomposition.plot()
plt.show()

# Using time series decomposition it is easier to quickly identify a changing 
# mean or variation in the data. 
