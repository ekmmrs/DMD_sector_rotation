# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 18:17:46 2019

@author: Colton Smith & Eric Kammers
"""

# Dependencies
import numpy as np
import pandas as pd
%matplotlib qt5
import matplotlib.pyplot as plt
import DMD_sector_rotation_functions as fn

##################
### Query Data ###
##################

# Inputs for ETF Data
# Go to https://www.alphavantage.co/support/#api-key
# to claim your free API key
key = 'YOUR_API_KEY' # AlphaVantage API key
ETFs = np.array(['XLY','XLP','XLE','XLF','XLV','XLI','XLB','XLK','XLU']) # all ETFs used
time_ago = 20 * 12 # 20 years ago of monthly returns data

returns_data = fn.get_security_returns(key, ETFs, time_ago).dropna() # gets monthly security returns

##########################################
### Implement DMD Sector Rotation Algo ###
##########################################

# inputs for set_strategy
r = 9
vals = np.array(returns_data)
lookback = 14

# Calculate returns for strategy and benchmark
port_rets, bench_rets = fn.deploy_strategy(returns_data, r, lookback)

# get sharp ratio for both bench and strategy
port_sharpe = fn.get_sharp(port_rets)
print(port_sharpe)
bench_sharpe = fn.get_sharp(bench_rets)
print(bench_sharpe)

# Print both cumulative returns
plt.plot(bench_rets.cum_return)
plt.plot(port_rets.cum_return)
plt.show()

