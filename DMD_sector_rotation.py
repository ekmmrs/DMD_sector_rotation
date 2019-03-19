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
key = '94Q8IKY6D04WOYKZ' #'YOUR_API_KEY' # AlphaVantage API key
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

# Get sharp ratio for both benchmark and strategy
port_sharpe = fn.get_sharp(port_rets)
print(port_sharpe)
bench_sharpe = fn.get_sharp(bench_rets)
print(bench_sharpe)

# Make plot comparing Benchmark and Strategy
params = {'legend.fontsize': 20,
          'legend.handlelength': 2}
plt.rcParams.update(params)

fig, ax = plt.subplots()
ax.plot(bench_rets.cum_return, label = 'Long All Sectors, Sharpe = ' + str(round(bench_sharpe,2)), linewidth=4)
ax.plot(port_rets.cum_return, label = 'LS Sector Rotation, Sharpe = ' + str(round(port_sharpe,2)), linewidth=4)
ax.legend()
ax.set_ylabel('Cumulative Return')
ax.set_xlabel('Time')
ax.set_xticks(ax.get_xticks()[::12])
plt.show()

