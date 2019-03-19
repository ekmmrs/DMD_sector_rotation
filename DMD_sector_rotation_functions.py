# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 20:50:48 2019

@author: Colton Smith & Eric Kammers
"""

# dependencies
from pydmd import DMD
import time
import ffn
from alpha_vantage.timeseries import TimeSeries
import numpy as np
import pandas as pd

## Functions for DMD sector rotation strategy ##

## Gets Securities Monthly reurns data using AlphaVantage API ##
def get_security_returns(API_key, securities, time_ago):
    # Queries security data from a numpy array of securities using AlphaVantage API
    #################################################################################################################################
    # Inputs:                                                                                                                       #
    #       - API_key:          a string of your AlphaVantage API key                                                               #
    #       - securities:       numpy array containing strings of securities to query                                               #
    #       - years_ago:        an integer of how many years ago (from now) of returns data you would like to keep (e.g 1, 5, 10)   #
    #                                                                                                                               #
    # Outputs:                                                                                                                      #
    #       - returns_data:     pandas dataframe of monthly returns data of securities queried by AlphaVantage API                  #
    #################################################################################################################################
    # Create time series object with API_key
    ts = TimeSeries(key = API_key, 
                    output_format = 'pandas', 
                    indexing_type='date')
    
    # Create empty data frame where securities return data will be added
    returns_data = pd.DataFrame()
    # Loop through requested securities
    print('Querying Securities, Estimated time: ' + str(round(len(securities)/5)) + ' minutes')
    for x in range(len(securities)):
        # AlphaVantage limits 5 calls a minute, unless you pay for premium
        if (x + 1) % 5 == 0:
            print('Waiting 1 minute before making more Calls')
            time.sleep(60) # wait 1 minutes after every 5 API calls
            print('Making more calls') 
        # Make API Calls
        data, meta_data = ts.get_monthly_adjusted(symbol=str(securities[x])) # Get monthly adjusted price data
        data = data['5. adjusted close'].to_returns().iloc[::-1]             # Convert adjusted close to returns
        data = pd.DataFrame(data).rename(index=str, columns={'5. adjusted close' : str(securities[x])})
        # Concatenate queried security to returns DF
        returns_data = pd.concat([returns_data,data], axis=1, sort=False)
        
    # Take (time_ago) months of data and sort so that first row is the latest date of securities
    returns_data = returns_data.iloc[:(time_ago)].iloc[::-1]
    # Return dataframe of return values
    return returns_data

## Sets strategy and benchmark into codified matrices ##
def set_strategy(vals, r, lookback):
    # Sets sector strategy on returns data from DMD predictions
    # and returns 2 matrices: signals (based on DMD predictions), bench (baseline where strategy is long)
    #################################################################################################################################
    # Inputs:                                                                                                                       #
    #       - vals:         numpy array of returns                                                                                  #
    #       - r:            integer of svd_rank for DMD model (should be anywhere from 1 to the # of securities used in strategy)   #
    #       - lookback:     integer of timewindow size for training DMD model                                                       #
    #                                                                                                                               #
    # Outputs:                                                                                                                      #
    #       - signals:      codified pandas dataframe indicating whether to go long or short on security at each time step          #
    #       - bench:        codified pandas dataframe indicating a benchmark of holding securities                                  #
    #################################################################################################################################
    # Create copies of original data
    signals = vals.copy()
    bench = vals.copy()
    
    # Create DMD predictions
    for i in range(lookback,len(vals)-1):
        dmd = DMD(svd_rank = r)                                         # create DMD object with specified svd_rank, r
        vals_sub = vals[i-lookback:i+1,:]                               # select slice of data
        dmd.fit(vals_sub.T)                                             # fit DMD model by time window from returns data
        dmd.dmd_time['tend'] *= (1+1/(lookback+1))
        signals[i+1,:] = dmd.reconstructed_data.real.T[lookback+1,:]    # predict 1 time step into future
    
    # Set any predictions that are nan to zero
    signals[np.isnan(signals)] = 0
    
    # np array of whether predictions are bad or not
    # A bad prediction is where all predictions were nan (then changed to 0) 
    badp = (np.sum(signals,axis=1) == 0)
    
    # Initilize initial training window to be going long for both benchmark and DMD strategy
    for i in range(0,lookback+1):
        for j in range(0,signals.shape[1]):
            bench[i,j] = 1
            signals[i,j] = 1
    
    for i in range(lookback+1, signals.shape[0]):
        bench[i,:] = 1                      # benchmark is going long
        row = signals[i,:]                  # select row of predicted returns from portfolio
        if (badp[i]):                       # if the prediction is deemed bad
            signals[i,:] = signals[i-1,:]   # carry over the same prediction from the last state
        else:
            median = np.median(row)         # identify median predicted returns for portfolio
            for j in range(0, len(row)):
                if (row[j] >= median):      # go long on security if predicted 
                    row[j] = 1              # returns is greater than median
                else:                       # go short on security if predicted 
                    row[j] = -1             # returns is less than median
            signals[i,:] = row              # codify strategy into signals

    # print out the number of bad predictions that are made
    print('Bad Predictions: ', sum(badp))
    
    # return codified matrices
    return signals, bench

## Gets returns data of portfolio ##
def get_port_ret(rets, dates):
    # Calculates mean returns of portfolio and cumulative returns
    #################################################################################################################################
    # Inputs:                                                                                                                       #
    #       - rets:          numpy array of returns                                                                                 #
    #       - dates:         pandas index from returns data frame                                                                   #
    #                                                                                                                               #
    # Outputs:                                                                                                                      #
    #       - port_rets:     numpy array of portfolio returns and cumulative returns                                                #
    #################################################################################################################################

    # calculate mean returns of portfolio with strategy
    port_rets = pd.DataFrame(index = dates)
    port_rets['Ret'] = 0.0
    for i in range(0,rets.shape[0]):
        port_rets.Ret[i] = np.mean(rets[i,:])
        
    # calculate cumulative returns of strategy
    port_rets['cum_return'] = np.exp(np.log1p(port_rets.Ret).cumsum())
    return port_rets

## Get Sharp Ratio ##
def get_sharp(port_rets):
    # Calculates sharpe ratio
    #################################################################################################################################
    # Inputs:                                                                                                                       #
    #       - port_rets:     numpy array of portfolio returns                                                                       #
    #                                                                                                                               #
    # Outputs:                                                                                                                      #
    #       - sharp ratio                                                                                                           #           
    #################################################################################################################################
    return (port_rets.Ret.mean()/port_rets.Ret.std())*np.sqrt(12)

# combines above def's to simplify DMD sector rotation strategy
def deploy_strategy(returns_data, r, lookback):
    # Deploys DMD Sector rotation strategy and returns portfolio returns for both benchmark and strategy
    #####################################################################################################################################
    # Inputs:                                                                                                                           #
    #       - returns_data:     pandas dataframe of monthly returns data of securities queried by AlphaVantage API                      #                                              #
    #       - r:                integer of svd_rank for DMD model (should be anywhere from 1 to the # of securities used in strategy)   #
    #       - lookback:         integer of timewindow size for training DMD model                                                       #
    #                                                                                                                                   #
    # Outputs:                                                                                                                          #
    #       - port_rets, bench_rets: returns and cumulative returns of both benchmark and strategy                                      #                                                                         #           
    #####################################################################################################################################
    vals = np.array(returns_data)
    signals, bench = set_strategy(vals, r, lookback)

    # multiply codified strategy of benchmark 
    rets = np.multiply(vals, signals)
    b_rets = np.multiply(vals, bench)
    
    # calculate returns
    port_rets = get_port_ret(rets, returns_data.index)
    bench_rets = get_port_ret(b_rets, returns_data.index)
    
    return port_rets, bench_rets