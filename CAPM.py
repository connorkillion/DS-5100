#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 11:09:26 2021

@author: connorkillion
"""
# load modules
import numpy as np
import pandas as pd

# risk-free Treasury rate
R_f = 0.0175 / 252
# read in the market data
data = pd.read_csv('capm_market_data.csv')
data.head()
df = data.copy()
del df['date']
df
returns = df.pct_change(axis=0)
returns.dropna(inplace=True)
returns.head()
returns.head(5)
spy = returns.spy_adj_close.values
print(spy[:5])
aapl = returns.aapl_adj_close.values
print(aapl[:5])
aapl_xs = aapl-R_f
spy_xs = spy-R_f
print(aapl_xs[-5:])
print(spy_xs[-5:])
import matplotlib.pyplot as plt
plt.scatter(spy_xs, aapl_xs)
plt.grid()
y = spy_xs.reshape(-1,1)
x = aapl_xs.reshape(-1,1)

xtx = np.matmul(x.transpose(), x)
xtxi = np.linalg.inv(xtx)
xtxixt = np.matmul(xtxi, x.transpose())
beta = np.matmul(xtxixt, y)
beta_hat = np.matmul(xtxixt, y)[0][0]
print('beta is: ')
print(beta_hat)
def beta_sensitivity(x,y):
    out = []
    sz = x.shape[0]
    for ix in range(sz):
        xx = np.delete(x, ix).reshape(-1,1) 
        yy = np.delete(y, ix).reshape(-1,1) 
        bi = np.matmul(np.matmul(np.linalg.inv(np.matmul(xx.transpose(),xx)),xx.transpose()),yy)[0][0]
        tup = (ix, bi)
        out.append(tup)
    
    return out
ret = beta_sensitivity(x,y)
print(ret[:5])
