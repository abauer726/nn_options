#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
naming scheme: modelname_optiontype+dimension
"""

import numpy as np


#M1 through M8 are sourced through mlOSP paper and named the same

m1_put1 = {'dim': 1, 'K': 40, 'x0': np.repeat(40, 1), 'sigma': np.repeat(0.2,1), 
          'r': 0.06, 'div': np.repeat(0, 1), 'T': 1, 'dt': 0.04, 'payoff.func': 'put.payoff'} 

m2_put1 = {'dim': 1, 'K': 40, 'x0': np.repeat(44, 1), 'sigma': np.repeat(0.2,1), 
          'r': 0.06, 'div': np.repeat(0, 1), 'T': 1, 'dt': 0.04, 'payoff.func': 'put.payoff'} 

m3_put2 = {'dim': 2, 'K': 40, 'x0': np.repeat(40, 2), 'sigma': np.repeat(0.2,2), 
          'r': 0.06, 'div': np.repeat(0, 2), 'T': 1, 'dt': 0.04, 'payoff.func': 'put.payoff'} 

m4_call2 = {'dim': 2, 'K': 100, 'x0': np.repeat(110, 2), 'sigma': np.repeat(0.2,2), 
          'r': 0.05, 'div': np.repeat(0.1, 2), 'T': 3, 'dt': 1/3, 'payoff.func': 'maxi.call.payoff'} 

# M5 not utilized 

m6_call3 = {'dim': 3, 'K': 100, 'x0': np.repeat(90, 3), 'sigma': np.repeat(0.2,3), 
          'r': 0.05, 'div': np.repeat(0.1, 3), 'T': 3, 'dt': 1/3, 'payoff.func': 'maxi.call.payoff'}

m7_call5 = {'dim': 5, 'K': 100, 'x0': np.repeat(100, 5), 'sigma': np.repeat(0.2,5), 
          'r': 0.05, 'div': np.repeat(0.1, 5), 'T': 3, 'dt': 1/3, 'payoff.func': 'maxi.call.payoff'} 

m8_call5 = {'dim': 5, 'K': 100, 'x0': np.repeat(70, 5), 'sigma': [0.08,0.16,0.24,0.32,0.4], 
          'r': 0.05, 'div': np.repeat(0.1, 5), 'T': 3, 'dt': 1/3, 'payoff.func': 'maxi.call.payoff'} 

# M9 not utilized

# M10 and M11 are Cosmin's 2d max call and 1d put

m10_put1 = {'dim': 1, 'K': 40, 'x0': np.repeat(40, 1), 'sigma': np.repeat(0.2,1), 
          'r': 0.06, 'div': np.repeat(0, 1), 'T': 1, 'dt': 0.04, 'payoff.func': 'put.payoff'} # cosmin: bput1


m11_call2 = {'dim': 5, 'K': 100, 'x0': np.repeat(70, 5), 'sigma': [0.08,0.16,0.24,0.32,0.4], 
          'r': 0.05, 'div': np.repeat(0.1,5), 'T': 3, 'dt': (1/3), 'payoff.func': 'maxi.call.payoff'} # cosmin: mcall1


