#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 16:50:55 2016

@author: Jeiel
"""

import decisiontree as dt
from random import sample
from random import uniform
from math import ceil

def holdout(data, pencentage = 2/3):
    mask = sample(range(0, len(data)), ceil(len(data)*pencentage)) #without replacement
    traindata = [data[i] for i in mask]
    testdata = [data[i] for i in range(0, len(data)) if i not in mask]
    tree = dt.train()
    
#    return(traindata, testdata)

def bootstrap(data):
    mask = [round(uniform(0, len(data)-1)) for i in range(0, len(data))]
    #uniform:Return a random floating point number N such that a <= N <= b for a <= b and b <= N <= a for b < a.     
    traindata = [data[i] for i in mask] 
    testdata = data

    
def crossvalidation(data, kfold = 10):
    traindata = []
    leftrows = {i for i in range(0, len(data))}
    for i in range(0, kfold):
        if i < kfold - 1:
            mask = set(sample(range(0, len(leftrows)), len(leftrows)//10)) #without replacement
        else:
            mask = leftrows
        traindata += [[data[i] for i in mask]]
        leftrows -= mask
    

        
        
    
    
def classify():
    1==1
    

def test(data, featurenames, preprune = False, postprune = False, threshold = 0.0):
    1==1
    
    
