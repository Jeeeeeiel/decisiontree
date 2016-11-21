#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 19:51:13 2016

@author: Jeiel
"""
from evaluation import test

if __name__ == '__main__':
    print('file: balance-scale.data')
    data = []   
    with open('balance-scale.data', 'r') as f:
        data = [line.split(',') for line in f]
    
    data = [list(map(int,row[1:8]))+[row[0]] for row in data]

    featurenames = ['Left-Weight', 'Left-Distanc', 'Right-Weight', 'Right-Distance', 'Class Name'] 
    method = 'gini' #'gini','entropy','classificationerror'
    
    test(data = data, featurenames = featurenames, adaboostOn = False, preprune = False, postprune = False, threshold = 0.5)
    
    print('done')