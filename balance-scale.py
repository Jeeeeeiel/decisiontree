#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 19:51:13 2016

@author: Jeiel
"""

import decisiontree as dt

if __name__ == '__main__':
    print('training...')
    data = []   
    with open('balance-scale.data', 'r') as f:
        data = [line.split(',') for line in f]
    
    data = [list(map(int,row[1:8]))+[row[0]] for row in data]

    featurenames = ['Left-Weight', 'Left-Distanc', 'Right-Weight', 'Right-Distance', 'Class Name'] 
    method = 'gini' #'gini','entropy','classificationerror'
    
    tree = dt.train(data, featurenames, method)
    
    errorcount = 0
    for row in data:
        row.append(dt.classifyobj(tree, row, featurenames))
        if row[-2] != row[-1]:
            errorcount += 1
    
    accuracy = 1 - errorcount / len(data)
    print('accuracy: ', accuracy)
    
    print('done')