#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 19:38:43 2016

@author: Jeiel
"""

import decisiontree as dt

if __name__ == '__main__':
    print('training...')
    data = []   
    with open('abalone.data', 'r') as f:
        data = [line.split(',') for line in f]
    
    data = [[row[0]]+list(map(float,row[1:8]))+[int(row[-1])] for row in data]
            
    featurenames = ['Sex', 'Length', 'Diameter', 'Height' ,'Whole weight', 'Shucked weight' , 'Viscera weight', 'Shell weight' ,'Rings'] #'Ring' 就是'Class'
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