#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 19:51:14 2016

@author: Jeiel
"""

import decisiontree as dt
from evaluation import test

#Attribute Values:
#
#buying       v-high, high, med, low
#maint        v-high, high, med, low
#doors        2, 3, 4, 5-more
#persons      2, 4, more
#lug_boot     small, med, big
#safety       low, med, high

#Class Distribution (number of instances per class)
#
#class      N          N[%]
#-----------------------------
#unacc     1210     (70.023 %) 
#acc        384     (22.222 %) 
#good        69     ( 3.993 %) 
#v-good      65     ( 3.762 %)


if __name__ == '__main__':
    print('test data: car.data')
    
    data = []   
    with open('car.data', 'r') as f:
        data = [line.split(',') for line in f]
    
    values = {}
    values[0] = ['low', 'med', 'high', 'vhigh']
    values[1] = ['low', 'med', 'high', 'vhigh']
    values[2] = ['2', '3', '4', '5more']
    values[3] = ['2', '4', 'more']
    values[4] = ['small', 'med', 'big']
    values[5] = ['low', 'med', 'high']
#    values[7] = {'unacc';'acc';'good','vgood'};
    for i in range(0, len(data)):
        for j in range(0, len(values)):
            data[i][j] = values[j].index(data[i][j])
            
    data = [list(map(int,row[:-1]))+[row[-1][:-1]] for row in data]
#    print(data)            

    featurenames = ['buying', 'maint', 'doors', 'persons' ,'lug_boot', 'safety', 'class']
    method = 'gini' #'gini','entropy','classificationerror'
    
#    tree = dt.train(data, featurenames, method)
#    
#    errorcount = 0
#    for row in data:
#        row.append(dt.classifyobj(tree, row, featurenames))
#        if row[-2] != row[-1]:
#            errorcount += 1
#    
#    accuracy = 1 - errorcount / len(data)
#    print('accuracy: ', accuracy)
    test(data = data, featurenames = featurenames, preprune = False, postprune = False, threshold = 0.0)

    print('done')