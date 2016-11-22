#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 12:43:44 2016

@author: Jeiel
"""


from evaluation import test
import decisiontree as dt

if __name__ == '__main__':
    print('file: car.data')
    
    data = [[2,0.1,1],
            [3,0.2,1],
            [5,0.3,1],
            [0,0.4,-1],
            [2,0.5,-1],
            [5,0.6,-1],
            [4,0.7,-1],
            [3,0.8,1],
            [4,0.9,1],
            [2,1.0,1],
            [1,1.1,1],
            [0,1.2,1],
            [3,1.3,1],
            [4,1.4,-1],
            [6,1.5,-1],
            [2,1.6,-1],
            [1,1.7,-1],
            [1,1.8,1],
            [2,1.9,1],
            [3,2.0,1]]
            
    
#    print(data)            

    featurenames = ['x', 'y']
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
    
#    test(data = data, featurenames = featurenames, adaboostOn = False, preprune = False, postprune = False, threshold = 0.1)

    
#    tree = dt.train(data, featurenames, method, preprune = False, postprune = False, threshold = 0.1)
#    errorcount = dt.classifydata(tree, data, featurenames)[1]
#    acc = 1 - errorcount/len(data)
#    print('acc: ', acc)
#    
#    
#    (classifiers,alpha) = dt.adaboost(data, featurenames, method, k = 100, preprune = False, postprune = False, threshold = 0.1)
#    errorcount = dt.classifydataforclassifier(classifiers, alpha, data, featurenames)[1]
#    acc = 1 - errorcount/len(data)
#    print('adaboost acc: ', acc)
#    print('done')
    test(data = data, featurenames = featurenames, adaboostOn = True, k=100, preprune = False, postprune = False, threshold = 0.1)