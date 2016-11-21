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
from statistics import mean

def holdout(data, pencentage = 2/3, featurenames = None, method = 'gini', preprune = False, postprune = False, threshold = 0.0):
    print('holdout:')
    print('training...')
    mask = sample(range(0, len(data)), ceil(len(data)*pencentage)) #without replacement
    traindata = [data[i] for i in mask]
    testdata = [data[i] for i in range(0, len(data)) if i not in mask]
    tree = dt.train(traindata, featurenames, method, preprune = False, postprune = False, threshold = 0.0)
    
    errorcount = classifydata(tree, testdata, featurenames)[1]
    acc = 1 - errorcount/len(testdata)
    print('holdout acc: ', acc)
    

def bootstrap(data):
    print('bootstrap:')
    print('training...')
    mask = [round(uniform(0, len(data)-1)) for i in range(0, len(data))]    #around 63.2% records
    #uniform:Return a random floating point number N such that a <= N <= b for a <= b and b <= N <= a for b < a.     
    traindata = [data[i] for i in mask] 
    testdata = [data[i] for i in range(0, len(data)) if i not in mask]

    
def crossvalidation(data, kfold = 10, featurenames = None, method = 'gini', preprune = False, postprune = False, threshold = 0.0):
    print('crossvalidation(k-fold):')
    print('training...')
    datasplit = [] #[[[obj],...,[obj]],...,[[obj],...,[obj]]]
    leftrows = {i for i in range(0, len(data))}
    for i in range(0, kfold):
        if i < kfold - 1:
            mask = set(sample(range(0, len(leftrows)), len(leftrows)//10)) #without replacement
        else:
            mask = leftrows
        datasplit += [[data[i] for i in mask]]
        leftrows -= mask

    acc = []
    for i in range(0, kfold):
        traindata = []
        for j in range(0, kfold):
            if j != i:
                traindata +=datasplit[j]  #[[obj],...,[obj]]
        testdata = datasplit[i] #[[obj],...,[obj]]
        for j in range(0, kfold - 1):
            tree = dt.train(traindata, featurenames, method, preprune = False, postprune = False, threshold = 0.0)
            errorcount = classifydata(tree, testdata, featurenames)[1]
            acc.append(1 - errorcount/len(testdata))
    print('crossvalidation(', kfold, '- fold) acc: ', mean(acc))
        
    
    
def classifydata(tree, testdata, featurenames):
    result = []
    errorcount = 0
    for i in range(0, len(testdata)):
        result.append(dt.classifyobj(tree, testdata[i], featurenames))
        if testdata[i][-1] != result[-1]:
            errorcount += 1
    return(result,errorcount)

def test(data = None, featurenames = None, method = 'gini', preprune = False, postprune = False, threshold = 0.0):
    holdout(data = data, pencentage = 2/3, featurenames = featurenames, method = 'gini', preprune = preprune, postprune = postprune, threshold = threshold)
    print()
    crossvalidation(data = data, kfold = 10, featurenames = featurenames, method = 'gini', preprune = preprune, postprune = postprune, threshold = threshold)
    
