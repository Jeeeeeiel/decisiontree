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

def holdout(data, pencentage = 2/3, featurenames = None, method = 'gini', adaboostOn = False, k = 10, preprune = False, postprune = False, threshold = 0.0):
    print('holdout:')
    print('training...')
    
    mask = sample(range(0, len(data)), ceil(len(data)*pencentage)) #without replacement
    traindata = [data[i] for i in mask]
    testdata = [data[i] for i in range(0, len(data)) if i not in mask]
    errorcount = 0
    if adaboostOn:
        (classifiers,alpha) = dt.adaboost(traindata, featurenames, method, k = 10, preprune = preprune, postprune = postprune, threshold = threshold)
        errorcount = dt.classifydataforclassifier(classifiers, alpha, testdata, featurenames)[1]
    else:
        tree = dt.train(traindata, featurenames, method, preprune = preprune, postprune = postprune, threshold = threshold)
        errorcount = dt.classifydata(tree, testdata, featurenames)[1]
    
    acc = 1 - errorcount/len(testdata)
    print('holdout acc: ', acc)
    

def bootstrap(data, bootstrap = 10, featurenames = None, method = 'gini', adaboostOn = False, k = 10, preprune = False, postprune = False, threshold = 0.0):
    print('bootstrap:')
    print('training...')
    
    acc = []
    for i in range(0, bootstrap):
        mask = [round(uniform(0, len(data)-1)) for i in range(0, len(data))]    #around 63.2% records
        #uniform:Return a random floating point number N such that a <= N <= b for a <= b and b <= N <= a for b < a.     
        traindata = [data[i] for i in mask] 
        testdata = [data[i] for i in range(0, len(data)) if i not in mask]
        errorcount = 0
        errorcountforwhole = 0
        if adaboostOn:
            (classifiers,alpha) = dt.adaboost(traindata, featurenames, method, k = 10, preprune = preprune, postprune = postprune, threshold = threshold)
            errorcount = dt.classifydataforclassifier(classifiers, alpha, testdata, featurenames)[1]
            errorcountforwhole = dt.classifydataforclassifier(classifiers, alpha, data, featurenames)[1]
        else:
            tree = dt.train(traindata, featurenames, method, preprune = preprune, postprune = postprune, threshold = threshold) 
            errorcount = dt.classifydata(tree, testdata, featurenames)[1]
            errorcountforwhole = dt.classifydata(tree, data, featurenames)[1]
        acc.append(0.632 * (1 - errorcount / len(testdata)) + 0.368 * (1 - errorcount / len(data)))
    print('not finish : acc_s')
    print('bootstrap( b =', bootstrap, ') acc: ', mean(acc))

    
def crossvalidation(data, kfold = 10, featurenames = None, method = 'gini', adaboostOn = False, k = 10, preprune = False, postprune = False, threshold = 0.0):
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
        errorcount = 0
        if adaboostOn:
            (classifiers,alpha) = dt.adaboost(traindata, featurenames, method, k = 10, preprune = preprune, postprune = postprune, threshold = threshold)
            errorcount = dt.classifydataforclassifier(classifiers, alpha, testdata, featurenames)[1]
        else:
            tree = dt.train(traindata, featurenames, method, preprune = preprune, postprune = postprune, threshold = threshold)
            errorcount = dt.classifydata(tree, testdata, featurenames)[1]
        acc.append(1 - errorcount/len(testdata))
    print('crossvalidation(', kfold, '- fold) acc: ', mean(acc))


def test(data = None, featurenames = None, method = 'gini', adaboostOn = False, k = 10, preprune = False, postprune = False, threshold = 0.0):
    print()
    print('adaboostOn: ', adaboostOn)
    holdout(data = data, pencentage = 2/3, featurenames = featurenames, method = 'gini', adaboostOn = adaboostOn, k = k, preprune = preprune, postprune = postprune, threshold = threshold)
    print()
    crossvalidation(data = data, kfold = 10, featurenames = featurenames, method = 'gini', adaboostOn = adaboostOn, k = k, preprune = preprune, postprune = postprune, threshold = threshold)
    print()
    bootstrap(data, bootstrap = 10, featurenames = featurenames, method = 'gini', adaboostOn = adaboostOn, k = k, preprune = preprune, postprune = postprune, threshold = threshold)
