#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 19:38:43 2016

@author: Jeiel
"""
from evaluation import test

if __name__ == '__main__':
    print('file: abalone.data')
    data = []   
    with open('abalone.data', 'r') as f:
        data = [line.split(',') for line in f]
    
    data = [[row[0]]+list(map(float,row[1:8]))+[int(row[-1])] for row in data]
            
    featurenames = ['Sex', 'Length', 'Diameter', 'Height' ,'Whole weight', 'Shucked weight' , 'Viscera weight', 'Shell weight' ,'Rings'] #'Ring' 就是'Class'
    method = 'entropy' #'gini','entropy','classificationerror'
    
    test(data = data, featurenames = featurenames, adaboostOn = True, k=50, preprune = False, postprune = False, threshold = 0.1)
    
    print('done')