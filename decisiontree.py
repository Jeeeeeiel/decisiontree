#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 21:54:42 2016

@author: Jeiel
"""

class DecisionNode:
    def __init__(self, feature = None, impurity = None, cond = None, leftchild = None, rightchild = None, nodetype = None, label = None, method = None):

        self.cond = cond	#numeric,<= cond represent leftchild;nominal, in cond represent leftchild
        self.feature = feature
        self.leftchild = leftchild
        self.rightchild = rightchild
        self.nodetype = nodetype
        self.label = label
        self.impurity = impurity
        

#calculate impurity 
def calcimpurity(data, method):
    from math import log2
    from collections import Counter
    
    total = len(data)
    categorydict = Counter([row[-1] for row in data])
    categorydict = {key:categorydict[key]/total for key in categorydict.keys()} #percentage
    impurity = None
    if method == 'gini':
        impurity = 1 - sum([categorydict[key]**2 for key in categorydict.keys()])
    elif method == 'entropy':
        impurity = - sum([categorydict[key]*log2(categorydict[key]) for key in categorydict.keys()])
    elif method == 'classificationerror':
        impurity = 1 - max(categorydict.values())
    else:
        print('no method match: ',method)
    return(impurity)
    
    

#divide data set on specific column(featureindex) according to value
def dividedata(data, col, value):
    split_function = None
    if isinstance(value, int) or isinstance(value,float):
        split_function = lambda row:row[col] <= value
    else:
        split_function = lambda row:row[col] in value
    
    data1 = [row for row in data if split_function(row)]
    data2 = [row for row in data if not split_function(row)]
    return(data1,data2)
    
    
def findbestsplitattr(parentimpurity, data, method):
    igdict = {}
    for i in range(0,len(data[0])-1):   #exclude category column
#        print('col: ', i, end=' ')
        value,impurity = findbestcond(data, i, method)
        igdict[i] = (value, parentimpurity-impurity) #binary tuple
    max_ig = max(igdict.items(),key = lambda item:item[1][1]) #find the item with biggest ig
    
#    print('bestattr: ', max_ig[0], ', cond: ', max_ig[1][0], ', impurity: ', impurity)
    return(max_ig[0], max_ig[1][0],igdict[1][1]) #index,value,ig
    
def findbestcond(data, col, method):     #return value,impurity
    from collections import Counter
    from itertools import combinations
    from math import ceil
    condimpuritydict = {}
    valuedict = Counter([row[col] for row in data])
    
    if isinstance(list(valuedict.keys())[0], int) or isinstance(list(valuedict.keys())[0], float):    #numeric
        keylist = sorted(valuedict.keys())
#        print('distinct value size(numeric): ', len(keylist))
        for i in range(1, len(keylist)): #get every split value
            mean = (keylist[i-1] + keylist[i])/2   #i start form 1, i-1 avoid only one distinct attr
            (data1, data2) = dividedata(data, col, mean)
            condimpuritydict[mean] = (calcimpurity(data1, method)*len(data1) + calcimpurity(data2, method)*len(data2))/len(data)
    else:   #nominal
        valuesetlist = []    
        for size in range(1, ceil(len(valuedict)/2)+1):
            valuesetlist += list(combinations(valuedict.keys(), size))
#        print('distinct value size(nominal): ', len(valuesetlist))
#        print('valuedict: ', valuedict)
#        print('valuesetlist:', valuesetlist)
#        if len(data[0]) < 5:
#            print('data: ', data)
        for valueset in valuesetlist:
            (data1, data2) = dividedata(data, col, valueset)
            condimpuritydict[valueset] = (calcimpurity(data1, method)*len(data1) + calcimpurity(data2, method)*len(data2))/len(data)
#        if len(data[0]) < 5:
#            print('data: ', data)
#            print('condimpuritydict: ', condimpuritydict)
        
    return(min(condimpuritydict.items(), key = lambda item:item[1])) #return the key with smallest value


def treegrowth(data, featureleft, method, preprune, postprune, threshold):
    if teststop(data):
        node = DecisionNode(impurity = calcimpurity(data,method), nodetype = 'leaf', method = method)
        node.label = classify(data)
        return(node)
    else:
        node = DecisionNode(impurity = calcimpurity(data,method),nodetype = 'node', method = method)
#        print('featuresize(exclude category): ', len(featureleft) - 1)
#        print('data size(exclude category):', len(data[0]) - 1)
        (featureindex, node.cond, ig) = findbestsplitattr(node.impurity, data, method)
        if preprune & (ig < threshold): #when preprune on
            node.nodetype = 'leaf'
            node.label = classify(data)
            return(node)
        node.feature = featureleft[featureindex]
        
        #remove feature used!!
        featureleft = [feature for feature in featureleft if feature != node.feature]

        (data1, data2) = dividedata(data, featureindex, node.cond)
#        if len(data) <=4:
#            print('data1: ', data1)
#            print('data2: ', data2)
        #remove column in data
        data1 = [[row[i] for i in range(0, len(data1[0])) if i != featureindex] for row in data1 ]
        data2 = [[row[i] for i in range(0, len(data2[0])) if i != featureindex] for row in data2 ]
        
        
        node.leftchild = treegrowth(data1, featureleft, method, preprune, postprune, threshold)
        node.rightchild = treegrowth(data2, featureleft, method, preprune, postprune, threshold)
        return(node)
        
def teststop(data):
    if len({row[-1] for row in data }) == 1:    #same category
        return True;
    elif len({tuple(row[:-2]) for row in data }) == 1:   #same attrs
        return True;
    elif len(data[0]) == 1:   #no attr left except category
        return True;
    else:
        return False;
        

def classify(data):
    from collections import Counter
    categorydict = Counter([row[-1] for row in data ])
#    print('classify: ', data)
#    print('classify: ', categorydict)
    category = max(categorydict.keys(), key = lambda k:categorydict[k])
    return(category)
    
def train(data, featurenames, method, preprune = False, postprune = False, threshold = 0.0):
    
    tree = treegrowth(data, featurenames, method, preprune, postprune, threshold)
    if postprune:
        tree = postpruning(tree, threshold)
    return tree

def postpruning(tree, threshold):  
    print('not defined!!!')

def printtree(tree): #queue
    from collections import deque
    nodequeue = deque()
    nodequeue.append(tree)
    
    while len(nodequeue) > 0:
        node = nodequeue.popleft()
        if node.nodetype != 'leaf':
            nodequeue.append(node.leftchild)
            nodequeue.append(node.rightchild)
            print(node.feature,'(cond: ', node.cond,')', end = ' | ')
        else:
            print('label: ', node.label, end = ' | ')
 
def countleaf(tree):
    from collections import deque
    nodequeue = deque()
    nodequeue.append(tree)
    count = 0
    while len(nodequeue) > 0:
        node = nodequeue.popleft()
        if node.nodetype != 'leaf':
            nodequeue.append(node.leftchild)
            nodequeue.append(node.rightchild)
        else:
            count += 1
    return count
            
def classifyobj(tree, obj, featurenames):
    if tree.nodetype == 'leaf':
        return tree.label
    elif isinstance(tree.cond, int) or isinstance(tree.cond, float):
        if obj[featurenames.index(tree.feature)] <= tree.cond:
            return classifyobj(tree.leftchild, obj, featurenames)
        else:
            return classifyobj(tree.rightchild, obj, featurenames)
    else:
        if obj[featurenames.index(tree.feature)] in tree.cond:
            return classifyobj(tree.leftchild, obj, featurenames)
        else:
            return classifyobj(tree.rightchild, obj, featurenames)
    

    
if __name__ == '__main__':
    
    tree = train()
    
    printtree(tree)