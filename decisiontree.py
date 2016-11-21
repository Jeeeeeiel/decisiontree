#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 21:54:42 2016

@author: Jeiel
"""

class DecisionNode:
    def __init__(self, feature = None, data = None, impurity = None, cond = None, leftchild = None, rightchild = None, nodetype = None, label = None, method = None):

        self.cond = cond	#numeric,<= cond represent leftchild;nominal, in cond represent leftchild
        self.feature = feature
        self.leftchild = leftchild
        self.rightchild = rightchild
        self.nodetype = nodetype
        self.label = label
        self.data = data
        self.impurity = calcimpurity(data, method)
        

#calculate impurity 
def calcimpurity(data, method):
    from math import log2
    from collections import Counter
    
    if len(data) == 0:
        return 0
    
    total = len(data)
    categorydict = Counter([row[-1] for row in data])
    categorydict = {key:categorydict[key]/total for key in categorydict.keys()} #percentage dict
    impurity = 1
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
    from math import floor
    condimpuritydict = {}
    valuedict = Counter([row[col] for row in data])
    if isinstance(list(valuedict.keys())[0], int) or isinstance(list(valuedict.keys())[0], float):    #numeric
        keylist = sorted(valuedict.keys())
#        print('distinct value size(numeric): ', len(keylist))

        if len(keylist) == 1:
            condimpuritydict[keylist[0]] = calcimpurity(data, method)
            
        else: #len(keylist)>1
            for i in range(1, len(keylist)): #get every split value
                mean = (keylist[i-1] + keylist[i])/2   #i start form 1, i-1 avoid only one distinct attr
                (data1, data2) = dividedata(data, col, mean)
                condimpuritydict[mean] = (calcimpurity(data1, method)*len(data1) + calcimpurity(data2, method)*len(data2))/len(data)

            
    else:   #nominal
        valuesetlist = []
        if len(valuedict) == 1:#only one value in column
            valuesetlist += [(data[0][col],)]
        for size in range(1, floor(len(valuedict)/2)+1):#len(valuedict) > 1
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


def treegrowth(data, featureleft, method, preprune, threshold):
    if teststop(data):
        node = DecisionNode(data = data, nodetype = 'leaf', method = method)
        node.label = classify(data)
        return(node)
    else:
        node = DecisionNode(data = data, nodetype = 'node', method = method)
#        print('featuresize(exclude category): ', len(featureleft) - 1)
#        print('data size(exclude category):', len(data[0]) - 1)
        (featureindex, node.cond, ig) = findbestsplitattr(node.impurity, data, method)
        if preprune & (ig < threshold): #when preprune on
#            print('prepruning on, threshold: ', threshold, ', ig: ', ig, ', pruned!')
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
        
        
        node.leftchild = treegrowth(data1, featureleft, method, preprune, threshold)
        node.rightchild = treegrowth(data2, featureleft, method, preprune, threshold)
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
#    print('method:',method,'preprune:',preprune,'postprune:',postprune,'threshold:',threshold)
    tree = treegrowth(data, featurenames, method, preprune, threshold)

    if postprune:
        postpruning(tree)

#    clean(tree)
    return tree

def postpruning(node, penalty = 0.5):  #Pessimistic error
    if node.nodetype == 'leaf':
        return(sum([row[-1] != node.label for row in node.data]), 1)
    else:
        (lefterrorcount, leftleafcount) = postpruning(node.leftchild)
        (righterrorcount, rightleafcount) = postpruning(node.rightchild)
        childleafcount = leftleafcount + rightleafcount
        childerrorcount = lefterrorcount + righterrorcount
        pebeforeprune = (childerrorcount + (childleafcount) * penalty) / len(node.data)
        
        label = classify(node.data)
        errorcount = sum([row[-1] != label for row in node.data])
        peafterprune = (errorcount + penalty) / len(node.data)
        
        if peafterprune < pebeforeprune:
#            print(peafterprune,pebeforeprune,childleafcount,childerrorcount,errorcount)
            node.nodetype = 'leaf'
            node.label = label
            node.leftchild = None
            node.rightchild = None
            return(errorcount, 1)
        else:
            return(childleafcount, childerrorcount)


def printtree(tree): #queue
    from collections import deque
    nodequeue = deque()
    nodequeue.append(tree)
    
    while len(nodequeue) > 0:
        node = nodequeue.popleft()
        if node.nodetype != 'leaf':
            nodequeue.append(node.leftchild)
            nodequeue.append(node.rightchild)
#            print(node.feature,'(cond: ', node.cond,')', end = ' | ')
        else:
            print('label: ', node.label, end = ' | ')
    print()
 
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
    
def classifydata(tree, testdata, featurenames):
    result = []#[label,label,...]
    errorcount = 0
    for i in range(0, len(testdata)):
        result.append(classifyobj(tree, testdata[i], featurenames))
        if testdata[i][-1] != result[-1]:
            errorcount += 1
    return(result,errorcount)
    
def classifydataforclassifier(classifiers, alpha, testdata, featurenames):
    from collections import Counter
    tmpresults = []#[[label,label,...],...]  
    errorcount = 0
    results = []
    for classifier in classifiers:
        tmpresult = classifydata(classifier, testdata, featurenames)[0]
        tmpresults.append(tmpresult)
    for i in range(0, len(testdata)):#vote for every obj with weight
        resultdict = Counter([row[i] for row in tmpresults]) #possible results for no.i obj in testdata
        votes = {}
        for key in resultdict.keys():#assume the key is the correct category for no.i obj in testdata
            vote = [alpha[tmpresults.index(result)] for result in tmpresults if result[i] == key ]
            votes[key] = sum(vote)
        max_item = max(votes.items(), key = lambda item:item[1]) #find the item with biggest value
        results.append(max_item[0])
        if testdata[i][-1] != results[-1]:
            errorcount += 1
    return(results, errorcount)
    
            
def adaboost(data, featurenames, method, k = 0, preprune = False, postprune = False, threshold = 0.0):
    from math import log
    from math import exp
    import numpy as np
    weight = [1/len(data)]*len(data)
    alpha = []
    classifiers = []
    i = 0
    while i < k:
        mask = np.random.choice(len(data), len(data), replace = True, p = weight)
        #uniform:Return a random floating point number N such that a <= N <= b for a <= b and b <= N <= a for b < a.
        traindata = [data[j] for j in mask] 
        tree = train(traindata, featurenames, method, preprune = preprune, postprune = postprune, threshold = threshold)
        classifiers.append(tree)
        (result, errorcount) = classifydata(tree, data, featurenames)
        
        isequal = lambda x, y: (x == y)*1 #0,1
        epsilon = 0
        for j in range(0, len(data)):
            epsilon += weight[j]*isequal(data[j][-1], result[j])
        epsilon /= len(data)
        if epsilon > 0.5:
            weight = [1/len(data)]*len(data)
            continue
        alpha.append(1/2*log((1 - epsilon)/epsilon))
        
        getcoefficient = lambda x,y: ((x != y)-0.5)/0.5 #-1,1
        weight = [exp(getcoefficient(data[j][-1], result[j])*alpha[-1])*weight[j] for i in range(0, len(data))]
        Z = sum(weight)
        weight = [w/Z for w in weight]
        
        i += 1
    return(classifiers,alpha)
        
    
if __name__ == '__main__':
    
    tree = train()
    
    printtree(tree)

