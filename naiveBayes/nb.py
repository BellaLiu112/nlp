import preprocess
from collections import Counter
import numpy as np
from sklearn.metrics import f1_score as f1_score
import pdb

def train(tokenized_text, label, smoothing_alpha = 0):
    x0 = [] # - large bag of class0 tokens
    x1 = [] # - large bag of class1 tokens

    # - calculate P(Y)
    p_y = {0:0, 1:0} # P(Y)
    for item in label:
        p_y[item] += 1
    for item in p_y:
        p_y[item] /= len(label)

    # - calculate P(X)
    p_x = {} # P(X)
    for row in tokenized_text:
        for item in row:
            if item in p_x:
                p_x[item] += 1
            else:
                p_x[item] = 1
    tot = sum(p_x.values())
    for item in p_x:
        p_x[item] /= tot

    # - concatenate to a large bag of words for each class
    n = len(label)
    for i in range(n):
        if label[i] == 0:
            x0.extend(tokenized_text[i])
        else:
            x1.extend(tokenized_text[i])
    x0 = dict(Counter(x0))
    x1 = dict(Counter(x1))

    # - calculate P(Xi | Yi)
    p_x_y = {}
    # - example {"the":[0.1, 0.1]} for class 0 and class 1
    smoothingSum = len(p_x) * smoothing_alpha
    demoninatorX0 = len(x0) + smoothingSum
    demoninatorX1 = len(x1) + smoothingSum

    for item in p_x:
        p_x_y[item] = []
        if item in x0:
            temp0 = (x0[item] + smoothing_alpha) / demoninatorX0
        else:
            temp0 = smoothing_alpha / demoninatorX0
        p_x_y[item].append(temp0)
        if item in x1:
            temp1 = (x1[item] + smoothing_alpha) / demoninatorX1
        else:
            temp1 = smoothing_alpha / demoninatorX1
        p_x_y[item].append(temp1)

    return p_x, p_y, p_x_y

def classify(doc, p_y, p_x_y):
    # - doc is a tokenized document: a list of words
    p_c_0 = np.log(p_y[0])
    p_c_1 = np.log(p_y[1])
    for token in doc:
        if token in p_x_y:
            if p_x_y[token][0] == 0:
                return 1
            if p_x_y[token][1] == 0:
                return 0
            p_c_0 += np.log(p_x_y[token][0])
            p_c_1 += np.log(p_x_y[token][1])
    if p_c_0 > p_c_1:
        c = 0
    else:
        c = 1
    return c
