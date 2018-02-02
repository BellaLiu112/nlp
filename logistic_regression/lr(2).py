import numpy as np
from scipy.sparse import csr_matrix
import scipy
import words_dic
import pdb

def sigmoid(Matrix, beta):
    x = Matrix * beta
    tmp = np.exp(-x)
    tmp += 1
    prob_array = 1 / tmp
    return prob_array

def loglikelihood(Xmatrix, Y, beta):
    tmp = Xmatrix * beta
    ll = sum(Y * tmp) - sum(np.log(1 + np.exp(tmp)))
    return ll

def computegradient(Xmatrix, Y_true, Y_pred):
    delta_ll = Xmatrix.T * (np.array(Y_true) - np.array(Y_pred))
    return delta_ll

def Separator(prob_array, threshold = 0.5):
    y_pred = []
    for num in prob_array:
        if num >= threshold:
            y_pred.append(1)
        else:
            y_pred.append(0)
    return y_pred

def logisticregression(Xmatrix, Y_true, learning_rate = 5e-3, numstep = 300000):

    n, f = Xmatrix.shape[0], Xmatrix.shape[1]
    intercept = csr_matrix(np.ones((n ,1), dtype=int))
    # - X has dimensions [n + 1, f]
    X = scipy.sparse.hstack((Xmatrix, intercept), format='csr')
    # - init beta [0, 0, ..., 0]
    oldBeta = np.zeros(f + 1)
    # - init y_pred
    parray = sigmoid(X, oldBeta)
    y_pred = Separator(parray)
    newBeta = oldBeta + learning_rate * computegradient(X, Y_true, y_pred)
    delta = 0.00001
    cnt = 0
    ll_array = [loglikelihood(X, Y_true, oldBeta)]
    print(ll_array)
    diff = sum(np.absolute(oldBeta - newBeta))
    while ( diff > delta and cnt < numstep):
        parray = sigmoid(X, newBeta)
        y_pred = Separator(parray)
        oldBeta = newBeta
        newBeta = oldBeta + learning_rate * computegradient(X, Y_true, y_pred)
        cnt += 1
        diff = sum(np.absolute(oldBeta - newBeta))
        if cnt % 1000 == 0:
            print("%d times, delta is %f" %(cnt, diff))
            ll = loglikelihood(X, Y_true, newBeta)
            ll_array.append(ll)
    return newBeta, ll_array

def predict(x_array, beta, threshold = 0.5):
    x = sum(x_array * beta)
    prob = 1/(1 + np.exp(-x))
    if prob >=  threshold:
        return 1
    else:
        return 0

if __name__ == '__main__':

    train_id, train_label, matrix = words_dic.generate_matrix('train.tsv', 'words_indices.txt', True)

    Y_true = [int(string) for string in train_label]

    beta, ll_array = logisticregression(matrix, Y_true)

    with open("beta_larger.txt", 'w') as file:
        for par in beta:
            file.write(str(par) + '\n')

    with open("ll_array_larger.txt", 'w') as file:
        for ll in ll_array:
            file.write(str(ll) + '\n')
