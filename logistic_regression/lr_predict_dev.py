from sklearn.metrics import f1_score as f1_score
from scipy.sparse import csr_matrix
import words_dic
import numpy as np
import scipy
from sklearn.metrics import f1_score as f1_score


beta = []
with open("beta_smaller.txt", 'r') as file:
    for item in file:
        beta.append(item)
beta = [float(string) for string in beta]

dev_id, dev_label, dev_matrix = words_dic.generate_matrix('dev.tsv', 'words_indices.txt')
dev_label = [int(string) for string in dev_label]

Y_true = [int(string) for string in dev_label]

intercept = csr_matrix(np.ones((dev_matrix.shape[0] ,1), dtype=int))
dev_model = scipy.sparse.hstack((dev_matrix, intercept), format='csr')

x = dev_model * beta
tmp = np.exp(-x)
tmp += 1
p = 1 / tmp

y_pred = []
for value in p:
    if value >= 0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)

f1 = f1_score(y_true = dev_label, y_pred = y_pred)
print("f1 score is ", f1)

# - generate report
with open("logistics_smaller_pred_result.csv", 'w') as f:
    n = len(dev_id)
    f.write("instance_id,class\n")
    for i in range(n):
        f.write(dev_id[i] + "," + str(y_pred[i]) + '\n')
