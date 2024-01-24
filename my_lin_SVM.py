import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from cvxopt import matrix, solvers

def svm_primal(X_train, y_train, C):
    m = len(y_train)
    P_arr = np.zeros([m+3, m+3])
    P_arr[1, 1] = 1
    P_arr[2, 2] = 1
    q_arr = np.concatenate((np.zeros(3), C*np.ones(m)), axis=0)
    G_arr = np.zeros([2*m, m+3])
    X_arr = np.array(X_train)
    y_arr = np.array(y_train)
    h_arr = np.concatenate((-1 * np.ones((m, 1)), np.zeros((m, 1))), axis=0) ##mozda treba da se pomeri

    for i in range(2*m):
        if i < m:
            x1, x2 = X_arr[i]
            y = y_arr[i]
            first_part = np.array([y, y*x1, y*x2])
            second_part = np.zeros(m)
            second_part[i] = 1
            G_arr[i, :] = -1 * np.concatenate((first_part, second_part), axis=0)
        else:
            G_arr[i, 3+i-m] = -1

    P = matrix(P_arr)
    q = matrix(q_arr)
    G = matrix(G_arr)
    h = matrix(h_arr)

    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h)
    param = np.array(sol['x'])
    param = np.squeeze(param)

    return param
def predict(X_test, y_test, param):
    b = param[0]
    w1 = param[1]
    w2 = param[2]
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    y_pred = np.ones(len(y_test))
    for i in range(len(y_test)):
        x1 = X_test[i][0]
        x2 = X_test[i][1]
        if x1*w1 + x2*w2 + b < 0:
            y_pred[i] = -1
    return y_pred
def crossvalidation(data, C, k):
    X = data[['x1', 'x2']]
    y = data['y']
    m = len(y)
    acc = np.zeros(k)
    for i in range(k):
        X_test = X[i*m//k:(i+1)*m//k]
        y_test = y[i*m//k:(i+1)*m//k]
        X_train = np.concatenate((X[0:i*m//k], X[(i+1)*m//k:]), axis=0)
        y_train = np.concatenate((y[0:i*m//k], y[(i+1)*m//k:]), axis=0)
        params = svm_primal(X_train, y_train, C)
        y_pred = predict(X_test, y_test, params)
        acc[i] = sum(y_pred == y_test)/len(y_test)
    return acc

data = pd.read_csv('svmData.csv', header=None, names=['x1', 'x2', 'y'])

C_list = [0.1, 0.5, 1, 2, 3, 4, 5, 10]
acc_mean = []
acc_std = []
for C in C_list:
    acc = crossvalidation(data, C, 5)
    acc_mean.append(np.mean(acc))
    acc_std.append(np.std(acc))
max_pos = acc_mean.index(max(acc_mean))
C_opt = C_list[max_pos]

plt.figure(figsize=(12, 4))
plt.title('Validation curve')
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.axvline(x=C_opt, color='red', label='Optimal C= ' + str(C_opt))
plt.scatter(C_list, acc_mean)
plt.plot(C_list, acc_mean, label='validation')
plt.fill_between(C_list, np.array(acc_mean)-np.array(acc_std), np.array(acc_mean)+np.array(acc_std), alpha=0.2, color='b')
plt.legend()
plt.show()

# Data splitting, calculating optimal SVM parameters and separation line
X_train, X_test, y_train, y_test = train_test_split(data[['x1', 'x2']], data['y'], test_size=0.2, random_state=42, stratify=data['y'])
param = svm_primal(X_train, y_train, 2)
b = param[0]
w1 = param[1]
w2 = param[2]
x = np.arange(min(X_train['x1']), max(X_train['x1']), 0.01)
y = -b/w2 - w1*x/w2

# Calculating support vectors
supp_vec_indices = []
ksi = []
for i in range(len(X_train)):
    x1, x2 = np.array(X_train)[i]
    fun_mar = np.array(y_train)[i]*(w1*x1 + w2*x2 + b)
    if fun_mar <= 1+1e-3:
        supp_vec_indices.append(i)
        ksi.append(round(fun_mar-1, 1))
supp_vec = np.array([np.array(X_train)[ind] for ind in supp_vec_indices])

plt.figure(figsize=(12, 6))
plt.scatter(supp_vec[:, 0], supp_vec[:, 1], s=100, marker='p', facecolors='none', edgecolor='green', linewidth=2, label='support vectors')
for i, txt in enumerate(ksi):
    plt.annotate(txt, (supp_vec[i,0], supp_vec[i,1]))
plt.plot(x, y, '--', c='g', label='separation line')
plt.scatter(X_train[y_train == 1]['x1'], X_train[y_train == 1]['x2'], marker='*', c='b')
plt.scatter(X_train[y_train == -1]['x1'], X_train[y_train == -1]['x2'], marker='o', c='r')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Primal SVM - Training set')
plt.legend()
plt.show()

y_pred = predict(X_test, y_test, param)
print(f"Accuracy test set[%]: {sum(y_pred == y_test)/len(y_test)*100}")