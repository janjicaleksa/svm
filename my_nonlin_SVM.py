import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from cvxopt import matrix, solvers

def gaussian_kernel(X_train, x, sigma):
    norm = np.linalg.norm(X_train-x, axis=1)**2
    return np.exp(-norm/(2*sigma**2))
def svm_dual(X_train, y_train, C, sigma):
    m = len(y_train)
    y_arr = np.array(y_train)
    X_arr = np.array(X_train)

    K = np.zeros((m, m))
    for i in range(m):
        x = X_arr[i, :]
        K[i, :] = gaussian_kernel(X_arr, x, sigma)

    P_arr = np.zeros([m, m])
    for i in range(m):
        for j in range(m):
            P_arr[i, j] = y_arr[i]*y_arr[j]*K[i, j]

    q_arr = -1 * np.ones(m)
    A_arr = y_arr.reshape(1, -1)
    h_arr = np.concatenate((np.zeros(m), C*np.ones(m)), axis=0)
    G_arr = np.zeros([2*m, m])
    for i in range(2*m):
        if i < m:
            G_arr[i, i] = -1
        else:
            G_arr[i, i-m] = 1

    P = matrix(P_arr)
    q = matrix(q_arr)
    G = matrix(G_arr)
    h = matrix(h_arr)
    A = matrix(A_arr*1.0)
    b = matrix(np.zeros(1))
    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol['x'])
    alphas = np.squeeze(alphas)

    return alphas, K
def calculate_b(alphas, y_train, K):
    y_arr = np.array(y_train)
    sup_vec_ind = np.argwhere((alphas > 1e-3) & (alphas <= C)).squeeze()
    b = 0
    for ind in sup_vec_ind:
        b += y_arr[ind]-(alphas[sup_vec_ind]*y_arr[sup_vec_ind]).reshape(1, -1)@K[sup_vec_ind, ind].reshape(-1, 1)
    b = b/len(sup_vec_ind)
    return b
def predict(X_test, X_train, y_train, sigma, alphas, b):
    X_arr = np.array(X_train)
    y_arr = np.array(y_train)
    X_test = np.array(X_test)
    y_pred = np.zeros(len(X_test))
    for i in range(len(X_test)):
        kernel = gaussian_kernel(X_arr, X_test[i], sigma)
        sum = (alphas*y_arr).reshape(1, -1) @ kernel
        y_pred[i] = np.sign(sum+b)
    return y_pred
def crossvalidation(data, C, sigma, k):
    X = data[['x1', 'x2']]
    y = data['y']
    m = len(y)
    acc = np.zeros(k)
    for i in range(k):
        X_test = X[i*m//k:(i+1)*m//k]
        y_test = y[i*m//k:(i+1)*m//k]
        X_train = np.concatenate((X[0:i*m//k], X[(i+1)*m//k:]), axis=0)
        y_train = np.concatenate((y[0:i*m//k], y[(i+1)*m//k:]), axis=0)
        alphas, K = svm_dual(X_train, y_train, C, sigma)
        b = calculate_b(alphas, y_train, K)
        y_pred = predict(X_test, X_train, y_train, sigma, alphas, b)
        acc[i] = sum(y_pred == y_test)/len(y_test)
    return acc

data = pd.read_csv('svmData.csv', header=None, names=['x1', 'x2', 'y'])

C_list = [50, 100, 200, 500]
sigma_list = [0.1, 0.5, 1, 2, 5]
acc_mean = []
acc_std = []
for C in C_list:
    for sigma in sigma_list:
        acc = crossvalidation(data, C, sigma, 5)
        acc_mean.append(np.mean(acc))
        acc_std.append(np.std(acc))
        #print(f"For C={C} and sigma={sigma} mean accuracy[%] is:{np.mean(acc)} with standard deviation: {np.std(acc)}")

# Data splitting, calculating optimal SVM parameters and separation line
X_train, X_test, y_train, y_test = train_test_split(data[['x1', 'x2']], data['y'], test_size=0.2, random_state=42, stratify=data['y'])
alphas, K = svm_dual(X_train, y_train, 200, 1)
b = calculate_b(alphas, y_train, K)
y_pred = predict(X_test, X_train, y_train, sigma, alphas, b)

# Calculating support vectors
sup_vec_ind = np.argwhere((alphas > 1e-3) & (alphas <= 200)).squeeze()
sup_vec = np.array([np.array(X_train)[ind] for ind in sup_vec_ind])

plt.scatter(sup_vec[:, 0], sup_vec[:, 1], s=100, marker='p', facecolors='none', edgecolor='g', linewidth=2, label='support vectors')
plt.scatter(X_train['x1'], X_train['x2'], c=np.array(y_train), cmap=plt.cm.jet)
xmin, xmax, ymin, ymax = plt.axis()
xx, yy = np.meshgrid(np.linspace(xmin, xmax, num=100, endpoint=True), np.linspace(ymin, ymax, num=100, endpoint=True))
y_pred = predict(np.c_[xx.ravel(), yy.ravel()], X_train, y_train, 1, alphas, b)
y_pred = y_pred.reshape(xx.shape)
cs = plt.contourf(xx, yy, y_pred, alpha=0.2, cmap='bwr')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Dual SVM - Training set')
plt.legend()
plt.show()

y_pred = predict(X_test, X_train, y_train, 1, alphas, b)
print(f"Accuracy test set[%]: {sum(y_pred == y_test)/len(y_test)*100}")