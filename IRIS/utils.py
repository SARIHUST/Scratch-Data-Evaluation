import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

seed = 423
np.random.seed(seed)

def shapley_values(X_train, y_train, X_test, y_test, epsilon=1e-8, evaluate='loss', max_p=3):
    '''
        The function is implemented based on the TMC-SV algorithm
    '''
    X_0s, X_1s = X_train[y_train==0], X_train[y_train==1]
    X_mean_0, X_mean_1 = np.mean(X_0s, axis=0), np.mean(X_1s, axis=0)
    n = len(X_train)
    phais = np.zeros(n)
    t = 0
    LR = LogisticRegression()
    LR.fit(X_train, y_train)
    y_predict = LR.predict(X_test)
    orig_X = np.array((X_mean_0, X_mean_1)) 
    # use the mean of the two types as the original model
    orig_y = np.array((0, 1))
    if evaluate == 'ac':
        total_score = accuracy_score(y_test, y_predict)
        orig_score = accuracy_score(y_test, np.zeros(len(y_test)))
    elif evaluate == 'loss':
        total_score = -log_loss(y_test, y_predict)
        orig_score = -log_loss(y_test, np.zeros(len(y_test)))

    record_1 = [0]
       
    while t < max_p * n:
        old_phais = phais.copy()
        t += 1
        vs = np.zeros(n + 1)
        vs[0] = orig_score
        # with out training the classifier assigns every data point to label 0
        pai_t = np.random.permutation(np.arange(0, n, step=1))
        for j in range(1, n + 1):
            idx = pai_t[j - 1]
            if total_score - vs[j - 1] <= epsilon:
                vs[j] = vs[j - 1]
            else:
                X_, y_ = X_train[pai_t[:j]], y_train[pai_t[:j]]
                LR.fit(np.vstack((orig_X, X_)), np.hstack((orig_y, y_)))
                y_predict = LR.predict(X_test)
                if evaluate == 'ac':
                    vs[j] = accuracy_score(y_test, y_predict)
                elif evaluate == 'loss':
                    vs[j] = -log_loss(y_test, y_predict)
            phais[idx] = phais[idx] * (t - 1) / t + (vs[j] - vs[j - 1]) / t

        record_1.append(phais[1])
        if t > n and sum(abs(old_phais - phais) < 1e-3) == n:
            break

    print(sum(abs(old_phais - phais) < 1e-3))
    return phais, np.array(record_1)