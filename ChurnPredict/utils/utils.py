import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

seed = 423
np.random.seed(seed)

def shapley_values(X_train, y_train, X_test, y_test, evaluate='loss', max_p=3):
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
    epsilon_X = X_train[:(int)(0.85 * n)]
    epsilon_y = y_train[:(int)(0.85 * n)]
    LR.fit(epsilon_X, epsilon_y)
    epsilon_predict = LR.predict(X_test)
    orig_X = np.array((X_mean_0, X_mean_1)) 
    # use the mean of the two types as the original model
    orig_y = np.array((0, 1))
    if evaluate == 'ac':
        total_score = accuracy_score(y_test, y_predict)
        orig_score = accuracy_score(y_test, np.zeros(len(y_test)))
        epsilon_score = accuracy_score(y_test, epsilon_predict)
    elif evaluate == 'loss':
        total_score = -log_loss(y_test, y_predict)
        orig_score = -log_loss(y_test, np.zeros(len(y_test)))
        epsilon_score = -log_loss(y_test, epsilon_predict)
    
    epsilon = abs(total_score - epsilon_score) / 8
    print('epsilon: {}'.format(epsilon))

    records = [[0] for _ in range(n)]
    start = time.time() 

    converge = 0

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

        for i in range(n):
            records[i].append(phais[i])
        if sum(abs(old_phais - phais) < epsilon / 60) - sum(phais == 0) == n:
            converge += 1
            if converge == 30:
                break
       
        end = time.time()
        print('iteration {}, converge {}({} remains 0), time cost {}'.format(t, sum(abs(old_phais - phais) < epsilon / 60), sum(phais == 0), end - start))

    print(sum(abs(old_phais - phais) < epsilon / 60))
    return phais, np.array(records)

def load_churn_data(refine=0):
    telecom = pd.read_csv('ChurnPredict\WA_Fn-UseC_-Telco-Customer-Churn.csv')   
    print(telecom.info())

    # deal with binary features
    bin_features = ['PhoneService', 'PaperlessBilling', 'Churn', 'Partner', 'Dependents']
    telecom[bin_features] = telecom[bin_features].apply(lambda x: x.map({'Yes': 1, 'No': 0}))

    telecom['gender'] = telecom['gender'].map({'Male': 1, 'Female': 0})

    # deal with multinominal features by using dummy features

    dummy = pd.get_dummies(telecom[['Contract', 'PaymentMethod', 'InternetService']], drop_first=True)
    telecom = pd.concat([telecom, dummy], axis=1)

    # deal with the rest by hand by droping all the No phone/internet service
    # MultipleLines
    ml = pd.get_dummies(telecom['MultipleLines'], prefix='MultipleLines')
    ml = ml.drop('MultipleLines_No phone service', axis=1)
    telecom = pd.concat([telecom, ml], axis=1)

    # OnlineSecurity
    os = pd.get_dummies(telecom['OnlineSecurity'], prefix='OnlineSecurity')
    os = os.drop('OnlineSecurity_No internet service', axis=1)
    telecom = pd.concat([telecom, os], axis=1)

    # OnlineBackup
    ob = pd.get_dummies(telecom['OnlineBackup'], prefix='OnlineBackup')
    ob = ob.drop('OnlineBackup_No internet service', axis=1)
    telecom = pd.concat([telecom, ob], axis=1)

    # DeviceProtection
    dp = pd.get_dummies(telecom['DeviceProtection'], prefix='DeviceProtection')
    dp = dp.drop(['DeviceProtection_No internet service'], axis=1)
    telecom = pd.concat([telecom, dp], axis=1)

    # TechSupport
    ts = pd.get_dummies(telecom['TechSupport'], prefix='TechSupport')
    ts = ts.drop(['TechSupport_No internet service'], axis=1)
    telecom = pd.concat([telecom, ts], axis=1)

    # StreamingTV
    st =pd.get_dummies(telecom['StreamingTV'], prefix='StreamingTV')
    st = st.drop(['StreamingTV_No internet service'], axis=1)
    # Adding the results to the telecom dataframe
    telecom = pd.concat([telecom, st], axis=1)


    # MultipleLines
    sm = pd.get_dummies(telecom['StreamingMovies'], prefix='StreamingMovies')
    sm = sm.drop(['StreamingMovies_No internet service'], axis=1)
    telecom = pd.concat([telecom, sm], axis=1)

    print(telecom.head())

    # drop the repeated features after creating dummies
    telecom = telecom.drop(['Contract', 'PaymentMethod', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'], axis=1)

    telecom['TotalCharges'] = pd.to_numeric(telecom["TotalCharges"].replace(" ",""),downcast="float")

    print(telecom.info())

    # check for null features data

    print(telecom.isnull().sum())
    # there are 11 data points with null feature TotalCharges
    telecom = telecom[~np.isnan(telecom['TotalCharges'])]

    X = telecom.drop(['Churn', 'customerID'], axis=1)
    y = telecom['Churn']
    if refine != 0:
        # under refine, set No(data labeled 0) = [1 or 2] * No(data labeled 1)
        # with out refine there are 3624 data points labeled 0, 1298 labeled 1 in 
        # the X_train and 1539 data points labeled 1, 571 labeled 1 in X_test
        if refine > 2:
            refine = 2
        label_num = sum(y == 1) * refine
        X_1 = X[y == 1]
        y_1 = y[y == 1]
        label_0_idx = np.arange(len(X))[y == 0]
        shuffle_idx = np.random.permutation(label_0_idx)[:label_num]
        X_0 = X.iloc[shuffle_idx, :]
        y_0 = y.iloc[shuffle_idx]
        X = pd.concat((X_0, X_1))
        y = pd.concat((y_1, y_0))
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=110810)
    # scale the continuous features
    scaler = StandardScaler()
    scaler.fit(X_train[['tenure', 'MonthlyCharges', 'TotalCharges']])
    X_train[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(X_train[['tenure', 'MonthlyCharges', 'TotalCharges']])
    X_test[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(X_test[['tenure', 'MonthlyCharges', 'TotalCharges']])
     
    return X_train.values, X_test.values, y_train.values, y_test.values