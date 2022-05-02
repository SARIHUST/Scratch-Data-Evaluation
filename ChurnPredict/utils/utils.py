import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def shapley_values(X_train, y_train, X_test, y_test, epsilon=1e-8, evaluate='ac'):
    '''
        The function is implemented based on the TMC-SV algorithm
    '''
    np.random.seed(110810423)
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
       
    while t < 3 * n:
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
                X_, y_ = X_train[:j], y_train[:j]
                LR.fit(np.vstack((orig_X, X_)), np.hstack((orig_y, y_)))
                y_predict = LR.predict(X_test)
                if evaluate == 'ac':
                    vs[j] = accuracy_score(y_test, y_predict)
                elif evaluate == 'loss':
                    vs[j] = -log_loss(y_test, y_predict)
            phais[idx] = phais[idx] * (t - 1) / t + (vs[j] - vs[j - 1]) / t
    return phais

def load_churn_data():
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=110810)
    # scale the continuous features
    scaler = StandardScaler()
    scaler.fit(X_train[['tenure', 'MonthlyCharges', 'TotalCharges']])
    X_train[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(X_train[['tenure', 'MonthlyCharges', 'TotalCharges']])
    X_test[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(X_test[['tenure', 'MonthlyCharges', 'TotalCharges']])

    return X_train, X_test, y_train, y_test