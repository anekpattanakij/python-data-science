# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data

from sklearn.preprocessing import OneHotEncoder 
from sklearn.compose import ColumnTransformer, make_column_transformer
preprocessor = make_column_transformer( (OneHotEncoder(),[3]),remainder="passthrough") # 3 is column of categorial variable
X = preprocessor.fit_transform(X) # transform will move to first indext

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# this accacy score for regression method only
print(regressor.score(X_test,y_test))

# add [1] to first column for preparation backward regression
import statsmodels.api as sm
X = np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)
X_opt = X[:,[0,1,2,3,4,5]]
X_opt = X_opt.astype(float)
regressor_OLS = sm.OLS(endog=y , exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,1,2,3,4]]
X_opt = X_opt.astype(float)
regressor_OLS = sm.OLS(endog=y , exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,1,3,4]]
X_opt = X_opt.astype(float)
regressor_OLS = sm.OLS(endog=y , exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,3,4]]
X_opt = X_opt.astype(float)
regressor_OLS = sm.OLS(endog=y , exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,3]]
X_opt = X_opt.astype(float)
regressor_OLS = sm.OLS(endog=y , exog = X_opt).fit()
regressor_OLS.summary()

# Backward Elimination with p-values only
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x

SL = 0.05

X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_opt = X_opt.astype(float)
X_Modeled = backwardElimination(X_opt, SL)


def backwardElimination(x, SL):
    numVars = len(x[0])
    temp = np.zeros((50,6)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_opt = X_opt.astype(float)
X_Modeled = backwardElimination(X_opt, SL)

