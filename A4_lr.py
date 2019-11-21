from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

boston = datasets.load_boston()
boston.data.shape   # (506,13): sample:506, dimension:13

X = boston.data
y = boston.target

lr = LinearRegression()
lr.fit(X,y)

lr.intercept_
lr.coef_

coef = pd.DataFrame(lr.coef_)
name = boston.feature_names.tolist()
coef.index = name
coef

print(np.where(abs(coef)==np.max(abs(coef))), np.max(abs(coef)))
print(boston.DESCR)
# the 5th element: NOX (nitric oxides concentration(parts per 10 million)) is the "most" influence. 
