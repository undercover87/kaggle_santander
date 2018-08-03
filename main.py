import pickle

from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import numpy as np
import sklearn.metrics as skm
from sklearn.cross_validation import cross_val_score
import os
# A function to calculate Root Mean Squared Logarithmic Error (RMSLE)
def rmsle(y, y_pred):
    assert len(y) == len(y_pred)
    terms_to_sum = [(np.log(y_pred[i] + 1) - np.log(y[i] + 1)) ** 2.0 for i, pred in enumerate(y_pred)]
    return (sum(terms_to_sum) * (1.0 / len(y))) ** 0.5

if not os.path.exists("data.dat"):
    data = pd.read_csv("D:\userdata\\bellas\\Downloads\\train.csv", header=0)
    data.set_index(["ID"], inplace=True)
    data.to_pickle("data.dat")
else:
    data = pd.read_pickle("data.dat")

if not os.path.exists("model.dat"):
    X = data.drop(["target"], axis=1)
    targets = data.target
    gb = GradientBoostingRegressor()
    gb.fit(X, targets)
    with open("model.dat", "wb") as f:
        pickle.dump(gb,f)
else:
    with open("model.dat","rb") as f:
        gb = pickle.load(f)

if not os.path.exists("test_data.dat"):
    test_data = pd.read_csv("D:\userdata\\bellas\\Downloads\\test.csv", header=0)
    test_data.set_index(["ID"], inplace=True)
    test_data.to_pickle("test_data.dat")
else:
    test_data = pd.read_pickle("test_data.dat")

predictions = pd.DataFrame([test_data.index, gb.predict(test_data)], index=["ID", "target"]).transpose()
predictions.to_csv("predictions.csv", header=True, index=False)
#scores = cross_val_score(X=X,y=targets, estimator=gb, scoring="neg_mean_squared_log_error")#scoring=skm.make_scorer(rmsle))
#print np.mean(scores)

