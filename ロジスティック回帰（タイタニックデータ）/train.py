from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import csv
import pickle

with open('x_train', 'rb') as f:
    x_train = pickle.load(f)

with open('y_train', 'rb') as f:
    y_train = pickle.load(f)

with open('x_test', 'rb') as f:
    x_test = pickle.load(f)

lr = LogisticRegression(C = 100, random_state = 1, solver = "lbfgs")
lr.fit(x_train, y_train)
pred_prob = lr.predict_proba(x_test)
pred_ans = []
for i in range(x_test.shape[0]):
    if pred_prob[i, 0] > 0.5:
        pred_ans.append(0)
    else:
        pred_ans.append(1)
print(pred_ans)
PassengeId = range(892, 1310)
Survived = pred_ans
gender_submission = pd.DataFrame(np.array([PassengeId, Survived]).T)
gender_submission.columns = ["PassengerID", "Survived"]


gender_submission.to_csv("gender_submission.csv", index= False)