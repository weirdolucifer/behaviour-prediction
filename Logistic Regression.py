import pandas as pd
from matplotlib import pyplot as plt


df = pd.read_csv('final.csv')
X = df.iloc[:, [0, 13]].values
y = df.iloc[:, 14].values

#plt.scatter(df.Conductance,df.Resistance,df.ConductanceVoltage)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


from sklearn.linear_model import LogisticRegression
model = LogisticRegression(C=10.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=10, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)
model.fit(X_train, y_train)



y_predicted = model.predict(X_test)

model.predict_proba(X_test)


print("*********************************************")
print("Accuracy :")
print(model.score(X_test,y_test)*100)
print("%")
print("*********************************************")



