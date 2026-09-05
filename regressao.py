import pickle

from sklearn.linear_model import LogisticRegression

with open('hr_turnover.pkl', 'rb') as f:
    dados = pickle.load(f)

x_train = dados["x_train"]
x_test = dados["x_test"]
y_train = dados["y_train"]
y_test = dados["y_test"]
preprocess = dados["preprocess"]

modelo = LogisticRegression(max_iter=100,class_weight="balanced")
modelo.fit(x_train, y_train)

previsoes = modelo.predict(x_test)