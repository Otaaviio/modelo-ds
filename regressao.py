import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

with open('hr_turnover.pkl', 'rb') as f:
    dados = pickle.load(f)

X_train = dados['X_train']
X_test = dados['X_test']
y_train = dados['y_train']
y_test = dados['y_test']
preprocess = dados['preprocess']

# 2) Criar e treinar o modelo
modelo = LogisticRegression(max_iter=100,  class_weight='balanced') #class_weight='balanced'
modelo.fit(X_train, y_train)

previsoes = modelo.predict(X_test)
acuracia = accuracy_score(y_test, previsoes)
print(f"Acurácia: {acuracia * 100:.2f}%")

# 5) Novo funcionário
novo_funcionario = pd.DataFrame([{
    'satisfaction_level': 0,
    'last_evaluation': 0.5,
    'num_project': 2,
    'average_montly_hours': 157,
    'time_spend_company': 5,
    'Work_accident': 2,
    'promotion_last_5years': 0,
    'depto': 'hr',
    'salary': 'high'
}])

novo_X = preprocess.transform(novo_funcionario)

# 7) Fazer a previsão
previsao = modelo.predict(novo_X)
probabilidade = modelo.predict_proba(novo_X)

print("\nPrevisão:", previsao[0])

print(f"Probabilidade de permanecer: {probabilidade[0][0] * 100:.2f}%")
print(f"Probabilidade de sair: {probabilidade[0][1] * 100:.2f}%")

# 8) Interpretar
if previsao[0] == 1:
    print("Resultado: funcionário com tendência de saída.")
else:
    print("Resultado: funcionário com tendência de permanência.")