import pickle
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# Para exibição sem truncar
pd.set_option('display.expand_frame_repr', False)



# 1) Carregar dados
database = pd.read_excel('HR_Abandono.xlsx', )

print(database.head())
print(database.dtypes)
print(database.isnull().sum())

print("\nEstatísticas antes da remoção de outliers:")
print(database.describe(include='all'))

# 2) Remover outliers
z = np.abs(stats.zscore(database['average_montly_hours']))
outlier_idx = database.index[z > 3]
database = (database.drop(index=outlier_idx))

print("\nEstatísticas após remoção de outliers:")
print(database.describe(include='all'))

# 3) Separar previsores e alvo
X = database.drop(columns=['id', 'left'])
y = database['left']

# 4) Identificar colunas
cat_cols = list(X.select_dtypes(include=['object', 'category']).columns)
num_cols = list(X.select_dtypes(include='number').columns)

# 5) Pré-processamento
preprocess = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
])

X = preprocess.fit_transform(X)

# 6) Separar treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=0,
    stratify=y
)

# 7) Salvar
with open('hr_turnover.pkl', 'wb') as f:
    pickle.dump({
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'preprocess': preprocess
    }, f)

print("Arquivo hr_turnover.pkl criado.")