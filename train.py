import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("data/breast-cancer.csv").replace("?", np.nan)
X = df.drop(columns="Class")
y = df["Class"]

ORDINAIS = {
    "age": ["10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-89", "90-99"],
    "tumor-size": ["0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34",
                   "35-39", "40-44", "45-49", "50-54", "55-59"],
    "inv-nodes": ["0-2", "3-5", "6-8", "9-11", "12-14", "15-17", "18-20", "21-23",
                  "24-26", "27-29", "30-32", "33-35", "36-39"],
}
COL_ORD = list(ORDINAIS)
COL_NOM = ["menopause", "node-caps", "breast", "breast-quad", "irradiat"]
COL_NUM = ["deg-malig"]

pre = ColumnTransformer([
    ("ord", Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("enc", OrdinalEncoder(categories=list(ORDINAIS.values()))),
        ("sc", StandardScaler()),
    ]), COL_ORD),
    ("nom", Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("enc", OneHotEncoder(handle_unknown="ignore")),
    ]), COL_NOM),
    ("num", StandardScaler(), COL_NUM),
])

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

modelos = {
    "Regressão Logística": (LogisticRegression(max_iter=1000),
                            {"clf__C": [0.01, 0.1, 1, 10]}),
    "KNN": (KNeighborsClassifier(),
            {"clf__n_neighbors": [3, 5, 7, 9, 11]}),
    "Árvore de Decisão": (DecisionTreeClassifier(random_state=42),
                          {"clf__max_depth": [2, 3, 4, 5, None]}),
    "Random Forest": (RandomForestClassifier(random_state=42),
                      {"clf__n_estimators": [100, 300], "clf__max_depth": [3, 5, None]}),
    "SVM": (SVC(),
            {"clf__C": [0.1, 1, 10], "clf__kernel": ["linear", "rbf"]}),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
resultados = []
melhores = {}

for nome, (clf, grade) in modelos.items():
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    busca = GridSearchCV(pipe, grade, cv=cv, scoring="accuracy").fit(X_tr, y_tr)
    acc_teste = accuracy_score(y_te, busca.predict(X_te))
    resultados.append({
        "modelo": nome,
        "acuracia_validacao_cruzada": round(busca.best_score_, 4),
        "acuracia_teste": round(acc_teste, 4),
        "melhores_parametros": busca.best_params_,
    })
    melhores[nome] = busca.best_estimator_

tabela = pd.DataFrame(resultados).sort_values("acuracia_validacao_cruzada", ascending=False)
tabela.to_csv("models/resultados.csv", index=False)
print(tabela.to_string(index=False))

vencedor = tabela.iloc[0]["modelo"]
modelo_final = melhores[vencedor]
pred = modelo_final.predict(X_te)

print("\nMelhor modelo:", vencedor)
print("Acurácia no teste:", round(accuracy_score(y_te, pred), 4))
print(confusion_matrix(y_te, pred))
print(classification_report(y_te, pred))

joblib.dump(modelo_final, "models/melhor_modelo.joblib")
print("Modelo salvo em models/melhor_modelo.joblib")