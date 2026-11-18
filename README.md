# Classificação Breast Cancer

Modelo de classificação do atributo Class (recurrence-events / no-recurrence-events) com a base Breast Cancer (UCI).

## Como executar

```
pip install -r requirements.txt
python train.py
python predict.py
```

## Treinamento (`train.py`)
- Valores `?` (node-caps e breast-quad) tratados como faltantes e imputados pela moda.
- age, tumor-size e inv-nodes: codificação ordinal (respeitando a ordem das faixas) e padronização.
- menopause, node-caps, breast, breast-quad e irradiat: One-Hot Encoding.
- Divisão treino/teste 80/20 estratificada.
- Modelos comparados com GridSearchCV (validação cruzada de 5 partições): Regressão Logística, KNN, Árvore de Decisão, Random Forest e SVM.
- O modelo com maior acurácia na validação cruzada (Árvore de Decisão) é salvo em `models/melhor_modelo.joblib`. A comparação fica em `models/resultados.csv`.

## Inferência (`predict.py`)
Recebe os dados de uma nova paciente e retorna o indicativo de diagnóstico (Class) com a probabilidade de cada classe.

## Observação
A base é pequena (286 registros) e desbalanceada (~70% sem recorrência), então a acurácia fica próxima do baseline.