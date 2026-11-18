import joblib
import pandas as pd

modelo = joblib.load("models/melhor_modelo.joblib")

TRADUCAO = {
    "no-recurrence-events": "sem recorrência (no-recurrence-events)",
    "recurrence-events": "com recorrência (recurrence-events)",
}


def inferir(paciente: dict):
    df = pd.DataFrame([paciente])
    classe = modelo.predict(df)[0]
    probs = dict(zip(modelo.classes_, modelo.predict_proba(df)[0]))
    return classe, probs


if __name__ == "__main__":
    nova = {
        "age": "50-59",
        "menopause": "premeno",
        "tumor-size": "20-24",
        "inv-nodes": "3-5",
        "node-caps": "yes",
        "deg-malig": 3,
        "breast": "left",
        "breast-quad": "left_low",
        "irradiat": "yes",
    }
    classe, probs = inferir(nova)
    print("Indicativo de diagnóstico (Class):", TRADUCAO[classe])
    for c, p in probs.items():
        print(f"  {c}: {p:.1%}")