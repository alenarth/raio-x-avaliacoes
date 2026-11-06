import pandas as pd

from reviews.evaluation import parse_labels, score


def test_parse_labels():
    assert parse_labels("comida:positivo|entrega:negativo") == {("comida", "positivo"), ("entrega", "negativo")}
    assert parse_labels("") == set()


def test_score_counts_aspects_and_sentiments():
    truth = pd.DataFrame({
        "mencoes_reais": ["comida:positivo|entrega:negativo", "pedido:negativo"],
        "sentimento_real": ["neutro", "negativo"],
        "urgente_real": [False, True],
    })
    result = pd.DataFrame({
        "mencoes": [[{"aspect": "comida", "sentiment": "positivo"}, {"aspect": "entrega", "sentiment": "positivo"}],
                    [{"aspect": "pedido", "sentiment": "negativo"}, {"aspect": "preco", "sentiment": "negativo"}]],
        "sentimento": ["neutro", "negativo"],
        "urgente": [False, True],
    })
    s = score(result, truth)
    assert s["aspecto_recall"] == 1.0
    assert s["aspecto_precisao"] == 0.75
    assert s["sentimento_por_aspecto"] == 2 / 3
    assert s["sentimento_geral"] == 1.0
    assert s["urgentes_encontrados"] == 1.0 and s["falsos_urgentes"] == 0
