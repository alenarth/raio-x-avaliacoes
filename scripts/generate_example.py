"""Generate a FICTIONAL review dataset for demos and evaluation.

The restaurant and reviews are invented. The data has a built-in story so the
dashboard has something to find: delivery complaints grow over the last weeks.
Each review keeps its true labels (aspects/sentiment) for evaluating the LLM.

Usage:  python -m scripts.generate_example
"""

import random
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

TEMPLATES = {
    ("comida", "positivo"): [
        "Comida muito saborosa, chegou quentinha", "O hambúrguer estava perfeito, ponto da carne certinho",
        "Melhor feijoada que pedi esse ano", "Tempero excelente, porção bem servida",
        "A batata estava crocante e o molho da casa é ótimo",
    ],
    ("comida", "negativo"): [
        "Carne veio crua por dentro", "Comida sem sal e sem gosto", "Arroz empapado, muito ruim",
        "A batata chegou murcha e fria", "Pizza com pouquíssimo recheio",
    ],
    ("entrega", "positivo"): [
        "Entrega super rápida", "Chegou antes do horário previsto", "Entregador muito educado e rápido",
    ],
    ("entrega", "negativo"): [
        "Demorou mais de 1h30 para chegar", "Pedido atrasou muito, chegou frio",
        "Entrega levou o dobro do tempo informado no app", "Esperei quase duas horas pelo pedido",
        "Atraso enorme de novo, já é a terceira vez",
    ],
    ("embalagem", "negativo"): [
        "Embalagem chegou aberta e o molho vazou", "O caldo vazou todo dentro da sacola",
        "Embalagem amassada, comida toda revirada",
    ],
    ("embalagem", "positivo"): ["Embalagem caprichada, nada vazou", "Gostei da embalagem, bem lacrada"],
    ("preco", "negativo"): ["Muito caro para o tamanho da porção", "Preço subiu bastante e a porção diminuiu"],
    ("preco", "neutro"): ["Preço ok, dentro do esperado", "Valor justo, nada demais"],
    ("preco", "positivo"): ["Ótimo custo-benefício", "Preço muito bom pela quantidade"],
    ("atendimento", "negativo"): [
        "Mandei mensagem e ninguém respondeu", "Atendimento grosso quando reclamei do atraso",
    ],
    ("atendimento", "positivo"): ["Resolveram meu problema rapidinho pelo chat", "Atendimento muito atencioso"],
    ("pedido", "negativo"): [
        "Veio sem o refrigerante que eu paguei", "Pedi sem cebola e veio com cebola",
        "Faltou a sobremesa do combo", "Mandaram o sabor errado da pizza",
    ],
}
URGENT = [
    ("Encontrei um fio de cabelo na comida, nojento", "comida:negativo"),
    ("Passei mal depois de comer, acho que estava estragado", "comida:negativo"),
    ("Pedido nunca chegou e cobraram no cartão", "entrega:negativo|pedido:negativo"),
]
RATING = {"positivo": (4, 5), "neutro": (3, 4), "negativo": (1, 2)}
START = date(2026, 9, 7)
WEEKS = 8


def aspect_weights(week: int) -> dict[tuple[str, str], float]:
    late = week >= WEEKS - 3  # the "story": delivery gets worse at the end
    return {
        ("comida", "positivo"): 6, ("comida", "negativo"): 1.5,
        ("entrega", "positivo"): 2 if late else 3, ("entrega", "negativo"): 5 if late else 1,
        ("embalagem", "negativo"): 1, ("embalagem", "positivo"): 0.7,
        ("preco", "negativo"): 1, ("preco", "neutro"): 1, ("preco", "positivo"): 1,
        ("atendimento", "negativo"): 1.3 if late else 0.6, ("atendimento", "positivo"): 0.8,
        ("pedido", "negativo"): 1.2,
    }


def make_review(rng: random.Random, week: int) -> dict:
    weights = aspect_weights(week)
    keys = list(weights)
    first = rng.choices(keys, weights=[weights[k] for k in keys])[0]
    parts, aspects, sentiments = [rng.choice(TEMPLATES[first])], [first[0]], [first[1]]
    if rng.random() < 0.35:  # some reviews mention a second aspect
        second = rng.choices(keys, weights=[weights[k] for k in keys])[0]
        if second[0] != first[0]:
            parts.append(rng.choice(TEMPLATES[second]).lower())
            aspects.append(second[0])
            sentiments.append(second[1])
    if "negativo" in sentiments and "positivo" in sentiments:
        sentiment = "neutro"  # mixed review: one good point, one bad point
    else:
        sentiment = sentiments[0]
    if len(set(sentiments)) > 1:
        text = ", mas ".join(parts)
    else:
        text = ". ".join([parts[0]] + [p[0].upper() + p[1:] for p in parts[1:]])
    return {"texto": text + ".", "mencoes_reais": "|".join(f"{a}:{s}" for a, s in zip(aspects, sentiments)),
            "sentimento_real": sentiment, "urgente_real": False}


def generate(out: str | Path = "data") -> int:
    rng = random.Random(7)
    rows = []
    for week in range(WEEKS):
        for _ in range(rng.randint(15, 22)):
            review = make_review(rng, week)
            review["data"] = START + timedelta(days=week * 7 + rng.randint(0, 6))
            review["nota"] = rng.randint(*RATING[review["sentimento_real"]])
            rows.append(review)
    for text, labels in URGENT:
        rows.append({"texto": text, "mencoes_reais": labels, "sentimento_real": "negativo",
                     "urgente_real": True, "nota": 1,
                     "data": START + timedelta(days=rng.randint(WEEKS * 7 - 21, WEEKS * 7 - 1))})

    df = pd.DataFrame(rows).sort_values("data").reset_index(drop=True)
    df["data"] = pd.to_datetime(df["data"]).dt.strftime("%d/%m/%Y")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    df[["data", "nota", "texto"]].to_csv(out / "exemplo_avaliacoes.csv", index=False, encoding="utf-8")
    df.to_csv(out / "exemplo_avaliacoes_gabarito.csv", index=False, encoding="utf-8")
    return len(df)


def main() -> None:
    n = generate()
    print(f"{n} avaliações fictícias em data/exemplo_avaliacoes.csv (+ gabarito)")


if __name__ == "__main__":
    main()
