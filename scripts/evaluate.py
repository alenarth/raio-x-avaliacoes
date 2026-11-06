"""Measure the LLM classifier against the labeled example.

Usage:  python -m scripts.evaluate
"""

import time

import pandas as pd

from reviews.analysis import classify, prepare
from reviews.evaluation import score
from reviews.llm import GeminiReviewLLM


def main() -> None:
    truth = pd.read_csv("data/exemplo_avaliacoes_gabarito.csv")
    df = prepare(truth, "texto", "nota", "data")
    start = time.perf_counter()
    result = classify(df, GeminiReviewLLM())
    seconds = time.perf_counter() - start
    s = score(result, truth)

    print("| Métrica | Valor |\n|---|---|")
    print(f"| Avaliações | {len(df)} |")
    print(f"| Assuntos identificados (F1) | {s['aspecto_f1']:.1%} |")
    print(f"| Sentimento por assunto correto | {s['sentimento_por_aspecto']:.1%} |")
    print(f"| Sentimento geral correto | {s['sentimento_geral']:.1%} |")
    print(f"| Casos urgentes encontrados | {s['urgentes_encontrados']:.0%} |")
    print(f"| Falsos alarmes de urgência | {s['falsos_urgentes']} |")
    print(f"| Tempo total | {seconds:.0f}s |")


if __name__ == "__main__":
    main()
