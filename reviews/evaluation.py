"""Compare LLM classifications with hand/true labels."""

import pandas as pd


def parse_labels(text: str) -> set[tuple[str, str]]:
    """'comida:positivo|entrega:negativo' -> {('comida','positivo'), ('entrega','negativo')}"""
    if not isinstance(text, str) or not text:
        return set()
    return {tuple(pair.split(":")) for pair in text.split("|")}


def score(result: pd.DataFrame, truth: pd.DataFrame) -> dict:
    """`result` from analysis.classify; `truth` with mencoes_reais, sentimento_real, urgente_real
    (same row order)."""
    tp_aspect = fp_aspect = fn_aspect = tp_pair = 0
    for predicted, expected in zip(result["mencoes"], truth["mencoes_reais"]):
        pred = {(m["aspect"], m["sentiment"]) for m in predicted}
        true = parse_labels(expected)
        pred_aspects, true_aspects = {a for a, _ in pred}, {a for a, _ in true}
        tp_aspect += len(pred_aspects & true_aspects)
        fp_aspect += len(pred_aspects - true_aspects)
        fn_aspect += len(true_aspects - pred_aspects)
        tp_pair += len(pred & true)

    precision = tp_aspect / (tp_aspect + fp_aspect) if tp_aspect + fp_aspect else 0.0
    recall = tp_aspect / (tp_aspect + fn_aspect) if tp_aspect + fn_aspect else 0.0
    urgent_true = truth["urgente_real"].astype(bool)
    return {
        "aspecto_precisao": precision,
        "aspecto_recall": recall,
        "aspecto_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "sentimento_por_aspecto": tp_pair / tp_aspect if tp_aspect else 0.0,
        "sentimento_geral": float((result["sentimento"].values == truth["sentimento_real"].values).mean()),
        "urgentes_encontrados": float(result["urgente"][urgent_true.values].mean()) if urgent_true.any() else 1.0,
        "falsos_urgentes": int((result["urgente"].values & ~urgent_true.values).sum()),
    }
