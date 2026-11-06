"""Classification in batches + the aggregations shown on the dashboard.

Everything here works on a pandas DataFrame with the columns:
    id, texto, nota (optional), data (optional)
and adds: mencoes (list of {aspect, sentiment}), aspectos, sentimento, urgente, resumo.
"""

import json
from typing import Callable

import pandas as pd

from .llm import PLAN_PROMPT, ReviewLLM
from .schema import ASPECT_LABELS, ASPECTS

BATCH_SIZE = 25
UNCLASSIFIED = "não classificado"


def prepare(df: pd.DataFrame, text_col: str, rating_col: str | None = None,
            date_col: str | None = None) -> pd.DataFrame:
    """Map the user's columns to our names and drop empty reviews."""
    out = pd.DataFrame({"texto": df[text_col].astype(str).str.strip()})
    if rating_col:
        out["nota"] = pd.to_numeric(df[rating_col], errors="coerce")
    if date_col:
        out["data"] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    out = out[out["texto"].str.len() > 0].reset_index(drop=True)
    out.insert(0, "id", range(len(out)))
    return out


def classify(df: pd.DataFrame, llm: ReviewLLM,
             on_progress: Callable[[float], None] | None = None) -> pd.DataFrame:
    """Send reviews in batches; anything the LLM skips is marked, never guessed."""
    results: dict[int, dict] = {}
    rows = df[["id", "texto"]].to_dict("records")
    batches = [rows[i:i + BATCH_SIZE] for i in range(0, len(rows), BATCH_SIZE)]

    for n, batch in enumerate(batches, start=1):
        valid_ids = {r["id"] for r in batch}
        for analysis in llm.classify(batch):
            if analysis.id in valid_ids:
                results[analysis.id] = analysis.model_dump()
        if on_progress:
            on_progress(n / len(batches))

    out = df.copy()
    out["mencoes"] = out["id"].map(lambda i: results[i]["mentions"] if i in results else [])
    out["aspectos"] = out["mencoes"].map(lambda ms: sorted({m["aspect"] for m in ms}))
    out["sentimento"] = out["id"].map(lambda i: results[i]["sentiment"] if i in results else UNCLASSIFIED)
    out["urgente"] = out["id"].map(lambda i: results[i]["urgent"] if i in results else False)
    out["resumo"] = out["id"].map(lambda i: results[i]["summary"] if i in results else "")
    return out


def kpis(df: pd.DataFrame) -> dict:
    classified = df[df["sentimento"] != UNCLASSIFIED]
    total = len(classified)
    negative = int((classified["sentimento"] == "negativo").sum())
    return {
        "total": len(df),
        "nao_classificadas": len(df) - total,
        "pct_negativas": negative / total if total else 0.0,
        "nota_media": float(df["nota"].mean()) if "nota" in df and df["nota"].notna().any() else None,
        "urgentes": int(df["urgente"].sum()),
    }


def mentions(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (review, aspect) with the sentiment about that aspect."""
    rows = [
        {"id": r.id, "data": getattr(r, "data", None), "aspecto": m["aspect"], "sentimento": m["sentiment"]}
        for r in df.itertuples() for m in r.mencoes
    ]
    return pd.DataFrame(rows, columns=["id", "data", "aspecto", "sentimento"])


def aspect_table(df: pd.DataFrame) -> pd.DataFrame:
    """Mentions per aspect split by sentiment, sorted by number of complaints."""
    m = mentions(df)
    table = (
        pd.crosstab(m["aspecto"], m["sentimento"])
        .reindex(index=list(ASPECTS), columns=["negativo", "neutro", "positivo"], fill_value=0)
    )
    table.index = [ASPECT_LABELS[a] for a in table.index]
    totals = table.sum(axis=1)
    table["% negativas"] = (table["negativo"] / totals.where(totals > 0) * 100).round(0)
    return table.sort_values("negativo", ascending=False)


def weekly_trend(df: pd.DataFrame) -> pd.DataFrame | None:
    """% of the week's reviews that complain about each aspect."""
    if "data" not in df or df["data"].isna().all():
        return None
    dated = df.dropna(subset=["data"]).copy()
    dated = dated[dated["sentimento"] != UNCLASSIFIED]
    dated["semana"] = dated["data"].dt.to_period("W").dt.start_time
    weekly_total = dated.groupby("semana").size()
    m = mentions(dated)
    m = m[m["sentimento"] == "negativo"].drop_duplicates(["id", "aspecto"])
    m["semana"] = m["data"].dt.to_period("W").dt.start_time
    counts = pd.crosstab(m["semana"], m["aspecto"]).reindex(weekly_total.index, fill_value=0)
    trend = counts.div(weekly_total, axis=0).mul(100).round(1)
    trend.columns = [ASPECT_LABELS[c] for c in trend.columns]
    return trend


def plan_context(df: pd.DataFrame, max_examples: int = 30) -> str:
    """Compact, factual summary handed to the LLM to write the action plan."""
    info = kpis(df)
    trend = weekly_trend(df)
    complaints = df[df["mencoes"].map(lambda ms: any(m["sentiment"] == "negativo" for m in ms))]
    negatives = complaints["resumo"].head(max_examples).tolist()
    context = {
        "total_avaliacoes": info["total"],
        "percentual_negativas": round(info["pct_negativas"] * 100, 1),
        "nota_media": None if info["nota_media"] is None else round(info["nota_media"], 2),
        "casos_urgentes": info["urgentes"],
        "por_aspecto": aspect_table(df).drop(columns="% negativas").to_dict("index"),
        "percentual_negativas_por_semana_e_aspecto": (
            None if trend is None else {str(k.date()): v for k, v in trend.to_dict("index").items()}
        ),
        "exemplos_de_reclamacoes": negatives,
    }
    return json.dumps(context, ensure_ascii=False, indent=1, default=str)


def action_plan(df: pd.DataFrame, llm: ReviewLLM) -> str:
    return llm.write(PLAN_PROMPT + plan_context(df))
