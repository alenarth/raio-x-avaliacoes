"""Raio-X de Avaliações: dashboard that turns raw delivery reviews into
priorities, using an LLM to classify each review and to draft an action plan."""

import hashlib
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from reviews.analysis import action_plan, aspect_table, classify, kpis, prepare, weekly_trend
from reviews.evaluation import score
from reviews.llm import GeminiReviewLLM
from reviews.schema import ASPECT_LABELS

EXAMPLE = Path("data/exemplo_avaliacoes.csv")
TRUTH = Path("data/exemplo_avaliacoes_gabarito.csv")
SENTIMENT_COLORS = ["#d64545", "#9e9e9e", "#2e9e5b"]  # negativo, neutro, positivo

st.set_page_config(page_title="Raio-X de Avaliações", page_icon="📊", layout="wide")


@st.cache_resource
def get_llm() -> GeminiReviewLLM:
    return GeminiReviewLLM()


@st.cache_resource
def ensure_example() -> None:
    """The fictional example is generated from a fixed seed (same data everywhere)."""
    if not EXAMPLE.exists() or not TRUTH.exists():
        from scripts.generate_example import generate

        generate(EXAMPLE.parent)


def run_evaluation() -> None:
    truth = pd.read_csv(TRUTH)
    df_eval = prepare(truth, "texto", "nota", "data")
    bar = st.progress(0.0, text="Avaliando...")
    start = time.perf_counter()
    result = classify(df_eval, get_llm(), on_progress=lambda p: bar.progress(p, text=f"Avaliando... {p:.0%}"))
    seconds = time.perf_counter() - start
    bar.empty()
    s = score(result, truth)
    st.markdown(
        "| Métrica | Valor |\n|---|---|\n"
        f"| Modelo | {get_llm().model} |\n"
        f"| Avaliações | {len(df_eval)} |\n"
        f"| Assuntos identificados (F1) | {s['aspecto_f1']:.1%} |\n"
        f"| Sentimento por assunto correto | {s['sentimento_por_aspecto']:.1%} |\n"
        f"| Sentimento geral correto | {s['sentimento_geral']:.1%} |\n"
        f"| Casos urgentes encontrados | {s['urgentes_encontrados']:.0%} |\n"
        f"| Falsos alarmes de urgência | {s['falsos_urgentes']} |\n"
        f"| Tempo total | {seconds:.0f}s |"
    )


def guess(columns: list[str], *hints: str) -> int:
    for i, col in enumerate(columns):
        if any(h in col.lower() for h in hints):
            return i
    return 0


def fingerprint(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df["texto"]).values.tobytes()).hexdigest()


ensure_example()

# ---------- Sidebar: data source ----------
with st.sidebar:
    st.header("1. Dados")
    uploaded = st.file_uploader("CSV com avaliações", type="csv")
    use_example = st.toggle("Usar exemplo fictício", value=uploaded is None)
    st.caption("O exemplo é um restaurante inventado, com 152 avaliações em 8 semanas.")

    if use_example or uploaded is None:
        raw = pd.read_csv(EXAMPLE)
    else:
        raw = pd.read_csv(uploaded, sep=None, engine="python")  # detects , or ;

    columns = list(raw.columns)
    none = "(não tem)"
    text_col = st.selectbox("Coluna do texto", columns, index=guess(columns, "texto", "coment", "review", "avalia"))
    rating_col = st.selectbox("Coluna da nota", [none] + columns, index=guess([none] + columns, "nota", "rating", "estrela"))
    date_col = st.selectbox("Coluna da data", [none] + columns, index=guess([none] + columns, "data", "date"))

    st.header("2. Premissa")
    manual_seconds = st.slider("Segundos para ler e anotar 1 avaliação à mão", 10, 120, 30)

    st.header("3. Qualidade da IA")
    st.caption("Compara a classificação da IA com o gabarito do exemplo fictício.")
    evaluate_clicked = st.button("Rodar avaliação")

df = prepare(raw, text_col, None if rating_col == none else rating_col, None if date_col == none else date_col)

# ---------- Header ----------
st.title("📊 Raio-X de Avaliações")
st.write(
    "Lê todas as avaliações do restaurante com IA, separa por assunto (comida, entrega, embalagem...) "
    "e mostra **o que mais gera reclamação, o que está piorando e o que fazer primeiro**."
)

if evaluate_clicked:
    with st.expander("Avaliação do classificador", expanded=True):
        try:
            run_evaluation()
        except RuntimeError as error:
            st.error(str(error))

key = fingerprint(df)
if st.session_state.get("key") != key:
    st.session_state.pop("result", None)
    st.session_state.pop("plan", None)
    st.session_state["key"] = key

if "result" not in st.session_state:
    st.info(f"{len(df)} avaliações carregadas. Clique para analisar.")
    if st.button("Analisar com IA", type="primary"):
        try:
            llm = get_llm()
        except RuntimeError as error:
            st.error(str(error))
            st.stop()
        bar = st.progress(0.0, text="Classificando avaliações...")
        start = time.perf_counter()
        try:
            result = classify(df, llm, on_progress=lambda p: bar.progress(p, text=f"Classificando... {p:.0%}"))
        except RuntimeError as error:
            bar.empty()
            st.error(str(error))
            st.stop()
        st.session_state["result"] = result
        st.session_state["seconds"] = time.perf_counter() - start
        bar.empty()
        st.rerun()
    st.dataframe(df.head(20), hide_index=True)
    st.stop()

result: pd.DataFrame = st.session_state["result"]
info = kpis(result)

# ---------- KPIs ----------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Avaliações", info["total"])
c2.metric("Negativas", f"{info['pct_negativas']:.0%}")
c3.metric("Nota média", "—" if info["nota_media"] is None else f"{info['nota_media']:.1f}")
c4.metric("Casos urgentes", info["urgentes"])
manual_min = info["total"] * manual_seconds / 60
c5.metric("Tempo economizado", f"{manual_min:.0f} min",
          help=f"Leitura manual estimada: {manual_min:.0f} min. IA: {st.session_state['seconds']:.0f} s.")
st.caption(f"Classificado com {get_llm().model} em {st.session_state['seconds']:.0f}s.")
if info["nao_classificadas"]:
    st.warning(f"{info['nao_classificadas']} avaliações não foram classificadas pela IA e ficaram de fora dos gráficos.")

overview, trend_tab, urgent_tab, all_tab, plan_tab = st.tabs(
    ["Visão geral", "Tendência", "Urgentes", "Todas as avaliações", "Plano de ação"])

with overview:
    table = aspect_table(result)
    left, right = st.columns([3, 2])
    left.subheader("Menções por assunto")
    left.bar_chart(table[["negativo", "neutro", "positivo"]], color=SENTIMENT_COLORS, horizontal=True)
    right.subheader("Ranking de reclamações")
    right.dataframe(table, column_config={"% negativas": st.column_config.NumberColumn(format="%d%%")})

with trend_tab:
    trend = weekly_trend(result)
    if trend is None:
        st.info("Selecione a coluna de data na barra lateral para ver a evolução semanal.")
    else:
        st.subheader("% de avaliações com reclamação, por semana")
        st.line_chart(trend)
        st.caption("Cada linha é um assunto. Linha subindo = problema piorando.")

with urgent_tab:
    urgent = result[result["urgente"]]
    if urgent.empty:
        st.success("Nenhum caso urgente.")
    else:
        st.error(f"{len(urgent)} caso(s) para responder hoje.")
        st.dataframe(urgent[[c for c in ["data", "nota", "texto", "resumo"] if c in urgent]], hide_index=True)

with all_tab:
    f1, f2 = st.columns(2)
    chosen_aspects = f1.multiselect("Assunto", list(ASPECT_LABELS), format_func=ASPECT_LABELS.get)
    chosen_sentiments = f2.multiselect("Sentimento", ["negativo", "neutro", "positivo"])
    view = result
    if chosen_aspects:
        view = view[view["aspectos"].map(lambda a: bool(set(a) & set(chosen_aspects)))]
    if chosen_sentiments:
        view = view[view["sentimento"].isin(chosen_sentiments)]
    shown = view.assign(aspectos=view["aspectos"].map(lambda a: ", ".join(ASPECT_LABELS[x] for x in a)))
    st.dataframe(shown.drop(columns="id"), hide_index=True)
    st.download_button("Baixar CSV classificado", shown.to_csv(index=False).encode("utf-8-sig"),
                       "avaliacoes_classificadas.csv", "text/csv")

with plan_tab:
    st.write("A IA recebe só os números do painel e os resumos das reclamações, e propõe prioridades para a semana.")
    if st.button("Gerar plano de ação", type="primary"):
        with st.spinner("Escrevendo plano..."):
            try:
                st.session_state["plan"] = action_plan(result, get_llm())
            except RuntimeError as error:
                st.error(str(error))
    if "plan" in st.session_state:
        st.markdown(st.session_state["plan"])
        st.download_button("Baixar plano (.md)", st.session_state["plan"], "plano_de_acao.md", "text/markdown")
