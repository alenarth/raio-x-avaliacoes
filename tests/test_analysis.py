"""Tests with a fake LLM that answers from the example's ground truth."""

import pandas as pd
import pytest

from reviews.analysis import UNCLASSIFIED, aspect_table, classify, kpis, plan_context, prepare, weekly_trend
from reviews.schema import AspectMention, ReviewAnalysis



class TruthLLM:
    """Answers with the true labels; can be told to drop some ids or invent others."""

    def __init__(self, truth: pd.DataFrame, drop: set[int] = frozenset()):
        self.truth = truth.set_index("texto")
        self.drop = drop
        self.batch_sizes = []

    def classify(self, batch):
        self.batch_sizes.append(len(batch))
        out = [ReviewAnalysis(id=999_999, mentions=[], sentiment="neutro", urgent=False, summary="inventado")]
        for row in batch:
            if row["id"] in self.drop:
                continue
            t = self.truth.loc[row["texto"]]
            t = t.iloc[0] if isinstance(t, pd.DataFrame) else t
            labels = [AspectMention(aspect=a, sentiment=s) for a, s in
                      (pair.split(":") for pair in t["mencoes_reais"].split("|"))]
            out.append(ReviewAnalysis(id=row["id"], mentions=labels,
                                      sentiment=t["sentimento_real"], urgent=bool(t["urgente_real"]),
                                      summary=row["texto"][:40]))
        return out

    def write(self, prompt):
        return "plano"


@pytest.fixture(scope="module")
def truth(tmp_path_factory):
    from scripts.generate_example import generate

    folder = tmp_path_factory.mktemp("data")
    generate(folder)
    return pd.read_csv(folder / "exemplo_avaliacoes_gabarito.csv")


@pytest.fixture
def prepared(truth):
    return prepare(truth, "texto", "nota", "data")


def test_prepare_maps_columns_and_drops_empty():
    raw = pd.DataFrame({"comentario": ["bom", "  ", "ruim"], "estrelas": ["5", "3", "x"]})
    df = prepare(raw, "comentario", "estrelas")
    assert df["texto"].tolist() == ["bom", "ruim"]
    assert df["id"].tolist() == [0, 1]
    assert df["nota"].isna().tolist() == [False, True]


def test_classify_batches_and_ignores_invented_ids(truth, prepared):
    llm = TruthLLM(truth)
    result = classify(prepared, llm)
    assert sum(llm.batch_sizes) == len(prepared) and max(llm.batch_sizes) <= 25
    assert len(result) == len(prepared)
    assert (result["sentimento"] != UNCLASSIFIED).all()


def test_missing_answers_are_marked_not_guessed(truth, prepared):
    result = classify(prepared, TruthLLM(truth, drop={0, 5}))
    assert result.loc[[0, 5], "sentimento"].eq(UNCLASSIFIED).all()
    assert kpis(result)["nao_classificadas"] == 2


def test_dashboard_finds_the_delivery_story(truth, prepared):
    result = classify(prepared, TruthLLM(truth))
    info = kpis(result)
    assert info["urgentes"] == 3
    assert 1 <= info["nota_media"] <= 5

    table = aspect_table(result)
    assert table.index[0] == "Entrega"  # most complaints

    trend = weekly_trend(result)
    first_weeks, last_weeks = trend["Entrega"].iloc[:4].mean(), trend["Entrega"].iloc[-3:].mean()
    assert last_weeks > first_weeks  # delivery is getting worse

    context = plan_context(result)
    assert "Entrega" in context and "percentual_negativas" in context


def test_trend_needs_dates(truth):
    df = prepare(truth, "texto", "nota")
    result = classify(df, TruthLLM(truth))
    assert weekly_trend(result) is None
