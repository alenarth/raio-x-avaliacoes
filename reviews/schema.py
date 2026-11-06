from typing import Literal

from pydantic import BaseModel, Field

Aspect = Literal["comida", "entrega", "embalagem", "preco", "atendimento", "pedido"]
Sentiment = Literal["positivo", "neutro", "negativo"]

ASPECTS: tuple[str, ...] = Aspect.__args__
ASPECT_LABELS = {
    "comida": "Comida",
    "entrega": "Entrega",
    "embalagem": "Embalagem",
    "preco": "Preço",
    "atendimento": "Atendimento",
    "pedido": "Pedido errado/incompleto",
}


class AspectMention(BaseModel):
    aspect: Aspect
    sentiment: Sentiment = Field(description="Sentimento sobre ESTE aspecto específico")


class ReviewAnalysis(BaseModel):
    id: int = Field(description="O mesmo id recebido na entrada")
    mentions: list[AspectMention] = Field(description="Cada aspecto mencionado, com o sentimento sobre ele")
    sentiment: Sentiment = Field(description="Tom geral da avaliação")
    urgent: bool = Field(description="true se exige ação imediata: intoxicação, objeto estranho, "
                                     "pedido não entregue, cobrança indevida ou ameaça de denúncia")
    summary: str = Field(description="Resumo do ponto principal em até 10 palavras")


class BatchAnalysis(BaseModel):
    reviews: list[ReviewAnalysis]
