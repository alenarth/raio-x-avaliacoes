import json
import os
from typing import Protocol

from .schema import BatchAnalysis, ReviewAnalysis

CLASSIFY_PROMPT = """Você analisa avaliações de clientes de um restaurante de delivery.
Para cada avaliação, identifique:
- mentions: cada aspecto mencionado (comida, entrega, embalagem, preco, atendimento,
  pedido = item faltando ou errado) com o sentimento sobre AQUELE aspecto.
  Ex.: "comida ótima, mas atrasou" -> comida/positivo e entrega/negativo.
  Lista vazia se nenhum aspecto for mencionado.
- sentiment: positivo, neutro ou negativo (o tom geral; elogio + crítica = neutro).
- urgent: true apenas para casos graves (intoxicação, objeto estranho na comida,
  pedido não entregue, cobrança indevida, ameaça de denúncia).
- summary: o ponto principal em até 10 palavras, em português.
Devolva exatamente um resultado por id recebido.

Avaliações (JSON):
"""

PLAN_PROMPT = """Você é consultor de operações de restaurantes de delivery.
Com base SOMENTE nos dados abaixo (não invente números), escreva em português um
plano de ação curto em Markdown:
- Um parágrafo de diagnóstico (2-3 frases) citando os números mais importantes.
- De 3 a 5 prioridades, da mais para a menos importante. Para cada uma:
  **título**, a evidência nos dados, uma ação concreta para esta semana e como medir se funcionou.
- Um ponto forte a manter, se houver.

Dados:
"""


class ReviewLLM(Protocol):
    def classify(self, batch: list[dict]) -> list[ReviewAnalysis]: ...
    def write(self, prompt: str) -> str: ...


class GeminiReviewLLM:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        from google import genai

        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Defina a variável de ambiente GEMINI_API_KEY.")
        self._client = genai.Client(api_key=api_key)
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-flash-latest")

    def classify(self, batch: list[dict]) -> list[ReviewAnalysis]:
        from google.genai import types

        response = self._client.models.generate_content(
            model=self.model,
            contents=CLASSIFY_PROMPT + json.dumps(batch, ensure_ascii=False),
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=BatchAnalysis,
                temperature=0,
            ),
        )
        return BatchAnalysis.model_validate_json(response.text).reviews

    def write(self, prompt: str) -> str:
        from google.genai import types

        response = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.3),
        )
        return response.text
