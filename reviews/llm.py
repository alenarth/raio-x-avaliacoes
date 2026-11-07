import json
import os
import re
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


_SKIP = ("lite", "image", "tts", "audio", "live", "embedding", "thinking", "exp", "latest", "computer")


def pick_flash_model(model_names: list[str]) -> str:
    """Choose the newest Gemini Flash model from the ones this API key can use.

    Model names change every few months, so instead of hard-coding one we
    look at what the API offers. Stable versions win over previews.
    """
    def version(name: str) -> tuple[int, ...]:
        return tuple(int(n) for n in re.findall(r"\d+", name.split("flash")[0]))

    names = [n.removeprefix("models/") for n in model_names]
    flash = [n for n in names if n.startswith("gemini-") and "flash" in n and not any(s in n for s in _SKIP)]
    stable = [n for n in flash if "preview" not in n]
    candidates = stable or flash
    if not candidates:
        raise RuntimeError("Nenhum modelo Gemini Flash disponível para esta chave de API.")
    return max(candidates, key=lambda n: (version(n), -len(n)))


def available_models(client) -> list[str]:
    return [m.name for m in client.models.list() if "generateContent" in (getattr(m, "supported_actions", None) or [])]


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
        self.model = model or os.environ.get("GEMINI_MODEL") or pick_flash_model(available_models(self._client))

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
