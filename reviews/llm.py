import json
import os
import re
import time
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


MIN_SECONDS_BETWEEN_CALLS = 5.0  # keeps us under the free tier's requests-per-minute limit


class QuotaError(RuntimeError):
    """The Gemini free tier limit was hit (HTTP 429)."""


def make_client(api_key: str):
    """Gemini client without silent automatic retries: on a quota error we
    want to tell the user right away instead of hanging for minutes."""
    from google import genai
    from google.genai import types

    try:
        options = types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=1))
        return genai.Client(api_key=api_key, http_options=options)
    except (AttributeError, TypeError, ValueError):
        return genai.Client(api_key=api_key)


class GeminiCaller:
    """Shared plumbing: picks the model, spaces out calls, translates errors."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Defina a variável de ambiente GEMINI_API_KEY.")
        self._client = make_client(api_key)
        self.model = model or os.environ.get("GEMINI_MODEL") or pick_flash_model(available_models(self._client))
        self._last_call = 0.0

    def _generate(self, contents, config):
        from google.genai import errors

        wait = MIN_SECONDS_BETWEEN_CALLS - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        try:
            return self._client.models.generate_content(model=self.model, contents=contents, config=config)
        except errors.ClientError as error:
            detail = str(error)[:400]
            if error.code == 429:
                raise QuotaError(
                    "Limite do plano gratuito do Gemini atingido. Espere alguns minutos "
                    f"(ou até o dia seguinte, se for o limite diário). Detalhe: {detail}"
                ) from error
            raise RuntimeError(f"Erro da API do Gemini ({error.code}): {detail}") from error
        finally:
            self._last_call = time.monotonic()


class ReviewLLM(Protocol):
    def classify(self, batch: list[dict]) -> list[ReviewAnalysis]: ...
    def write(self, prompt: str) -> str: ...


class GeminiReviewLLM(GeminiCaller):
    def classify(self, batch: list[dict]) -> list[ReviewAnalysis]:
        from google.genai import types

        response = self._generate(
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

        response = self._generate(contents=prompt, config=types.GenerateContentConfig(temperature=0.3))
        return response.text
