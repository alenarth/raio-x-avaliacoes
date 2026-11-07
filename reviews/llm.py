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


def flash_models_by_preference(model_names: list[str]) -> list[str]:
    """Gemini Flash models this key can use, best first: newest stable
    versions, then previews. Names change every few months, so we read
    them from the API instead of hard-coding one."""
    def version(name: str) -> tuple[int, ...]:
        return tuple(int(n) for n in re.findall(r"\d+", name.split("flash")[0]))

    names = [n.removeprefix("models/") for n in model_names]
    flash = [n for n in names if n.startswith("gemini-") and "flash" in n and not any(s in n for s in _SKIP)]
    def newest_first(group: list[str]) -> list[str]:
        return sorted(set(group), key=lambda n: (version(n), -len(n)), reverse=True)

    ordered = newest_first([n for n in flash if "preview" not in n]) + newest_first([n for n in flash if "preview" in n])
    if not ordered:
        raise RuntimeError("Nenhum modelo Gemini Flash disponível para esta chave de API.")
    return ordered


def pick_flash_model(model_names: list[str]) -> str:
    return flash_models_by_preference(model_names)[0]


def available_models(client) -> list[str]:
    return [m.name for m in client.models.list() if "generateContent" in (getattr(m, "supported_actions", None) or [])]


MIN_SECONDS_BETWEEN_CALLS = 5.0  # keeps us under the free tier's requests-per-minute limit
SERVER_RETRY_SECONDS = 5.0
REQUEST_TIMEOUT_MS = 90_000  # an overloaded model can hang: give up and try the next one


class QuotaError(RuntimeError):
    """The Gemini free tier limit was hit (HTTP 429)."""


def make_client(api_key: str):
    """Gemini client without silent automatic retries: on a quota error we
    want to tell the user right away instead of hanging for minutes."""
    from google import genai
    from google.genai import types

    try:
        options = types.HttpOptions(timeout=REQUEST_TIMEOUT_MS, retry_options=types.HttpRetryOptions(attempts=1))
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
        chosen = model or os.environ.get("GEMINI_MODEL")
        self._models = [chosen] if chosen else flash_models_by_preference(available_models(self._client))[:3]
        self.model = self._models[0]
        self._last_call = 0.0
        self._low_thinking = True

    def _with_low_thinking(self, config):
        """Extraction and classification don't need long reasoning: asking for
        low thinking makes answers several times faster. Falls back silently on
        SDK versions or models that don't support it."""
        if config is None or not self._low_thinking:
            return config
        try:
            from google.genai import types

            return config.model_copy(update={"thinking_config": types.ThinkingConfig(thinking_level="low")})
        except Exception:
            self._low_thinking = False
            return config

    def _call_once(self, contents, config):
        from google.genai import errors

        wait = MIN_SECONDS_BETWEEN_CALLS - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        try:
            try:
                return self._client.models.generate_content(
                    model=self.model, contents=contents, config=self._with_low_thinking(config))
            except errors.ClientError as error:
                if error.code != 400 or not self._low_thinking or "think" not in str(error).lower():
                    raise
                self._low_thinking = False  # model rejects the option: retry without it
                return self._client.models.generate_content(model=self.model, contents=contents, config=config)
        finally:
            self._last_call = time.monotonic()

    def _generate(self, contents, config):
        """Call Gemini. If the model is overloaded (5xx) or out of free quota
        (429), retry once and then fall back to the next Flash model, since
        each model has its own capacity and quota."""
        from google.genai import errors

        last_error = None
        for model in list(self._models[self._models.index(self.model):]):
            self.model = model
            for attempt in range(2):
                try:
                    return self._call_once(contents, config)
                except errors.ServerError as error:
                    last_error = error
                    time.sleep(SERVER_RETRY_SECONDS * (attempt + 1))
                except errors.ClientError as error:
                    if error.code != 429:
                        raise RuntimeError(f"Erro da API do Gemini ({error.code}): {str(error)[:400]}") from error
                    last_error = error
                    break  # quota is per model: try the next one
                except Exception as error:  # network timeouts (httpx) when the model hangs
                    if "timeout" not in type(error).__name__.lower():
                        raise
                    last_error = error
                    break
        detail = str(last_error)[:400]
        if isinstance(last_error, errors.ClientError):
            raise QuotaError(
                "Limite do plano gratuito do Gemini atingido. Espere alguns minutos "
                f"(ou até o dia seguinte, se for o limite diário). Detalhe: {detail}"
            ) from last_error
        raise RuntimeError(f"O Gemini está sobrecarregado no momento. Tente de novo em instantes. Detalhe: {detail}") from last_error

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
