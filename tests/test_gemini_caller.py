"""GeminiCaller with a stubbed google.genai: no network, no key needed."""

import sys
import types as pytypes

import pytest


class FakeClientError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class FakeServerError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class FakeModel:
    def __init__(self, name):
        self.name = name
        self.supported_actions = ["generateContent"]


@pytest.fixture
def fake_genai(monkeypatch):
    behaviour = {"error": None, "calls": []}

    class Models:
        def list(self):
            return [FakeModel("models/gemini-3-flash"), FakeModel("models/gemini-2.5-flash"),
                    FakeModel("models/gemini-3-pro")]

        def generate_content(self, model, contents, config):
            behaviour["calls"].append(model)
            error = behaviour["error"]
            if callable(error):
                error = error(model)
            if error:
                raise error
            return pytypes.SimpleNamespace(text='{"ok": true}')

    class Client:
        def __init__(self, api_key, http_options=None):
            self.models = Models()

    genai = pytypes.ModuleType("google.genai")
    genai.Client = Client
    types_mod = pytypes.ModuleType("google.genai.types")
    types_mod.HttpOptions = lambda **kw: kw
    types_mod.HttpRetryOptions = lambda **kw: kw
    errors_mod = pytypes.ModuleType("google.genai.errors")
    errors_mod.ClientError = FakeClientError
    errors_mod.ServerError = FakeServerError
    genai.types, genai.errors = types_mod, errors_mod
    google = pytypes.ModuleType("google")
    google.genai = genai
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", types_mod)
    monkeypatch.setitem(sys.modules, "google.genai.errors", errors_mod)
    return behaviour


def make_caller(monkeypatch):
    from reviews.llm import GeminiCaller

    monkeypatch.setattr("reviews.llm.MIN_SECONDS_BETWEEN_CALLS", 0)
    monkeypatch.setattr("reviews.llm.SERVER_RETRY_SECONDS", 0)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    return GeminiCaller(api_key="fake")


def test_picks_model_from_api(fake_genai, monkeypatch):
    caller = make_caller(monkeypatch)
    assert caller.model == "gemini-3-flash"
    assert caller._generate("oi", None).text == '{"ok": true}'


def test_quota_error_tries_next_model_then_reports(fake_genai, monkeypatch):
    from reviews.llm import QuotaError

    caller = make_caller(monkeypatch)
    fake_genai["error"] = FakeClientError(429, "RESOURCE_EXHAUSTED")
    with pytest.raises(QuotaError, match="Limite do plano gratuito"):
        caller._generate("oi", None)
    assert fake_genai["calls"] == ["gemini-3-flash", "gemini-2.5-flash"]


def test_overloaded_model_falls_back_to_next(fake_genai, monkeypatch):
    caller = make_caller(monkeypatch)
    fake_genai["error"] = lambda model: FakeServerError(503, "UNAVAILABLE") if model == "gemini-3-flash" else None
    assert caller._generate("oi", None).text == '{"ok": true}'
    assert fake_genai["calls"] == ["gemini-3-flash", "gemini-3-flash", "gemini-2.5-flash"]
    assert caller.model == "gemini-2.5-flash"


def test_all_models_overloaded_is_a_clear_error(fake_genai, monkeypatch):
    caller = make_caller(monkeypatch)
    fake_genai["error"] = FakeServerError(503, "UNAVAILABLE")
    with pytest.raises(RuntimeError, match="sobrecarregado"):
        caller._generate("oi", None)


def test_other_client_errors_become_runtime_errors(fake_genai, monkeypatch):
    caller = make_caller(monkeypatch)
    fake_genai["error"] = FakeClientError(404, "model not found")
    with pytest.raises(RuntimeError, match="404"):
        caller._generate("oi", None)


def test_missing_key_is_a_clear_error(monkeypatch):
    from reviews.llm import GeminiCaller

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        GeminiCaller()


def test_retries_without_thinking_option_when_model_rejects_it(fake_genai, monkeypatch):
    caller = make_caller(monkeypatch)
    caller._with_low_thinking = lambda config: "low-thinking-config"
    seen = []
    original = caller._client.models.generate_content

    def generate(model, contents, config):
        seen.append(config)
        if config == "low-thinking-config":
            raise FakeClientError(400, "thinking_level is not supported")
        return original(model, contents, config)

    caller._client.models.generate_content = generate
    assert caller._generate("oi", "plain-config").text == '{"ok": true}'
    assert seen == ["low-thinking-config", "plain-config"] and caller._low_thinking is False
