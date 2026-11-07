"""GeminiCaller with a stubbed google.genai: no network, no key needed."""

import sys
import types as pytypes

import pytest


class FakeClientError(Exception):
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
            return [FakeModel("models/gemini-3-flash"), FakeModel("models/gemini-3-pro")]

        def generate_content(self, model, contents, config):
            behaviour["calls"].append(model)
            if behaviour["error"]:
                raise behaviour["error"]
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
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    return GeminiCaller(api_key="fake")


def test_picks_model_from_api(fake_genai, monkeypatch):
    caller = make_caller(monkeypatch)
    assert caller.model == "gemini-3-flash"
    assert caller._generate("oi", None).text == '{"ok": true}'


def test_quota_error_is_reported_not_retried(fake_genai, monkeypatch):
    from reviews.llm import QuotaError

    caller = make_caller(monkeypatch)
    fake_genai["error"] = FakeClientError(429, "RESOURCE_EXHAUSTED")
    with pytest.raises(QuotaError, match="Limite do plano gratuito"):
        caller._generate("oi", None)
    assert len(fake_genai["calls"]) == 1


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
