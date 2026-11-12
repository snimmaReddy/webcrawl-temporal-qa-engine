"""Optional LLM step, provider-agnostic, standard library only (urllib).

Provider choice, first match wins:
  1. KB_LLM_PROVIDER=anthropic|gemini|openai|compatible forces one (KB_LLM=0 turns the step off).
  2. KB_LLM_BASE_URL (+ KB_LLM_API_KEY, KB_LLM_MODEL): any OpenAI-compatible server
     (Azure-style gateways, Groq, Together, OpenRouter, vLLM, Ollama, ...).
  3. ANTHROPIC_API_KEY  -> Claude (native Messages API)
  4. GEMINI_API_KEY / GOOGLE_API_KEY -> Gemini (native generateContent API)
  5. OPENAI_API_KEY -> OpenAI (chat completions)
  6. GROQ_API_KEY, MISTRAL_API_KEY, DEEPSEEK_API_KEY, OPENROUTER_API_KEY -> their OpenAI-compatible endpoints
KB_LLM_MODEL overrides the default model of whichever provider is picked; KB_LLM_TIMEOUT (seconds) caps one call.
No key -> `LLM.load()` returns None and the system answers with rules only.
"""
from __future__ import annotations

import json
import os
import ssl
import urllib.request

DEFAULTS = {
    "anthropic": ("https://api.anthropic.com/v1", "claude-haiku-5-5"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta", "gemini-2.5-flash"),
    "openai": ("https://api.openai.com/v1", "gpt-4.1-mini"),
}
COMPATIBLE = [  # (key variable, base url, default model)
    ("GROQ_API_KEY", "https://api.groq.com/openai/v1", "llama-3.3-70b-versatile"),
    ("MISTRAL_API_KEY", "https://api.mistral.ai/v1", "mistral-small-latest"),
    ("DEEPSEEK_API_KEY", "https://api.deepseek.com/v1", "deepseek-chat"),
    ("OPENROUTER_API_KEY", "https://openrouter.ai/api/v1", "openai/gpt-4.1-mini"),
]


def _env(*names):
    for n in names:
        v = os.environ.get(n, "").strip()
        if v:
            return v
    return ""


def _ssl_ctx():
    try:
        import certifi  # macOS python.org builds often lack system certificates
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        pass
    import os
    for f in ("/etc/ssl/cert.pem", "/etc/ssl/certs/ca-certificates.crt", "/etc/pki/tls/certs/ca-bundle.crt"):
        if os.path.exists(f) and not ssl.get_default_verify_paths().cafile:
            try:
                return ssl.create_default_context(cafile=f)
            except Exception:
                pass
    return ssl.create_default_context()


class LLM:
    def __init__(self, provider: str, base: str, key: str, model: str, timeout: float):
        self.provider, self.base, self.key, self.model, self.timeout = provider, base.rstrip("/"), key, model, timeout
        self.ctx = _ssl_ctx()
        self.failures = 0          # consecutive hard errors (bad key, no network); after 3 the step switches off
        self.timeouts = 0          # consecutive timeouts; after 8 the provider is too slow and the step switches off
        self.usage = {"calls": 0, "in": 0, "out": 0}   # token accounting (cost visibility in tests)

    @classmethod
    def load(cls):
        if os.environ.get("KB_LLM", "1") == "0":
            return None
        timeout = float(os.environ.get("KB_LLM_TIMEOUT", "3.0"))
        model = _env("KB_LLM_MODEL")
        forced = _env("KB_LLM_PROVIDER").lower()
        anth = _env("ANTHROPIC_API_KEY")
        gem = _env("GEMINI_API_KEY", "GOOGLE_API_KEY")
        oai = _env("OPENAI_API_KEY")
        base = _env("KB_LLM_BASE_URL")

        def make(p, key, b=None, m=None):
            d_base, d_model = DEFAULTS.get(p, (b, m))
            return cls(p, b or d_base, key, model or m or d_model, timeout)

        if forced == "anthropic" and anth:
            return make("anthropic", anth)
        if forced == "gemini" and gem:
            return make("gemini", gem)
        if forced == "openai" and oai:
            return make("openai", oai)
        if base and (forced in ("", "compatible")):
            if not model:
                return None
            return cls("compatible", base, _env("KB_LLM_API_KEY"), model, timeout)
        if forced:
            return None                     # forced provider without its key: stay off rather than guess
        if anth:
            return make("anthropic", anth)
        if gem:
            return make("gemini", gem)
        if oai:
            return make("openai", oai)
        for var, b, m in COMPATIBLE:
            k = _env(var)
            if k:
                return cls("compatible", b, k, model or m, timeout)
        return None

    @property
    def alive(self) -> bool:
        return self.failures < 3 and self.timeouts < 8

    # ---- one call --------------------------------------------------------------------------------------
    def complete(self, system: str, user: str, max_tokens: int = 300) -> str | None:
        if not self.alive:
            return None
        try:
            url, headers, body = getattr(self, "_req_" + ("openai" if self.provider == "compatible" else self.provider))(
                system, user, max_tokens)
            req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=self.timeout, context=self.ctx) as r:
                data = json.loads(r.read().decode())
            text = getattr(self, "_out_" + ("openai" if self.provider == "compatible" else self.provider))(data)
            u = data.get("usage") or data.get("usageMetadata") or {}
            self.usage["calls"] += 1
            self.usage["in"] += int(u.get("input_tokens") or u.get("prompt_tokens") or u.get("promptTokenCount") or 0)
            self.usage["out"] += int(u.get("output_tokens") or u.get("completion_tokens") or u.get("candidatesTokenCount") or 0)
            self.failures = self.timeouts = 0
            return (text or "").strip() or None
        except Exception as e:
            slow = isinstance(e, TimeoutError) or "timed out" in str(e).lower()
            if slow:
                self.timeouts += 1         # one slow call: fall back to the rule answer, keep the step on
            else:
                self.failures += 1
            return None

    def _req_anthropic(self, system, user, max_tokens):
        # Current Claude models reject sampling params (temperature etc.) and think by default. For short extraction
        # answers: low effort, and thinking off where the model allows it (Haiku); answers are cached per question,
        # so repeat consistency does not depend on sampling.
        body = {"model": self.model, "max_tokens": max_tokens, "system": system,
                "messages": [{"role": "user", "content": user}], "output_config": {"effort": "low"}}
        if "haiku" in self.model:
            body["thinking"] = {"type": "disabled"}
        return (f"{self.base}/messages",
                {"x-api-key": self.key, "anthropic-version": "2023-06-01", "content-type": "application/json"}, body)

    @staticmethod
    def _out_anthropic(d):
        return "".join(b.get("text", "") for b in d.get("content", []) if b.get("type") == "text")

    def _req_gemini(self, system, user, max_tokens):
        cfg = {"temperature": 0, "maxOutputTokens": max_tokens}
        if self.model.startswith("gemini-2.5-flash"):
            cfg["thinkingConfig"] = {"thinkingBudget": 0}     # thinking adds seconds; not needed for extraction
        return (f"{self.base}/models/{self.model}:generateContent",
                {"x-goog-api-key": self.key, "content-type": "application/json"},
                {"systemInstruction": {"parts": [{"text": system}]},
                 "contents": [{"role": "user", "parts": [{"text": user}]}], "generationConfig": cfg})

    @staticmethod
    def _out_gemini(d):
        parts = (d.get("candidates") or [{}])[0].get("content", {}).get("parts", [])
        return "".join(p.get("text", "") for p in parts if not p.get("thought"))

    def _req_openai(self, system, user, max_tokens):
        body = {"model": self.model, "messages": [{"role": "system", "content": system},
                                                  {"role": "user", "content": user}]}
        reasoning = self.model.split("/")[-1].startswith(("o1", "o3", "o4", "gpt-5"))
        if not reasoning:
            body["temperature"] = 0
        body["max_completion_tokens" if self.provider == "openai" else "max_tokens"] = max_tokens
        headers = {"content-type": "application/json"}
        if self.key:
            headers["authorization"] = f"Bearer {self.key}"
        return f"{self.base}/chat/completions", headers, body

    @staticmethod
    def _out_openai(d):
        return ((d.get("choices") or [{}])[0].get("message") or {}).get("content") or ""

    def describe(self) -> str:
        return f"{self.provider}:{self.model}"
