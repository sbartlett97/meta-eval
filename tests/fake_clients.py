"""Network-free stand-ins for the Anthropic and OpenAI SDK clients.

They record each request and return a canned response, so the judges'
request shape and response handling can be tested without an API call.
"""

from types import SimpleNamespace


class FakeAnthropic:
    """``client.messages.create(**kw)`` -> a Message-like object."""

    def __init__(self, text="", stop_reason="end_turn", stop_details=None):
        self.requests = []
        outer = self

        class _Messages:
            def create(self, **kwargs):
                outer.requests.append(kwargs)
                return SimpleNamespace(
                    content=[SimpleNamespace(type="text", text=text)] if text else [],
                    stop_reason=stop_reason,
                    stop_details=stop_details,
                    usage=SimpleNamespace(output_tokens=42),
                )

        self.messages = _Messages()


class FakeOpenAI:
    """``client.chat.completions.create(**kw)`` -> a ChatCompletion-like object."""

    def __init__(self, text="", finish_reason="stop", refusal=None):
        self.requests = []
        outer = self

        class _Completions:
            def create(self, **kwargs):
                outer.requests.append(kwargs)
                message = SimpleNamespace(content=text, refusal=refusal)
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
                    usage=SimpleNamespace(completion_tokens=7),
                )

        self.chat = SimpleNamespace(completions=_Completions())
