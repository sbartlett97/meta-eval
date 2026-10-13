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


class FakeAnthropicBatches(FakeAnthropic):
    """FakeAnthropic plus ``messages.batches``. ``answers`` maps custom_id ->
    a result: a text answer (succeeded), or a result type such as "errored".
    Results come back in reverse order, as a real batch may reorder them."""

    def __init__(self, answers, sync_text="", polls_before_end=1):
        super().__init__(text=sync_text)
        self.batch_requests = []
        self.retrieves = 0
        outer = self

        class _Batches:
            def create(self, requests):
                outer.batch_requests.extend(requests)
                return SimpleNamespace(id="batch-1", processing_status="in_progress")

            def retrieve(self, batch_id):
                outer.retrieves += 1
                status = "ended" if outer.retrieves >= polls_before_end else "in_progress"
                return SimpleNamespace(id=batch_id, processing_status=status)

            def results(self, batch_id):
                out = []
                for req in reversed(outer.batch_requests):
                    answer = answers.get(req["custom_id"])
                    if answer is None or answer in ("errored", "expired", "canceled"):
                        out.append(SimpleNamespace(custom_id=req["custom_id"],
                                                   result=SimpleNamespace(type=answer or "errored")))
                        continue
                    msg = SimpleNamespace(
                        content=[SimpleNamespace(type="text", text=answer)],
                        stop_reason="end_turn", stop_details=None,
                        usage=SimpleNamespace(output_tokens=5),
                    )
                    out.append(SimpleNamespace(custom_id=req["custom_id"],
                                               result=SimpleNamespace(type="succeeded", message=msg)))
                return out

        self.messages.batches = _Batches()


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
