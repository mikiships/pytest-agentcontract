"""Tests for SDK interceptors."""

from __future__ import annotations

import asyncio

from agentcontract.recorder.core import Recorder
from agentcontract.recorder.interceptors import patch_anthropic, patch_openai


class _Container:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class _OpenAICompletions:
    def __init__(self, response):
        self._response = response

    def create(self, *args, **kwargs):  # noqa: ANN002, ANN003, ARG002
        return self._response


class _AnthropicMessages:
    def __init__(self, response):
        self._response = response

    def create(self, *args, **kwargs):  # noqa: ANN002, ANN003, ARG002
        return self._response


class _AsyncOpenAICompletions:
    def __init__(self, response):
        self._response = response

    async def create(self, *args, **kwargs):  # noqa: ANN002, ANN003, ARG002
        return self._response


class _AsyncAnthropicMessages:
    def __init__(self, response):
        self._response = response

    async def create(self, *args, **kwargs):  # noqa: ANN002, ANN003, ARG002
        return self._response


def test_patch_openai_handles_missing_usage_and_optional_fields():
    response = _Container(
        choices=[_Container(message=_Container(content="Hello from assistant"))],
        model="gpt-test",
    )
    client = _Container(chat=_Container(completions=_OpenAICompletions(response)))
    original_create = client.chat.completions.create

    recorder = Recorder(scenario="openai-no-usage")
    with recorder.recording():
        unpatch = patch_openai(client, recorder)
        client.chat.completions.create(model="unused", messages=[])
        unpatch()

    assert client.chat.completions.create == original_create
    assert recorder.run.model.provider == "openai"
    assert recorder.run.model.model == "gpt-test"
    assert len(recorder.run.turns) == 1
    assert recorder.run.turns[0].content == "Hello from assistant"
    assert recorder.run.turns[0].tokens is None
    assert recorder.run.turns[0].tool_calls == []


def test_patch_openai_handles_dict_like_responses():
    response = {
        "choices": [
            {
                "message": {
                    "content": "Need to call a tool",
                    "tool_calls": [
                        {
                            "id": "tc1",
                            "function": {"name": "lookup_order", "arguments": '{"order_id":"123"}'},
                        }
                    ],
                }
            }
        ],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7},
        "model": "gpt-dict",
    }
    client = _Container(chat=_Container(completions=_OpenAICompletions(response)))

    recorder = Recorder(scenario="openai-dict")
    with recorder.recording():
        unpatch = patch_openai(client, recorder)
        client.chat.completions.create(model="unused", messages=[])
        unpatch()

    turn = recorder.run.turns[0]
    assert turn.content == "Need to call a tool"
    assert turn.tokens is not None
    assert turn.tokens.prompt == 11
    assert turn.tokens.completion == 7
    assert turn.tool_calls[0].function == "lookup_order"
    assert turn.tool_calls[0].arguments == {"order_id": "123"}


def test_patch_openai_skips_unexpected_response_shape():
    response = {"model": "gpt-stream-like"}
    client = _Container(chat=_Container(completions=_OpenAICompletions(response)))

    recorder = Recorder(scenario="openai-unexpected-shape")
    with recorder.recording():
        unpatch = patch_openai(client, recorder)
        client.chat.completions.create(model="unused", messages=[])
        unpatch()

    assert recorder.run.model.provider == "openai"
    assert recorder.run.model.model == "gpt-stream-like"
    assert recorder.run.turns == []


def test_patch_openai_ignores_malformed_tool_call_entries():
    response = {
        "choices": [
            {
                "message": {
                    "content": "Done",
                    "tool_calls": [{"id": "bad", "function": None}],
                }
            }
        ],
    }
    client = _Container(chat=_Container(completions=_OpenAICompletions(response)))

    recorder = Recorder(scenario="openai-malformed-tool-calls")
    with recorder.recording():
        unpatch = patch_openai(client, recorder)
        client.chat.completions.create(model="unused", messages=[])
        unpatch()

    assert len(recorder.run.turns) == 1
    assert recorder.run.turns[0].tool_calls == []


def test_patch_anthropic_handles_dict_like_response_without_usage():
    response = {
        "content": [
            {"type": "text", "text": "Checking now. "},
            {"type": "text", "text": "Done."},
            {"type": "tool_use", "id": "tc1", "name": "lookup_order", "input": {"order_id": "123"}},
        ],
        "model": "claude-test",
    }
    client = _Container(messages=_AnthropicMessages(response))

    recorder = Recorder(scenario="anthropic-dict")
    with recorder.recording():
        unpatch = patch_anthropic(client, recorder)
        client.messages.create(model="unused", messages=[])
        unpatch()

    assert recorder.run.model.provider == "anthropic"
    assert recorder.run.model.model == "claude-test"
    turn = recorder.run.turns[0]
    assert turn.content == "Checking now. Done."
    assert turn.tokens is None
    assert turn.tool_calls[0].function == "lookup_order"
    assert turn.tool_calls[0].arguments == {"order_id": "123"}


def test_patch_anthropic_skips_unexpected_response_shape():
    response = {"content": {"type": "text", "text": "not-a-list"}, "model": "claude-stream-like"}
    client = _Container(messages=_AnthropicMessages(response))

    recorder = Recorder(scenario="anthropic-unexpected-shape")
    with recorder.recording():
        unpatch = patch_anthropic(client, recorder)
        client.messages.create(model="unused", messages=[])
        unpatch()

    assert recorder.run.model.provider == "anthropic"
    assert recorder.run.model.model == "claude-stream-like"
    assert recorder.run.turns == []


def test_patch_openai_handles_non_string_tool_arguments():
    response = {
        "choices": [
            {
                "message": {
                    "content": "Calling tool",
                    "tool_calls": [
                        {
                            "id": "tc1",
                            "function": {"name": "lookup_order", "arguments": {"order_id": "123"}},
                        }
                    ],
                }
            }
        ],
    }
    client = _Container(chat=_Container(completions=_OpenAICompletions(response)))

    recorder = Recorder(scenario="openai-tool-args-dict")
    with recorder.recording():
        unpatch = patch_openai(client, recorder)
        client.chat.completions.create(model="unused", messages=[])
        unpatch()

    turn = recorder.run.turns[0]
    assert turn.tool_calls[0].function == "lookup_order"
    assert turn.tool_calls[0].arguments == {"order_id": "123"}


def test_patch_anthropic_coerces_null_tool_input_to_empty_dict():
    response = {
        "content": [
            {"type": "tool_use", "id": "tc1", "name": "lookup_order", "input": None},
        ],
        "model": "claude-test",
    }
    client = _Container(messages=_AnthropicMessages(response))

    recorder = Recorder(scenario="anthropic-null-tool-input")
    with recorder.recording():
        unpatch = patch_anthropic(client, recorder)
        client.messages.create(model="unused", messages=[])
        unpatch()

    turn = recorder.run.turns[0]
    assert turn.tool_calls[0].arguments == {}


def test_patch_openai_handles_non_integer_usage_values():
    response = {
        "choices": [
            {
                "message": {
                    "content": "Calling tool",
                    "tool_calls": [],
                }
            }
        ],
        "usage": {"prompt_tokens": "not-a-number", "completion_tokens": None},
    }
    client = _Container(chat=_Container(completions=_OpenAICompletions(response)))

    recorder = Recorder(scenario="openai-bad-usage")
    with recorder.recording():
        unpatch = patch_openai(client, recorder)
        client.chat.completions.create(model="unused", messages=[])
        unpatch()

    turn = recorder.run.turns[0]
    assert turn.tokens is None


def test_patch_anthropic_handles_non_integer_usage_values():
    response = {
        "content": [{"type": "text", "text": "Done."}],
        "usage": {"input_tokens": "not-a-number", "output_tokens": {}},
    }
    client = _Container(messages=_AnthropicMessages(response))

    recorder = Recorder(scenario="anthropic-bad-usage")
    with recorder.recording():
        unpatch = patch_anthropic(client, recorder)
        client.messages.create(model="unused", messages=[])
        unpatch()

    turn = recorder.run.turns[0]
    assert turn.tokens is None


def test_patch_openai_handles_async_create():
    response = {
        "choices": [{"message": {"content": "Hello from async assistant"}}],
        "usage": {"prompt_tokens": 4, "completion_tokens": 2},
        "model": "gpt-async",
    }
    client = _Container(chat=_Container(completions=_AsyncOpenAICompletions(response)))

    recorder = Recorder(scenario="openai-async")
    with recorder.recording():
        unpatch = patch_openai(client, recorder)
        returned = asyncio.run(client.chat.completions.create(model="unused", messages=[]))
        unpatch()

    assert returned == response
    assert recorder.run.model.provider == "openai"
    assert recorder.run.model.model == "gpt-async"
    turn = recorder.run.turns[0]
    assert turn.content == "Hello from async assistant"
    assert turn.tokens is not None
    assert turn.tokens.prompt == 4
    assert turn.tokens.completion == 2


def test_patch_anthropic_handles_async_create():
    response = {
        "content": [
            {"type": "text", "text": "Async "},
            {"type": "text", "text": "reply"},
        ],
        "usage": {"input_tokens": 3, "output_tokens": 1},
        "model": "claude-async",
    }
    client = _Container(messages=_AsyncAnthropicMessages(response))

    recorder = Recorder(scenario="anthropic-async")
    with recorder.recording():
        unpatch = patch_anthropic(client, recorder)
        returned = asyncio.run(client.messages.create(model="unused", messages=[]))
        unpatch()

    assert returned == response
    assert recorder.run.model.provider == "anthropic"
    assert recorder.run.model.model == "claude-async"
    turn = recorder.run.turns[0]
    assert turn.content == "Async reply"
    assert turn.tokens is not None
    assert turn.tokens.prompt == 3
    assert turn.tokens.completion == 1
