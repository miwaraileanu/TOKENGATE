from __future__ import annotations
import json
from typing import Literal

import httpx


_OAI_BODY = {
    "id": "chatcmpl-mock",
    "object": "chat.completion",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "Mock response"}, "finish_reason": "stop"}],
    "model": "mock",
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}

_ANT_BODY = {
    "id": "msg_mock",
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": "Mock response"}],
    "model": "mock",
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 10, "output_tokens": 5},
}

_OAI_STREAM = (
    'data: {"id":"chatcmpl-mock","object":"chat.completion.chunk",'
    '"choices":[{"index":0,"delta":{"role":"assistant","content":"Mock"},"finish_reason":null}]}\n\n'
    'data: {"id":"chatcmpl-mock","object":"chat.completion.chunk",'
    '"choices":[{"index":0,"delta":{"content":" response"},"finish_reason":"stop"}],'
    '"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}\n\n'
    'data: [DONE]\n\n'
)

_ANT_STREAM = (
    'event: message_start\n'
    'data: {"type":"message_start","message":{"id":"msg_mock","type":"message",'
    '"role":"assistant","content":[],"model":"mock","stop_reason":null,'
    '"usage":{"input_tokens":10,"output_tokens":0}}}\n\n'
    'event: content_block_delta\n'
    'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Mock response"}}\n\n'
    'event: message_delta\n'
    'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}\n\n'
    'event: message_stop\n'
    'data: {"type":"message_stop"}\n\n'
)


class MockTransport(httpx.AsyncBaseTransport):
    def __init__(
        self,
        mode: Literal["normal", "error"] = "normal",
        error_status: int = 500,
    ):
        self.mode = mode
        self.error_status = error_status
        # Records every request body received — useful in tests
        self.requests: list[dict] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(body)

        if self.mode == "error":
            err = json.dumps({"error": {"message": "upstream error", "type": "server_error"}})
            return httpx.Response(
                self.error_status,
                content=err.encode(),
                headers={"content-type": "application/json"},
            )

        is_anthropic = "/v1/messages" in str(request.url)
        is_streaming = body.get("stream", False)

        if is_streaming:
            content = (_ANT_STREAM if is_anthropic else _OAI_STREAM).encode()
            return httpx.Response(200, content=content, headers={"content-type": "text/event-stream"})

        resp = {**(_ANT_BODY if is_anthropic else _OAI_BODY), "model": body.get("model", "mock")}
        return httpx.Response(200, json=resp)
