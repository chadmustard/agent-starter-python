"""Tests for ScorecardPublisher against a fake room/local participant: no
network, no real LiveKit room. Covers publishing (JSON on the scorecard
topic, latest stored, send failures swallowed) and the get-scorecard RPC
handler (null before any publish, the latest payload after).
"""

import json

import pytest

from publisher import GET_SCORECARD_RPC, SCORECARD_TOPIC, ScorecardPublisher


class FakeLocalParticipant:
    def __init__(self, *, fail: bool = False) -> None:
        self.sent: list[tuple[str, str]] = []  # (text, topic)
        self.rpc_handlers: dict[str, object] = {}
        self._fail = fail

    async def send_text(self, text, *, topic="", **kwargs):
        if self._fail:
            raise RuntimeError("boom")
        self.sent.append((text, topic))

    def register_rpc_method(self, method_name, handler) -> None:
        self.rpc_handlers[method_name] = handler


class FakeRoom:
    def __init__(self, *, fail_send: bool = False) -> None:
        self.local_participant = FakeLocalParticipant(fail=fail_send)


@pytest.mark.asyncio
async def test_publish_sends_json_on_the_scorecard_topic() -> None:
    room = FakeRoom()
    publisher = ScorecardPublisher(room)
    payload = {"version": 1, "status": "setup"}

    await publisher.publish(payload)

    assert publisher.latest == payload
    assert len(room.local_participant.sent) == 1
    text, topic = room.local_participant.sent[0]
    assert topic == SCORECARD_TOPIC
    assert json.loads(text) == payload


@pytest.mark.asyncio
async def test_publish_stores_latest_even_when_send_fails() -> None:
    room = FakeRoom(fail_send=True)
    publisher = ScorecardPublisher(room)
    payload = {"version": 1, "status": "setup"}

    await publisher.publish(payload)  # must not raise

    assert publisher.latest == payload


@pytest.mark.asyncio
async def test_rpc_handler_returns_null_before_any_publish() -> None:
    room = FakeRoom()
    publisher = ScorecardPublisher(room)
    publisher.register_rpc()

    handler = room.local_participant.rpc_handlers[GET_SCORECARD_RPC]
    assert await handler(None) == "null"


@pytest.mark.asyncio
async def test_rpc_handler_returns_the_latest_payload_after_publish() -> None:
    room = FakeRoom()
    publisher = ScorecardPublisher(room)
    publisher.register_rpc()
    payload = {"version": 1, "status": "in_progress"}

    await publisher.publish(payload)

    handler = room.local_participant.rpc_handlers[GET_SCORECARD_RPC]
    result = await handler(None)
    assert json.loads(result) == payload
