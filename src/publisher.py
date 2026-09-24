"""Publishes the golf scorecard to the frontend: a JSON text stream on topic
`golf.scorecard` after every change, plus an RPC method that answers with the
latest payload for a client that joins after the first push.
"""

import json
import logging

from livekit import rtc

logger = logging.getLogger("publisher")

SCORECARD_TOPIC = "golf.scorecard"
GET_SCORECARD_RPC = "golf.get_scorecard"


class ScorecardPublisher:
    def __init__(self, room: rtc.Room) -> None:
        self._room = room
        self.latest: dict | None = None

    async def publish(self, payload: dict) -> None:
        """Store the payload as the latest scorecard, then send it to the
        frontend on the scorecard text-stream topic. Publishing is best
        effort: a failure to send (for example, the room isn't connected
        yet) is logged and swallowed so it never breaks the conversation.
        """
        self.latest = payload
        try:
            await self._room.local_participant.send_text(
                json.dumps(payload), topic=SCORECARD_TOPIC
            )
        except Exception:
            logger.exception("failed to publish the scorecard")

    def register_rpc(self) -> None:
        """Answer golf.get_scorecard with the latest payload, or "null" if
        nothing has been published yet, for a client that joins late.
        """

        async def handler(_data: rtc.RpcInvocationData) -> str:
            return json.dumps(self.latest) if self.latest is not None else "null"

        self._room.local_participant.register_rpc_method(GET_SCORECARD_RPC, handler)
