# Agent behavior is covered by the simulations in scenarios.yaml, which run full
# conversations against the agent on LiveKit Cloud (see README.md). The eval
# below is kept as an example of the in-process testing framework
# (https://docs.livekit.io/agents/start/testing/) for turn-level checks that
# don't need a live session. Uncomment it and run `uv run pytest` to use it.
#
# import textwrap
#
# import pytest
# from livekit.agents import AgentSession, inference, llm
#
# from agent import Assistant
#
#
# def _judge_llm() -> llm.LLM:
#     return inference.LLM(model="openai/gpt-4.1-mini")
#
#
# @pytest.mark.asyncio
# async def test_offers_assistance() -> None:
#     """Evaluation of the agent's friendly nature."""
#     async with (
#         _judge_llm() as judge_llm,
#         AgentSession() as session,
#     ):
#         await session.start(Assistant())
#
#         # Run an agent turn following the user's greeting
#         result = await session.run(user_input="Hello")
#
#         # Evaluate the agent's response for friendliness
#         await (
#             result.expect.next_event()
#             .is_message(role="assistant")
#             .judge(
#                 judge_llm,
#                 intent=textwrap.dedent(
#                     """\
#                     Greets the user in a friendly manner.
#
#                     Optional context that may or may not be included:
#                     - Offer of assistance with any request the user may have
#                     - Other small talk or chit chat is acceptable, so long as it is friendly and not too intrusive
#                     """
#                 ),
#             )
#         )
#
#         # Ensures there are no function calls or other unexpected events
#         result.expect.no_more_events()


import pytest
from livekit.agents import AgentSession, inference, llm

from agent import Assistant


def _judge_llm() -> llm.LLM:
    return inference.LLM(model="openai/gpt-4.1-mini")


@pytest.mark.asyncio
async def test_looks_up_weather() -> None:
    """The weather tool is called with the requested location and the result is relayed."""
    async with (
        _judge_llm() as judge_llm,
        AgentSession() as session,
    ):
        await session.start(Assistant())

        result = await session.run(user_input="What's the weather in Tokyo?")

        result.expect.next_event().is_function_call(
            name="lookup_weather", arguments={"location": "Tokyo"}
        )
        result.expect.next_event().is_function_call_output(
            output="sunny with a temperature of 70 degrees."
        )
        await (
            result.expect.next_event()
            .is_message(role="assistant")
            .judge(
                judge_llm,
                intent="Informs the user that the weather in Tokyo is sunny with a temperature of 70 degrees.",
            )
        )
        result.expect.no_more_events()


@pytest.mark.asyncio
async def test_declines_questions_outside_available_tools() -> None:
    """The agent is a closed system: it must not answer from its own knowledge
    when no tool covers the request, even if it "knows" the answer."""
    async with (
        _judge_llm() as judge_llm,
        AgentSession() as session,
    ):
        await session.start(Assistant())

        result = await session.run(user_input="What's the altitude of Denver?")

        decline = result.expect.next_event().is_message(role="assistant")
        await decline.judge(
            judge_llm,
            intent=(
                "Declines to answer and explains that it isn't able to help "
                "with that. Does not state or guess an actual altitude value."
            ),
        )

        content = " ".join(decline.event().item.content).lower()
        assert "tool" not in content and "function" not in content, (
            f"response leaked an internal implementation term: {content!r}"
        )

        result.expect.no_more_events()
