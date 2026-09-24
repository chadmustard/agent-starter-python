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
# from caddie import CaddieData, RoundSetupAgent
#
#
# def _judge_llm() -> llm.LLM:
#     return inference.LLM(model="openai/gpt-4.1-mini")
#
#
# @pytest.mark.asyncio
# async def test_greets_and_asks_for_course() -> None:
#     """Evaluation of the setup agent's opening greeting."""
#     async with (
#         _judge_llm() as judge_llm,
#         AgentSession[CaddieData](userdata=CaddieData(golf_api=...)) as session,
#     ):
#         result = await session.start(RoundSetupAgent(), capture_run=True)
#
#         # Evaluate the agent's response for a friendly greeting
#         await (
#             result.expect.next_event()
#             .is_message(role="assistant")
#             .judge(
#                 judge_llm,
#                 intent=textwrap.dedent(
#                     """\
#                     Greets the golfer in a friendly way as their caddie and asks
#                     which golf course they played today.
#                     """
#                 ),
#             )
#         )
#
#         # Ensures there are no function calls or other unexpected events
#         result.expect.no_more_events()


def test_agent_module_imports_and_exposes_server() -> None:
    """A cheap smoke test that doesn't need an LLM: importing the entrypoint
    module works and the AgentServer it defines is present."""
    import agent

    assert agent.server is not None
