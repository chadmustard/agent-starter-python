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
# from golf_api import OpenGolfAPI
#
#
# def _judge_llm() -> llm.LLM:
#     return inference.LLM(model="openai/gpt-4.1-mini")
#
#
# @pytest.mark.llm
# @pytest.mark.asyncio
# async def test_greets_and_asks_for_course() -> None:
#     """Evaluation of the setup agent's opening greeting."""
#     golf_api = OpenGolfAPI()
#     async with (
#         _judge_llm() as judge_llm,
#         AgentSession[CaddieData](userdata=CaddieData(golf_api=golf_api)) as session,
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
#
#     await golf_api.aclose()


def test_agent_module_imports_and_exposes_server() -> None:
    """A cheap smoke test that doesn't need an LLM: importing the entrypoint
    module works and the AgentServer it defines is present."""
    import agent

    assert agent.server is not None


def test_entrypoint_registers_rpc_before_the_session_starts() -> None:
    """session.start() connects the room. Register the scorecard RPC before the
    first push so a frontend joining right after can call it.
    """
    import ast
    import inspect

    import agent

    tree = ast.parse(inspect.getsource(agent))
    entrypoint = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "my_agent"
    )

    def first_line(call: str) -> int:
        lines = [
            node.lineno
            for node in ast.walk(entrypoint)
            if isinstance(node, ast.Call) and ast.unparse(node.func) == call
        ]
        assert lines, f"{call}() is not called in my_agent"
        return min(lines)

    start = first_line("session.start")
    register = first_line("publisher.register_rpc")
    push = first_line("session.userdata.push")

    assert start < register < push
