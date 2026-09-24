"""Tests for RoundSetupAgent. The first group calls the setup tools directly
with a stand-in run context: search result text, course selection, tee
matching, and the handoff to HoleByHoleAgent. The rest are in-process LLM
tests: greeting, searching and confirming the course, asking about tees,
disambiguating a tee shared by men and women, handing off to the hole-by-hole
agent, and handling a course that isn't found.
"""

import dataclasses
import json
from types import SimpleNamespace

import pytest
from livekit.agents import AgentSession, ToolError, inference, llm
from livekit.agents.voice.run_result import AgentHandoffEvent, FunctionCallEvent

from caddie import CaddieData, HoleByHoleAgent, RoundSetupAgent
from golf_api import GolfAPIError

BLUE_ASH_TEE_OPTIONS = (
    "Tee options: Black (6657 yards), Gold (6340 yards), White (5957 yards), "
    "Green (5010 yards)."
)


@pytest.fixture
def setup_data(fake_golf_api, publisher) -> CaddieData:
    """CaddieData as it is when the golfer first joins: nothing set up yet."""
    return CaddieData(golf_api=fake_golf_api, publish=publisher)


def _context(data: CaddieData) -> SimpleNamespace:
    # The setup tools only read `context.userdata`.
    return SimpleNamespace(userdata=data)


# --- setup tools (no LLM) --------------------------------------------------------


@pytest.mark.asyncio
async def test_search_lists_results_and_stores_them(setup_data, fake_golf_api) -> None:
    agent = RoundSetupAgent()
    text = await agent.search_courses(_context(setup_data), "Blue Ash", state="oh")

    assert text.startswith("1. Blue Ash Golf Course, Blue Ash, OH, par 72")
    assert fake_golf_api.calls == [
        ("search_courses", {"query": "Blue Ash", "state": "OH", "limit": 5})
    ]
    assert [c.name for c in setup_data.search_results] == ["Blue Ash Golf Course"]


@pytest.mark.asyncio
async def test_search_with_no_results(setup_data, fake_golf_api) -> None:
    fake_golf_api.return_no_results()
    text = await RoundSetupAgent().search_courses(_context(setup_data), "Zzyzx Links")

    assert text.startswith("No courses matched")
    assert "city or state" in text
    assert setup_data.search_results == []


@pytest.mark.asyncio
async def test_search_when_directory_is_down(setup_data, fake_golf_api) -> None:
    async def fail(*args, **kwargs):
        raise GolfAPIError("timed out")

    fake_golf_api.search_courses = fail
    with pytest.raises(ToolError, match="unavailable"):
        await RoundSetupAgent().search_courses(_context(setup_data), "Blue Ash")


@pytest.mark.asyncio
async def test_select_course_lists_tees_and_pushes_setup(setup_data, publisher) -> None:
    agent = RoundSetupAgent()
    await agent.search_courses(_context(setup_data), "Blue Ash")
    text = await agent.select_course(_context(setup_data), 1)

    assert "Blue Ash Golf Course" in text
    assert BLUE_ASH_TEE_OPTIONS in text
    assert setup_data.course is not None
    assert setup_data.course.name == "Blue Ash Golf Course"
    assert publisher.last["status"] == "setup"
    assert publisher.last["course"]["name"] == "Blue Ash Golf Course"


@pytest.mark.asyncio
async def test_select_course_out_of_range(setup_data) -> None:
    agent = RoundSetupAgent()
    await agent.search_courses(_context(setup_data), "Blue Ash")
    with pytest.raises(ToolError):
        await agent.select_course(_context(setup_data), 2)
    with pytest.raises(ToolError):
        await agent.select_course(_context(setup_data), 0)
    assert setup_data.course is None


@pytest.mark.asyncio
async def test_select_course_without_tee_data(
    setup_data, fake_golf_api, blue_ash_course
) -> None:
    fake_golf_api.course = dataclasses.replace(blue_ash_course, tees=[])
    agent = RoundSetupAgent()
    await agent.search_courses(_context(setup_data), "Blue Ash")
    text = await agent.select_course(_context(setup_data), 1)

    assert "no tee" in text.lower()
    assert "any tee name" in text.lower()


@pytest.mark.asyncio
async def test_start_round_requires_a_course(setup_data) -> None:
    with pytest.raises(ToolError, match=r"(?i)confirm the course"):
        await RoundSetupAgent().start_round(_context(setup_data), "Gold", 18, 1)
    assert setup_data.round is None


@pytest.mark.asyncio
async def test_start_round_ambiguous_tee(setup_data, blue_ash_course) -> None:
    setup_data.course = blue_ash_course
    with pytest.raises(ToolError, match="men's or women's White"):
        await RoundSetupAgent().start_round(_context(setup_data), "white", 18, 1)
    assert setup_data.round is None


@pytest.mark.asyncio
async def test_start_round_unknown_tee(setup_data, blue_ash_course) -> None:
    setup_data.course = blue_ash_course
    with pytest.raises(ToolError, match="Black, Gold, White, Green"):
        await RoundSetupAgent().start_round(_context(setup_data), "Red", 18, 1)
    assert setup_data.round is None


@pytest.mark.asyncio
async def test_start_round_invalid_round(setup_data, blue_ash_course) -> None:
    setup_data.course = blue_ash_course
    with pytest.raises(ToolError, match="nine or eighteen"):
        await RoundSetupAgent().start_round(_context(setup_data), "Gold", 12, 1)
    assert setup_data.round is None


@pytest.mark.asyncio
async def test_start_round_hands_off(setup_data, blue_ash_course, publisher) -> None:
    setup_data.course = blue_ash_course
    agent, text = await RoundSetupAgent().start_round(
        _context(setup_data), "white tees", 9, 10, tee_gender="female"
    )

    assert isinstance(agent, HoleByHoleAgent)
    assert text.startswith("Round set up")
    round_ = setup_data.round
    assert round_ is not None
    assert round_.tee is not None and round_.tee.gender == "female"
    assert [spec.number for spec in round_.holes] == list(range(10, 19))
    assert setup_data.status == "in_progress"
    assert publisher.last["status"] == "in_progress"
    assert publisher.last["tee"]["gender"] == "female"


@pytest.mark.asyncio
async def test_start_round_without_tee_data(
    setup_data, blue_ash_course, publisher
) -> None:
    setup_data.course = dataclasses.replace(blue_ash_course, tees=[])
    agent, _ = await RoundSetupAgent().start_round(_context(setup_data), "Blue", 18, 1)

    assert isinstance(agent, HoleByHoleAgent)
    assert setup_data.round.tee is None
    assert setup_data.round.tee_name == "Blue"
    assert publisher.last["status"] == "in_progress"


# --- LLM behavior ---------------------------------------------------------------


def _judge_llm() -> llm.LLM:
    return inference.LLM(model="openai/gpt-4.1-mini")


def _arguments(fnc_call) -> dict:
    return json.loads(fnc_call.event().item.arguments)


def _calls(result, name: str) -> list[FunctionCallEvent]:
    return [
        event
        for event in result.events
        if isinstance(event, FunctionCallEvent) and event.item.name == name
    ]


def _handoffs(result) -> list[AgentHandoffEvent]:
    return [event for event in result.events if isinstance(event, AgentHandoffEvent)]


async def _start(session: AgentSession) -> None:
    # capture_run consumes the on_enter greeting so it can't leak into the
    # events of the first session.run().
    await session.start(RoundSetupAgent(), capture_run=True)


async def _search_turn(session: AgentSession):
    result = await session.run(user_input="I played Blue Ash today.")
    call = result.expect.contains_function_call(name="search_courses")
    assert "blue ash" in _arguments(call)["course_name"].lower()
    assert not _calls(result, "select_course")
    return result


async def _confirm_turn(session: AgentSession):
    result = await session.run(user_input="Yes, that's the one.")
    result.expect.contains_function_call(
        name="select_course", arguments={"result_number": 1}
    )
    return result


@pytest.mark.asyncio
async def test_greets_and_asks_for_course(setup_data) -> None:
    async with (
        _judge_llm() as judge_llm,
        AgentSession[CaddieData](userdata=setup_data) as session,
    ):
        result = await session.start(RoundSetupAgent(), capture_run=True)

        result.expect.skip_next_event_if(type="agent_handoff")
        await (
            result.expect.next_event()
            .is_message(role="assistant")
            .judge(
                judge_llm,
                intent=(
                    "Greets the golfer in a friendly way as their caddie and asks "
                    "which golf course they played today."
                ),
            )
        )
        result.expect.no_more_events()


@pytest.mark.asyncio
async def test_searches_and_asks_to_confirm_course(setup_data, fake_golf_api) -> None:
    async with (
        _judge_llm() as judge_llm,
        AgentSession[CaddieData](userdata=setup_data) as session,
    ):
        await _start(session)

        result = await _search_turn(session)

        await (
            result.expect[-1]
            .is_message(role="assistant")
            .judge(
                judge_llm,
                intent=(
                    "Asks the golfer to confirm that the course they played is "
                    "Blue Ash Golf Course in Blue Ash, Ohio."
                ),
            )
        )

    assert fake_golf_api.calls[0][0] == "search_courses"
    assert setup_data.course is None


@pytest.mark.asyncio
async def test_confirmed_course_asks_for_tees(setup_data, publisher) -> None:
    async with (
        _judge_llm() as judge_llm,
        AgentSession[CaddieData](userdata=setup_data) as session,
    ):
        await _start(session)
        await _search_turn(session)

        result = await _confirm_turn(session)

        await (
            result.expect[-1]
            .is_message(role="assistant")
            .judge(
                judge_llm,
                intent=(
                    "Asks the golfer which tees they played from, and mentions at "
                    "least some of the tee options, such as Black, Gold, White, "
                    "or Green."
                ),
            )
        )
        assert not _calls(result, "start_round")

    assert setup_data.course is not None
    assert setup_data.course.name == "Blue Ash Golf Course"
    assert publisher.last["status"] == "setup"


@pytest.mark.asyncio
async def test_ambiguous_tee_asks_mens_or_womens(setup_data) -> None:
    async with (
        _judge_llm() as judge_llm,
        AgentSession[CaddieData](userdata=setup_data) as session,
    ):
        await _start(session)
        await _search_turn(session)
        await _confirm_turn(session)

        result = await session.run(
            user_input="I played the white tees, eighteen holes, starting on hole one."
        )

        # Either the round is attempted and rejected for the shared tee name,
        # or the agent asks first. Both are fine; a handoff is not.
        for call in _calls(result, "start_round"):
            output = next(
                event
                for event in result.events
                if event.type == "function_call_output"
                and event.item.call_id == call.item.call_id
            )
            assert output.item.is_error
        assert not _handoffs(result)

        await (
            result.expect[-1]
            .is_message(role="assistant")
            .judge(
                judge_llm,
                intent=(
                    "Asks the golfer whether they played the men's or the women's "
                    "white tees."
                ),
            )
        )

    assert setup_data.round is None
    assert isinstance(session.current_agent, RoundSetupAgent)


@pytest.mark.asyncio
async def test_hands_off_to_hole_by_hole(
    setup_data, blue_ash_course, publisher
) -> None:
    # Simulate an earlier select_course: the course is already confirmed.
    setup_data.course = blue_ash_course
    async with AgentSession[CaddieData](userdata=setup_data) as session:
        await _start(session)

        result = await session.run(user_input="Gold tees, nine holes, started on ten.")

        call = result.expect.contains_function_call(name="start_round")
        args = _arguments(call)
        assert args["tee_name"].lower().removesuffix(" tees") == "gold"
        assert args["holes_played"] == 9
        assert args["starting_hole"] == 10
        result.expect.contains_agent_handoff(new_agent_type=HoleByHoleAgent)

    assert setup_data.round is not None
    assert setup_data.round.holes[0].number == 10
    assert setup_data.status == "in_progress"
    assert publisher.last["status"] == "in_progress"


@pytest.mark.asyncio
async def test_course_not_found(setup_data, fake_golf_api) -> None:
    fake_golf_api.return_no_results()
    async with (
        _judge_llm() as judge_llm,
        AgentSession[CaddieData](userdata=setup_data) as session,
    ):
        await _start(session)

        result = await session.run(user_input="I played Zzyzx Links.")

        result.expect.contains_function_call(name="search_courses")
        assert not _calls(result, "select_course")
        await (
            result.expect[-1]
            .is_message(role="assistant")
            .judge(
                judge_llm,
                intent=(
                    "Tells the golfer it couldn't find that course and asks for "
                    "the city or state, or a different spelling of the name. It "
                    "does not name or suggest any other specific golf course."
                ),
            )
        )

    assert setup_data.course is None
