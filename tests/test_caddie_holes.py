"""Tests for HoleByHoleAgent. The first group checks the record_hole result
text and par handling directly. The rest are in-process LLM tests: collecting
hole-by-hole stats by voice, converting golf terms to strokes, asking for
missing details, handling corrections, and finishing the round.
"""

import dataclasses
import json
from types import SimpleNamespace

import pytest
from livekit.agents import AgentSession, ToolError, inference, llm
from livekit.agents.llm.utils import build_legacy_openai_schema

from caddie import CaddieData, HoleByHoleAgent, record_hole_result
from scorecard import ScorecardError

# --- record_hole result text (no LLM) -----------------------------------------


def test_record_result_ignores_par_equal_to_known_par(make_caddie_data) -> None:
    data = make_caddie_data()
    text = record_hole_result(
        data.round, 1, strokes=5, putts=2, green="short", fairway="hit", par=4
    )
    assert text.startswith("Recorded hole 1: bogey (5), 2 putts.")
    assert "changed" not in text
    assert data.round.hole(1).par == 4


def test_record_result_announces_a_changed_par(make_caddie_data) -> None:
    data = make_caddie_data()
    text = record_hole_result(
        data.round, 1, strokes=5, putts=2, green="short", fairway="hit", par=5
    )
    assert text.startswith(
        "Par for hole 1 changed from 4 to 5. Recorded hole 1: par (5), 2 putts."
    )
    assert data.round.hole(1).par == 5


def test_record_result_failed_call_keeps_par(make_caddie_data) -> None:
    data = make_caddie_data()
    with pytest.raises(ScorecardError):
        record_hole_result(
            data.round, 1, strokes=5, putts=9, green="short", fairway="hit", par=5
        )
    assert data.round.hole(1).par == 4
    assert 1 not in data.round.scores


def test_record_result_next_hole_and_completion(make_caddie_data) -> None:
    data = make_caddie_data(holes_played=9)
    text = record_hole_result(
        data.round, 1, strokes=4, putts=2, green="hit", fairway="hit"
    )
    assert text.endswith("Next is hole 2, par 3, 175 yards. Ask about hole 2 now.")

    for number in range(2, 9):
        par = data.round.hole(number).par
        record_hole_result(
            data.round, number, strokes=par, putts=2, green="hit", fairway="hit"
        )
    text = record_hole_result(
        data.round, 9, strokes=5, putts=2, green="hit", fairway="hit"
    )
    assert text.endswith(
        "All 9 holes are recorded. Total 37, +1 to par. "
        "Read the total back and ask the golfer to confirm."
    )


def test_record_result_with_unknown_putts(make_caddie_data) -> None:
    data = make_caddie_data()
    text = record_hole_result(
        data.round, 1, strokes=5, putts=None, green="left", fairway="hit"
    )
    assert text.startswith("Recorded hole 1: bogey (5), putts unknown.")
    assert data.round.scores[1].putts is None


def test_record_hole_schema_allows_unknown_details(make_caddie_data) -> None:
    tool = HoleByHoleAgent(make_caddie_data().round).record_hole
    params = build_legacy_openai_schema(tool, internally_tagged=True)["parameters"]
    props = params["properties"]

    # putts has no default: the LLM must pass a number or an explicit null.
    assert "putts" in params["required"]
    assert {"type": "null"} in props["putts"]["anyOf"]
    assert "default" not in props["putts"]
    assert "unknown" in props["green"]["enum"]
    fairway_enum = next(t for t in props["fairway"]["anyOf"] if "enum" in t)["enum"]
    assert "unknown" in fairway_enum


# --- get_scorecard (no LLM) ---------------------------------------------------


def _context(data: CaddieData) -> SimpleNamespace:
    # The tools only read `context.userdata`.
    return SimpleNamespace(userdata=data)


@pytest.mark.asyncio
async def test_scorecard_text_includes_front_and_back_nine(make_caddie_data) -> None:
    data = make_caddie_data(holes_played=18, starting_hole=1)
    data.round.record_hole(1, strokes=5, putts=2, green="short", fairway="hit")
    data.round.record_hole(2, strokes=3, putts=1, green="hit")
    data.round.record_hole(10, strokes=6, putts=3, green="left", fairway="right")

    text = await HoleByHoleAgent(data.round).get_scorecard(_context(data))

    assert "Front nine: 8 strokes, par 7, 3 putts." in text
    assert "Back nine: 6 strokes, par 5, 3 putts." in text


@pytest.mark.asyncio
async def test_scorecard_text_leaves_out_a_nine_with_nothing_recorded(
    make_caddie_data,
) -> None:
    data = make_caddie_data(holes_played=18, starting_hole=1)
    data.round.record_hole(1, strokes=5, putts=2, green="short", fairway="hit")

    text = await HoleByHoleAgent(data.round).get_scorecard(_context(data))

    assert "Front nine: 5 strokes, par 4, 2 putts." in text
    assert "Back nine" not in text


@pytest.mark.asyncio
async def test_record_hole_tool_records_unknown_details(
    make_caddie_data, publisher
) -> None:
    data = make_caddie_data()
    agent = HoleByHoleAgent(data.round)

    await agent.record_hole(_context(data), 1, 5, None, "unknown", fairway="unknown")

    hole = publisher.last["holes"][0]
    assert (hole["strokes"], hole["putts"], hole["green"], hole["fairway"]) == (
        5,
        None,
        None,
        None,
    )


@pytest.mark.asyncio
async def test_record_hole_tool_still_needs_the_fairway(make_caddie_data) -> None:
    data = make_caddie_data()
    agent = HoleByHoleAgent(data.round)

    with pytest.raises(ToolError, match="fairway"):
        await agent.record_hole(_context(data), 1, 5, 2, "hit")
    assert 1 not in data.round.scores


# --- change_round_setup (no LLM) -------------------------------------------------


@pytest.mark.asyncio
async def test_change_starting_hole_keeps_scores_and_refreshes_prompt(
    make_caddie_data, publisher
) -> None:
    data = make_caddie_data(holes_played=18, starting_hole=1)
    data.round.record_hole(1, strokes=5, putts=2, green="short", fairway="hit")
    agent = HoleByHoleAgent(data.round)
    assert "starting on hole 1." in agent.instructions

    text = await agent.change_round_setup(_context(data), starting_hole=10)

    round_ = data.round
    assert [spec.number for spec in round_.holes][:2] == [10, 11]
    assert round_.starting_hole == 10
    assert round_.scores[1].strokes == 5
    assert "starting hole from 1 to 10" in text
    assert "dropped" not in text.lower()
    assert text.endswith("Ask about hole 10 now.")
    assert "starting on hole 10." in agent.instructions
    assert "Play order: 10, 11," in agent.instructions
    assert publisher.last["starting_hole"] == 10
    assert publisher.last["status"] == "in_progress"
    assert publisher.last["summary"]["total_strokes"] == 5


@pytest.mark.asyncio
async def test_change_to_nine_holes_drops_holes_outside_the_round(
    make_caddie_data, publisher
) -> None:
    data = make_caddie_data(holes_played=18, starting_hole=1)
    data.round.record_hole(1, strokes=5, putts=2, green="short", fairway="hit")
    data.round.record_hole(12, strokes=4, putts=2, green="hit", fairway="hit")

    text = await HoleByHoleAgent(data.round).change_round_setup(
        _context(data), holes_played=9
    )

    assert data.round.holes_played == 9
    assert set(data.round.scores) == {1}
    assert "holes played from 18 to 9" in text
    assert "Dropped the scores for hole 12" in text
    assert text.endswith("Ask about hole 2 now.")
    assert len(publisher.last["holes"]) == 9


@pytest.mark.asyncio
async def test_change_tee_updates_tee_and_yardages(make_caddie_data, publisher) -> None:
    data = make_caddie_data(holes_played=18, starting_hole=1)
    data.round.record_hole(1, strokes=5, putts=2, green="short", fairway="hit")

    text = await HoleByHoleAgent(data.round).change_round_setup(
        _context(data), tee_name="white tees", tee_gender="female"
    )

    round_ = data.round
    assert round_.tee_name == "White"
    assert round_.tee is not None and round_.tee.gender == "female"
    assert round_.hole(1).yardage == 355
    assert round_.scores[1].strokes == 5
    assert "tees from Gold to White" in text
    assert publisher.last["tee"]["name"] == "White"
    assert publisher.last["tee"]["gender"] == "female"


@pytest.mark.asyncio
async def test_change_tee_gender_only_keeps_the_tee_name(make_caddie_data) -> None:
    data = make_caddie_data()
    agent = HoleByHoleAgent(data.round)
    await agent.change_round_setup(_context(data), tee_name="White", tee_gender="male")

    await agent.change_round_setup(_context(data), tee_gender="female")

    assert data.round.tee_name == "White"
    assert data.round.tee.gender == "female"


@pytest.mark.asyncio
async def test_change_tee_shared_by_men_and_women_asks(make_caddie_data) -> None:
    data = make_caddie_data()
    before = data.round

    with pytest.raises(ToolError, match="men's or women's White"):
        await HoleByHoleAgent(data.round).change_round_setup(
            _context(data), tee_name="white"
        )
    assert data.round is before


@pytest.mark.asyncio
async def test_change_to_an_unknown_tee_lists_the_options(make_caddie_data) -> None:
    data = make_caddie_data()
    before = data.round

    with pytest.raises(ToolError, match="Black, Gold, White, Green"):
        await HoleByHoleAgent(data.round).change_round_setup(
            _context(data), tee_name="Red"
        )
    assert data.round is before


@pytest.mark.asyncio
async def test_change_to_an_invalid_round_is_rejected(
    make_caddie_data, publisher
) -> None:
    data = make_caddie_data()
    before = data.round
    agent = HoleByHoleAgent(data.round)
    instructions = agent.instructions

    with pytest.raises(ToolError, match="nine or eighteen"):
        await agent.change_round_setup(_context(data), holes_played=12)
    with pytest.raises(ToolError, match="between one and 18"):
        await agent.change_round_setup(_context(data), starting_hole=19)
    assert data.round is before
    assert agent.instructions == instructions
    assert publisher.payloads == []


@pytest.mark.asyncio
async def test_change_with_nothing_to_change(make_caddie_data) -> None:
    data = make_caddie_data()
    with pytest.raises(ToolError):
        await HoleByHoleAgent(data.round).change_round_setup(_context(data))


@pytest.mark.asyncio
async def test_change_keeps_a_par_the_golfer_gave(make_caddie_data) -> None:
    data = make_caddie_data(holes_played=18, starting_hole=1)
    data.round.record_hole(1, strokes=5, putts=2, green="hit", fairway="hit", par=5)

    await HoleByHoleAgent(data.round).change_round_setup(
        _context(data), starting_hole=10
    )

    assert data.round.hole(1).par == 5


@pytest.mark.asyncio
async def test_change_tee_on_a_course_without_tee_data(
    make_caddie_data, blue_ash_course
) -> None:
    data = make_caddie_data()
    course = dataclasses.replace(blue_ash_course, tees=[])
    data.course = course
    data.round = dataclasses.replace(data.round, course=course, tee=None)

    await HoleByHoleAgent(data.round).change_round_setup(
        _context(data), tee_name="Blue tees"
    )

    assert data.round.tee is None
    assert data.round.tee_name == "Blue"


@pytest.mark.asyncio
async def test_prompt_lists_recorded_holes(make_caddie_data) -> None:
    data = make_caddie_data(holes_played=18, starting_hole=1)
    agent = HoleByHoleAgent(data.round)
    assert "No holes are recorded yet." in agent.instructions

    await agent.record_hole(_context(data), 1, 5, 2, "short", fairway="hit")
    await agent.record_hole(_context(data), 3, 4, 2, "hit", fairway="hit")

    assert "Already recorded: 1, 3." in agent.instructions


@pytest.mark.asyncio
async def test_tools_read_the_round_from_userdata(make_caddie_data) -> None:
    data = make_caddie_data(holes_played=18, starting_hole=1)
    agent = HoleByHoleAgent(data.round)
    await agent.change_round_setup(_context(data), holes_played=9)

    text = await agent.record_hole(_context(data), 9, 5, 2, "hit", fairway="hit")

    assert "Ask about hole 1 now." in text
    assert "of 9." in await agent.get_scorecard(_context(data))


# --- LLM behavior ---------------------------------------------------------------


def _judge_llm() -> llm.LLM:
    return inference.LLM(model="openai/gpt-4.1-mini")


def _arguments(fnc_call) -> dict:
    return json.loads(fnc_call.event().item.arguments)


async def _start(session: AgentSession, data: CaddieData) -> None:
    # capture_run consumes the on_enter greeting so it can't leak into the
    # events of the first session.run().
    await session.start(HoleByHoleAgent(data.round), capture_run=True)


@pytest.mark.llm
@pytest.mark.asyncio
async def test_records_hole_with_full_detail(make_caddie_data, publisher) -> None:
    data = make_caddie_data(holes_played=18, starting_hole=1)
    async with (
        _judge_llm() as judge_llm,
        AgentSession[CaddieData](userdata=data) as session,
    ):
        await _start(session, data)

        result = await session.run(
            user_input=(
                "On the first hole I made a five. I hit the fairway, missed the "
                "green short, and two putted."
            )
        )

        result.expect.next_event().is_function_call(
            name="record_hole",
            arguments={
                "hole_number": 1,
                "strokes": 5,
                "putts": 2,
                "fairway": "hit",
                "green": "short",
            },
        )
        result.expect.next_event().is_function_call_output(is_error=False)
        await (
            result.expect.next_event()
            .is_message(role="assistant")
            .judge(
                judge_llm,
                intent=(
                    "Briefly acknowledges the score on hole one (a five, or a "
                    "bogey) and asks about hole two."
                ),
            )
        )
        result.expect.no_more_events()

    assert publisher.last is not None
    hole_one = next(h for h in publisher.last["holes"] if h["number"] == 1)
    assert hole_one["strokes"] == 5
    assert publisher.last["status"] == "in_progress"


@pytest.mark.llm
@pytest.mark.asyncio
async def test_converts_golf_terms_and_skips_fairway_on_par_three(
    make_caddie_data,
) -> None:
    data = make_caddie_data(holes_played=18, starting_hole=1)
    data.round.record_hole(1, strokes=5, putts=2, green="short", fairway="hit")
    async with (
        _judge_llm() as judge_llm,
        AgentSession[CaddieData](userdata=data) as session,
    ):
        await _start(session, data)

        result = await session.run(
            user_input="Hole two I made par, hit the green and two putted."
        )

        call = result.expect.next_event().is_function_call(
            name="record_hole",
            arguments={"hole_number": 2, "strokes": 3, "putts": 2, "green": "hit"},
        )
        assert _arguments(call).get("fairway") is None
        result.expect.next_event().is_function_call_output(is_error=False)
        await (
            result.expect.next_event()
            .is_message(role="assistant")
            .judge(
                judge_llm,
                intent=(
                    "Acknowledges par on hole two and moves on to hole three. "
                    "Does not ask about the fairway for hole two."
                ),
            )
        )
        result.expect.no_more_events()

    assert data.round.scores[2].strokes == 3
    assert data.round.scores[2].fairway is None
    assert data.round.hole(2).par == 3


@pytest.mark.llm
@pytest.mark.asyncio
async def test_asks_for_missing_details_before_recording(make_caddie_data) -> None:
    data = make_caddie_data(holes_played=18, starting_hole=1)
    async with (
        _judge_llm() as judge_llm,
        AgentSession[CaddieData](userdata=data) as session,
    ):
        await _start(session, data)

        result = await session.run(user_input="I made a bogey on the first hole.")

        await (
            result.expect.next_event()
            .is_message(role="assistant")
            .judge(
                judge_llm,
                intent=(
                    "Asks the golfer for at least one missing detail about hole "
                    "one, such as the fairway, the green, or the number of putts, "
                    "instead of assuming it."
                ),
            )
        )
        result.expect.no_more_events()

    assert 1 not in data.round.scores


@pytest.mark.llm
@pytest.mark.asyncio
async def test_records_unknown_putts_without_guessing(
    make_caddie_data, publisher
) -> None:
    data = make_caddie_data(holes_played=18, starting_hole=1)
    async with (
        _judge_llm() as judge_llm,
        AgentSession[CaddieData](userdata=data) as session,
    ):
        await _start(session, data)

        result = await session.run(
            user_input=(
                "Hole one I made a five, hit the fairway and missed the green "
                "left, but I honestly don't remember how many putts, no idea on "
                "putts."
            )
        )

        call = result.expect.next_event().is_function_call(
            name="record_hole",
            arguments={
                "hole_number": 1,
                "strokes": 5,
                "fairway": "hit",
                "green": "left",
            },
        )
        assert _arguments(call).get("putts") is None
        result.expect.next_event().is_function_call_output(is_error=False)
        await (
            result.expect.next_event()
            .is_message(role="assistant")
            .judge(
                judge_llm,
                intent=(
                    "Briefly acknowledges hole one and asks about hole two. Does "
                    "not ask about the putts on hole one again."
                ),
            )
        )
        result.expect.no_more_events()

    assert data.round.scores[1].strokes == 5
    assert data.round.scores[1].putts is None
    hole_one = publisher.last["holes"][0]
    assert hole_one["strokes"] == 5
    assert hole_one["putts"] is None


@pytest.mark.llm
@pytest.mark.asyncio
async def test_changes_the_starting_hole_after_setup(
    make_caddie_data, publisher
) -> None:
    data = make_caddie_data(holes_played=18, starting_hole=1)
    data.round.record_hole(1, strokes=5, putts=2, green="short", fairway="hit")
    async with (
        _judge_llm() as judge_llm,
        AgentSession[CaddieData](userdata=data) as session,
    ):
        await _start(session, data)

        result = await session.run(
            user_input="Wait, sorry, I actually started on the tenth hole."
        )

        call = result.expect.next_event().is_function_call(
            name="change_round_setup", arguments={"starting_hole": 10}
        )
        args = _arguments(call)
        assert args.get("holes_played") in (None, 18)
        assert args.get("tee_name") is None
        result.expect.next_event().is_function_call_output(is_error=False)
        await (
            result.expect.next_event()
            .is_message(role="assistant")
            .judge(
                judge_llm,
                intent=(
                    "Acknowledges that the round started on hole ten and asks "
                    "how hole ten went."
                ),
            )
        )
        result.expect.no_more_events()

    assert data.round.holes[0].number == 10
    assert data.round.scores[1].strokes == 5
    assert publisher.last["starting_hole"] == 10


@pytest.mark.llm
@pytest.mark.asyncio
async def test_corrects_an_earlier_hole(make_caddie_data) -> None:
    data = make_caddie_data(holes_played=18, starting_hole=1)
    async with AgentSession[CaddieData](userdata=data) as session:
        await _start(session, data)

        first = await session.run(
            user_input=(
                "On the first hole I made a five. I hit the fairway, missed the "
                "green short, and two putted."
            )
        )
        first.expect.contains_function_call(
            name="record_hole", arguments={"hole_number": 1, "strokes": 5}
        )

        result = await session.run(
            user_input="Actually, hole one was a six, not a five."
        )

        result.expect.contains_function_call(
            name="record_hole",
            arguments={
                "hole_number": 1,
                "strokes": 6,
                "putts": 2,
                "fairway": "hit",
                "green": "short",
            },
        )

    score = data.round.scores[1]
    assert score.strokes == 6
    assert score.putts == 2
    assert score.fairway == "hit"
    assert score.green == "short"
    assert data.round.hole(1).par == 4


# Holes 1-8 at Blue Ash (pars 4, 3, 4, 3, 5, 4, 4, 5): 36 strokes on a par 32.
_FRONT_EIGHT = {
    1: (5, 2, "short", "hit"),
    2: (3, 2, "hit", None),
    3: (4, 2, "hit", "hit"),
    4: (4, 2, "left", None),
    5: (5, 2, "hit", "right"),
    6: (5, 2, "long", "hit"),
    7: (4, 2, "hit", "hit"),
    8: (6, 2, "short", "left"),
}


@pytest.mark.llm
@pytest.mark.asyncio
async def test_finishes_round_after_confirmation(make_caddie_data, publisher) -> None:
    data = make_caddie_data(holes_played=9, starting_hole=1)
    for number, (strokes, putts, green, fairway) in _FRONT_EIGHT.items():
        data.round.record_hole(
            number, strokes=strokes, putts=putts, green=green, fairway=fairway
        )

    async with (
        _judge_llm() as judge_llm,
        AgentSession[CaddieData](userdata=data) as session,
    ):
        await _start(session, data)

        result = await session.run(
            user_input=(
                "On nine I made a four. I hit the fairway, hit the green, and "
                "two putted."
            )
        )

        result.expect.next_event().is_function_call(
            name="record_hole",
            arguments={
                "hole_number": 9,
                "strokes": 4,
                "putts": 2,
                "fairway": "hit",
                "green": "hit",
            },
        )
        result.expect.next_event().is_function_call_output(is_error=False)
        await (
            result.expect.next_event()
            .is_message(role="assistant")
            .judge(
                judge_llm,
                intent=(
                    "States the round total of forty strokes, four over par, and "
                    "asks the golfer to confirm it."
                ),
            )
        )
        result.expect.no_more_events()
        assert data.status == "in_progress"

        confirm = await session.run(user_input="Yep, that's right.")
        confirm.expect.contains_function_call(name="finish_round")

    assert data.status == "complete"
    assert publisher.last is not None
    assert publisher.last["status"] == "complete"
    assert publisher.last["summary"]["total_strokes"] == 40
