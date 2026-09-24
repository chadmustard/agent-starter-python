"""Tests for HoleByHoleAgent. The first group checks the record_hole result
text and par handling directly. The rest are in-process LLM tests: collecting
hole-by-hole stats by voice, converting golf terms to strokes, asking for
missing details, handling corrections, and finishing the round.
"""

import json
from types import SimpleNamespace

import pytest
from livekit.agents import AgentSession, inference, llm

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
