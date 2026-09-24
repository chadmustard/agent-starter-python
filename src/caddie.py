"""The golf caddie agents and the session state they share.

`CaddieData` is the session userdata. `HoleByHoleAgent` walks the golfer
through their round one hole at a time and records each hole on the scorecard.
Every change is pushed to the frontend through `CaddieData.push()`.
"""

import logging
import textwrap
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Literal, Protocol

from livekit.agents import (
    Agent,
    ChatContext,
    RunContext,
    ToolError,
    function_tool,
    inference,
)

from golf_api import CourseDetail, CourseSummary
from scorecard import (
    Round,
    RoundStatus,
    ScorecardError,
    build_payload,
    score_name,
)

logger = logging.getLogger("caddie")

AGENT_LLM_MODEL = "google/gemma-4-31b-it"

ShotResultArg = Literal["hit", "left", "right", "short", "long"]

_VOICE_OUTPUT_RULES = textwrap.dedent(
    """\
    # Output rules

    You are talking to the golfer by voice, so everything you say is read aloud by a text-to-speech system:

    - Respond in plain text only. Never use JSON, markdown, lists, tables, code, emojis, or other formatting.
    - Keep replies brief: one to three sentences. Ask one question at a time.
    - Spell out numbers, for example "hole four" and "a six".
    - Never use the words "tool" or "function", and never recite identifiers, internal reasoning, or raw results. Speak only about the golfer's round.
    """
)


class GolfCourseSource(Protocol):
    async def search_courses(
        self, query: str, state: str | None = None, limit: int = 5
    ) -> list[CourseSummary]: ...

    async def get_course(self, course_id: str) -> CourseDetail: ...


async def _noop_publish(payload: dict) -> None:
    return None


@dataclass
class CaddieData:
    golf_api: GolfCourseSource
    publish: Callable[[dict], Awaitable[None]] = _noop_publish
    search_results: list[CourseSummary] = field(default_factory=list)
    course: CourseDetail | None = None
    round: Round | None = None
    status: RoundStatus = "setup"

    def payload(self) -> dict:
        return build_payload(self.status, self.course, self.round)

    async def push(self) -> None:
        """Send the current scorecard to the frontend. Publishing is best
        effort: a failure is logged and never interrupts the conversation.
        """
        try:
            await self.publish(self.payload())
        except Exception:
            logger.exception("failed to publish the scorecard")


def _format_to_par(diff: int) -> str:
    if diff == 0:
        return "even"
    return f"{diff:+d}"


def _hole_label(round_: Round, number: int) -> str:
    spec = round_.hole(number)
    if spec.par is None:
        return f"hole {number}, par unknown (ask the golfer for the par)"
    label = f"hole {number}, par {spec.par}"
    if spec.yardage is not None:
        label += f", {spec.yardage} yards"
    return label


def _round_context(round_: Round) -> str:
    order = ", ".join(str(spec.number) for spec in round_.holes)
    pars = ", ".join(
        f"{spec.number}: par {spec.par if spec.par is not None else 'unknown'}"
        for spec in round_.holes
    )
    return (
        f"The golfer played {round_.holes_played} holes at {round_.course.name} "
        f"from the {round_.tee_name} tees, starting on hole {round_.starting_hole}. "
        f"Play order: {order}. Hole pars: {pars}."
    )


def _scorecard_text(round_: Round) -> str:
    summary = round_.summary()
    text = (
        f"Holes completed: {summary['holes_completed']} of {round_.holes_played}. "
        f"Total strokes {summary['total_strokes']}, "
        f"{_format_to_par(summary['score_to_par'])} to par. "
        f"Putts {summary['total_putts']}. "
        f"Fairways hit {summary['fairways_hit']} of {summary['fairways_possible']}. "
        f"Greens hit {summary['greens_hit']} of {summary['greens_possible']}."
    )
    missing = round_.missing_holes()
    if missing:
        text += " Missing holes: " + ", ".join(str(n) for n in missing) + "."
    else:
        text += " No holes are missing."
    return text


def _require_round(context: RunContext[CaddieData]) -> Round:
    round_ = context.userdata.round
    if round_ is None:
        raise ToolError("There is no round set up yet.")
    return round_


class HoleByHoleAgent(Agent):
    def __init__(self, round_: Round, *, chat_ctx: ChatContext | None = None) -> None:
        self._round = round_
        super().__init__(
            llm=inference.LLM(model=AGENT_LLM_MODEL),
            chat_ctx=chat_ctx,
            instructions=textwrap.dedent(
                """\
                You are a friendly, upbeat golf caddie helping a golfer fill out their scorecard after their round. The golfer tells you how each hole went and you record it.

                # Round

                {round_context}

                {output_rules}
                # Recording holes

                - Walk through the holes in play order, but accept holes in any order. If the golfer describes several holes at once, record each hole separately.
                - For every hole you need: the number of strokes, the number of putts, and the green result: hit, or missed left, right, short, or long.
                - On par fours and par fives you also need the fairway result: hit, or missed left, right, short, or long. Never ask about the fairway on a par three.
                - Convert golf terms using the hole's par: birdie is par minus one, bogey is par plus one, double bogey is par plus two, and "made par" means strokes equal to par. "Two putted" means two putts, "one putt" means one putt. "Hit the green" or "green in regulation" means the green result is hit.
                - If the golfer leaves out any detail, ask for just the missing details before you record the hole. Never guess or fill in a default.
                - As soon as you have every detail for a hole, record it right away, before saying anything.
                - If the golfer corrects a hole you already recorded, record that hole again with all of its details, keeping the details they did not change.
                - After recording, briefly acknowledge the score in a few words, for example "Bogey on one, got it," then ask about exactly the hole the recording result says is next, naming its number and par. That is the first hole still missing from the scorecard, so trust it even if you haven't discussed the holes before it. Do not read back every stat.
                - When every hole is recorded, tell the golfer their total score and how it compares to par, and ask them to confirm it's right. Only after they confirm, finish the round.
                - If the golfer asks how they are doing, check the scorecard and give a short summary.
                """
            ).format(
                round_context=_round_context(round_),
                output_rules=_VOICE_OUTPUT_RULES,
            ),
        )

    async def on_enter(self) -> None:
        spec = self._round.next_hole() or self._round.holes[0]
        self.session.generate_reply(
            instructions=(
                "Tell the golfer you're ready to go through their round, then ask "
                f"how {_hole_label(self._round, spec.number)} went. Mention the "
                "hole number and its par."
            )
        )

    @function_tool
    async def record_hole(
        self,
        context: RunContext[CaddieData],
        hole_number: int,
        strokes: int,
        putts: int,
        green: ShotResultArg,
        fairway: ShotResultArg | None = None,
        par: int | None = None,
    ) -> str:
        """Record the golfer's result on one hole. Call this once per hole, as soon as you have every detail for it. Calling it again for the same hole replaces that hole, which is how corrections are made.

        Args:
            hole_number: The hole's number on the course, for example 1 for the first hole.
            strokes: Total strokes on the hole, including putts. Convert golf terms using the hole's par, for example a bogey on a par four is 5.
            putts: Number of putts on the hole.
            green: "hit" if the approach finished on the green in regulation, otherwise the side the golfer missed on: "left", "right", "short", or "long".
            fairway: Tee shot result on par fours and par fives: "hit", "left", "right", "short", or "long". Leave this out on par threes.
            par: The hole's par. Only pass this when the par is unknown for this hole or the golfer corrects it; otherwise leave it out.
        """
        round_ = _require_round(context)
        try:
            score = round_.record_hole(
                hole_number,
                strokes=strokes,
                putts=putts,
                green=green,
                fairway=fairway,
                par=par,
            )
        except ScorecardError as e:
            raise ToolError(str(e)) from e

        context.userdata.status = "in_progress"
        await context.userdata.push()

        spec = round_.hole(hole_number)
        logger.info(
            "recorded hole %s: %s strokes, %s putts", hole_number, strokes, putts
        )
        text = (
            f"Recorded hole {hole_number}: {score_name(score.strokes, spec.par)} "
            f"({score.strokes}), {score.putts} putts."
        )
        next_spec = round_.next_hole()
        if next_spec is not None:
            # The trailing directive keeps the LLM from jumping back to an
            # earlier hole that was recorded before this conversation began.
            return (
                f"{text} Next is {_hole_label(round_, next_spec.number)}. "
                f"Ask about hole {next_spec.number} now."
            )

        summary = round_.summary()
        return (
            f"{text} All {round_.holes_played} holes are recorded. "
            f"Total {summary['total_strokes']}, "
            f"{_format_to_par(summary['score_to_par'])} to par. "
            "Read the total back and ask the golfer to confirm."
        )

    @function_tool
    async def get_scorecard(self, context: RunContext[CaddieData]) -> str:
        """Get a short summary of the golfer's scorecard so far: holes completed, total strokes, score to par, putts, fairways and greens hit, and any holes still missing."""
        return _scorecard_text(_require_round(context))

    @function_tool
    async def finish_round(self, context: RunContext[CaddieData]) -> str:
        """Finish the round and finalize the scorecard. Only call this after every hole is recorded and the golfer has confirmed their total score."""
        round_ = _require_round(context)
        missing = round_.missing_holes()
        if missing:
            holes = ", ".join(str(n) for n in missing)
            raise ToolError(
                f"The round isn't finished yet. These holes still need scores: {holes}."
            )

        context.userdata.status = "complete"
        await context.userdata.push()
        logger.info("round complete")
        return (
            f"{_scorecard_text(round_)} "
            "Congratulate the golfer and tell them the scorecard is ready."
        )
