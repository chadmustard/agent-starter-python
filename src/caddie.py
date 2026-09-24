"""The golf caddie agents and the session state they share.

`CaddieData` is the session userdata. `RoundSetupAgent` finds the course and
collects the tee box, the number of holes, and the starting hole, then hands
off to `HoleByHoleAgent`, which walks the golfer through their round one hole
at a time and records each hole on the scorecard. Every change is pushed to
the frontend through `CaddieData.push()`.
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

from golf_api import (
    AmbiguousTeeError,
    CourseDetail,
    CourseSummary,
    GolfAPIError,
    Tee,
    find_tee,
)
from scorecard import (
    Round,
    RoundStatus,
    ScorecardError,
    build_payload,
    score_name,
)

logger = logging.getLogger("caddie")

# A Large Language Model (LLM) is your agent's brain, processing user input and
# generating a response. See all available models at
# https://docs.livekit.io/agents/models/llm/
#
# To use a realtime model instead of a voice pipeline, replace the `llm=`
# argument on each Agent below with a realtime model and remove the STT/TTS
# from the AgentSession in agent.py. (Note: This is for OpenAI GPT-Live, the
# recommended speech-to-speech model. For other providers, see
# https://docs.livekit.io/agents/models/realtime/)
# 1. Install livekit-agents[openai]
# 2. Set OPENAI_API_KEY in .env.local
# 3. Add `from livekit.plugins import openai` to the top of this file
# 4. Replace the llm argument with:
#    llm=openai.realtime.GPTLiveModel(voice="marin"),
AGENT_LLM_MODEL = "google/gemma-4-31b-it"

MAX_SEARCH_RESULTS = 5

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
    for label, key in (("Front nine", "front_nine"), ("Back nine", "back_nine")):
        nine = summary[key]
        # A nine outside the round is None; one with nothing recorded yet has
        # zero strokes and would only confuse the summary.
        if nine and nine["strokes"]:
            text += (
                f" {label}: {nine['strokes']} strokes, par {nine['par']}, "
                f"{nine['putts']} putts."
            )
    missing = round_.missing_holes()
    if missing:
        text += " Missing holes: " + ", ".join(str(n) for n in missing) + "."
    else:
        text += " No holes are missing."
    return text


def record_hole_result(
    round_: Round,
    hole_number: int,
    *,
    strokes: int,
    putts: int,
    green: str,
    fairway: str | None = None,
    par: int | None = None,
) -> str:
    """Record one hole on the round and return the text the LLM sees.

    A `par` equal to the hole's known par is treated as not passed. A `par`
    that differs from a known par is applied (the golfer corrected it) and
    announced at the start of the result, so a wrong par the LLM volunteers
    is heard rather than silently changing the score. Raises ScorecardError
    without changing anything when the input is invalid.
    """
    known_par = round_.hole(hole_number).par
    if par is not None and par == known_par:
        par = None

    score = round_.record_hole(
        hole_number,
        strokes=strokes,
        putts=putts,
        green=green,
        fairway=fairway,
        par=par,
    )
    spec = round_.hole(hole_number)

    text = (
        f"Recorded hole {hole_number}: {score_name(score.strokes, spec.par)} "
        f"({score.strokes}), {score.putts} putts."
    )
    if par is not None and known_par is not None:
        text = f"Par for hole {hole_number} changed from {known_par} to {par}. {text}"

    next_spec = round_.next_hole()
    if next_spec is not None:
        # The trailing directive keeps the LLM from jumping back to an earlier
        # hole that was recorded before this conversation began.
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
            par: The hole's par. Only pass this when the golfer states the par themselves or the course has no par for this hole; otherwise leave it out.
        """
        round_ = _require_round(context)
        try:
            text = record_hole_result(
                round_,
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
        logger.info(
            "recorded hole %s: %s strokes, %s putts", hole_number, strokes, putts
        )
        return text

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


def _search_result_line(number: int, course: CourseSummary) -> str:
    parts = [course.name]
    parts += [part for part in (course.city, course.state) if part]
    if course.par is not None:
        parts.append(f"par {course.par}")
    return f"{number}. " + ", ".join(parts)


def _unique_tees(tees: list[Tee]) -> list[Tee]:
    """One tee per name, in course order (a name can appear once per gender)."""
    seen: dict[str, Tee] = {}
    for tee in tees:
        seen.setdefault(tee.name.lower(), tee)
    return list(seen.values())


def _tee_names(tees: list[Tee]) -> str:
    return ", ".join(tee.name for tee in _unique_tees(tees))


def _tee_options(course: CourseDetail) -> str:
    if not course.tees:
        return "This course has no tee data, so any tee name the golfer gives is fine."
    options = ", ".join(
        f"{tee.name} ({tee.yardage} yards)" if tee.yardage is not None else tee.name
        for tee in _unique_tees(course.tees)
    )
    return f"Tee options: {options}."


def _spoken_tee_name(name: str) -> str:
    """The tee name without a trailing "tees" or "tee": "white tees" -> "white"."""
    name = name.strip()
    for suffix in (" tees", " tee"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)].strip()
    return name


def _match_tee(course: CourseDetail, tee_name: str, gender: str | None) -> Tee:
    """Find the golfer's tee on the course, raising ToolError with a message
    the LLM can act on when the name is unknown or shared by men and women.
    """
    try:
        return find_tee(course.tees, tee_name, gender)
    except AmbiguousTeeError as e:
        spoken = _spoken_tee_name(tee_name)
        name = next(
            (tee.name for tee in course.tees if tee.name.lower() == spoken.lower()),
            spoken,
        )
        raise ToolError(
            f"The {name} tees are listed for both men and women. Ask the golfer "
            f"whether they played the men's or women's {name} tees."
        ) from e
    except LookupError as e:
        if gender is not None:
            # The name exists but not for that gender. With only one tee of
            # that name, the gender doesn't change which tee it is.
            try:
                return find_tee(course.tees, tee_name)
            except LookupError:
                pass
        raise ToolError(
            f"This course has no {_spoken_tee_name(tee_name)} tees. The tee "
            f"options are: {_tee_names(course.tees)}. Ask the golfer which one "
            "they played."
        ) from e


class RoundSetupAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            llm=inference.LLM(model=AGENT_LLM_MODEL),
            instructions=textwrap.dedent(
                """\
                You are a friendly, upbeat golf caddie helping a golfer fill out their scorecard after their round. First you set up the scorecard.

                {output_rules}
                # Setting up the scorecard

                You need four things: the course, the tees they played, whether they played nine or eighteen holes, and the hole they started on.

                - Ask for one thing at a time. If the golfer gives several answers at once, use all of them and only ask for what is still missing.
                - As soon as the golfer names a course, search for it.
                - If the search finds one course, ask the golfer to confirm it, saying its name and city. If it finds several, read back up to three by name and city and ask which one they played. Do not select a course until the golfer confirms it.
                - If the search finds nothing, tell the golfer you couldn't find it and ask for the city or state, or a different spelling, then search again.
                - Never make up a course or suggest one the search did not return.
                - Once the golfer confirms the course, select it right away, then ask which tees they played, naming the tee options.
                - Never assume the starting hole. Ask for it unless the golfer said it. "Started on the first tee" or "started on one" means hole one.
                - Never assume whether the golfer played the men's or women's tees. If a tee name is listed for both, ask which one they played.
                - As soon as the course is selected and you know the tees, the number of holes, and the starting hole, start the round right away, before saying anything. Don't ask about any hole's score until the round is started.
                """
            ).format(output_rules=_VOICE_OUTPUT_RULES),
        )

    async def on_enter(self) -> None:
        course = self.session.userdata.course
        if course is None:
            self.session.generate_reply(
                instructions=(
                    "Greet the golfer warmly as their caddie, then ask which golf "
                    "course they played today."
                )
            )
            return

        # The course was selected before this agent took over, so its search
        # and confirmation aren't in the chat history. Tell the LLM.
        await self.update_instructions(
            f"{self.instructions}\n# Current setup\n\n"
            f"The golfer's course is already confirmed and selected: {course.name}. "
            f"{_tee_options(course)}\n"
        )
        self.session.generate_reply(
            instructions=(
                f"Greet the golfer warmly as their caddie, mention {course.name}, "
                "and ask which tees they played."
            )
        )

    @function_tool
    async def search_courses(
        self,
        context: RunContext[CaddieData],
        course_name: str,
        state: str | None = None,
    ) -> str:
        """Search the course directory for the course the golfer played.

        Args:
            course_name: The course name as the golfer said it, for example "Blue Ash".
            state: Two-letter US state code, for example "OH". Pass it only if the golfer mentioned a state, or a city whose state is clear; otherwise leave it out.
        """
        if state is not None:
            state = state.strip().upper()
            if len(state) != 2 or not state.isalpha():
                state = None

        async def search(state: str | None) -> list[CourseSummary]:
            try:
                results = await context.userdata.golf_api.search_courses(
                    course_name, state=state, limit=MAX_SEARCH_RESULTS
                )
            except GolfAPIError as e:
                logger.warning("course search failed: %s", e)
                raise ToolError(
                    "The course directory is unavailable right now. Tell the "
                    "golfer and ask them to try again in a moment."
                ) from e
            logger.info(
                "course search %r (%s): %d results", course_name, state, len(results)
            )
            return results[:MAX_SEARCH_RESULTS]

        results = await search(state)
        # The golfer or the LLM may have the state wrong (a course near a state
        # line, a misheard city), so try once more without it.
        widened = False
        if not results and state is not None:
            results = await search(None)
            widened = bool(results)

        context.userdata.search_results = results

        if not results:
            return (
                f"No courses matched {course_name!r}; ask for the city or "
                "state or another spelling."
            )

        lines = [_search_result_line(i, course) for i, course in enumerate(results, 1)]
        if widened:
            lines.insert(0, f"Nothing matched in {state}, but these did elsewhere:")
        if len(results) == 1:
            lines.append(
                "Ask the golfer to confirm this is their course before selecting it."
            )
        else:
            lines.append(
                "Read back up to three of these by name and city and ask which one "
                "the golfer played."
            )
        return "\n".join(lines)

    @function_tool
    async def select_course(
        self, context: RunContext[CaddieData], result_number: int
    ) -> str:
        """Select the golfer's course. Call this only after the golfer confirms which search result is their course.

        Args:
            result_number: The 1-based number of the course in the latest search results.
        """
        results = context.userdata.search_results
        if not results:
            raise ToolError("Search for the course before selecting it.")
        if not (1 <= result_number <= len(results)):
            raise ToolError(
                f"There is no result number {result_number}. The latest search has "
                f"results one through {len(results)}."
            )

        try:
            course = await context.userdata.golf_api.get_course(
                results[result_number - 1].id
            )
        except GolfAPIError as e:
            logger.warning("course lookup failed: %s", e)
            raise ToolError(
                "The course directory is unavailable right now. Tell the golfer "
                "and ask them to try again in a moment."
            ) from e

        context.userdata.course = course
        context.userdata.status = "setup"
        await context.userdata.push()
        logger.info("selected course %s", course.name)
        return (
            f"Selected {course.name}. {_tee_options(course)} "
            "Ask the golfer which tees they played."
        )

    @function_tool
    async def start_round(
        self,
        context: RunContext[CaddieData],
        tee_name: str,
        holes_played: int,
        starting_hole: int,
        tee_gender: Literal["male", "female"] | None = None,
    ) -> tuple[Agent, str]:
        """Set up the scorecard and start going through the round. Call this once the course is selected and you know the tees, the number of holes, and the starting hole.

        Args:
            tee_name: The name of the tees the golfer played, for example "Gold".
            holes_played: 9 or 18.
            starting_hole: The hole the golfer started on, for example 1 or 10.
            tee_gender: "male" or "female". Only pass this when the golfer said they played the men's or women's tees; never guess.
        """
        course = context.userdata.course
        if course is None:
            raise ToolError(
                "No course is selected. Confirm the course with the golfer first."
            )

        if course.tees:
            tee: Tee | None = _match_tee(course, tee_name, tee_gender)
            name = tee.name
        else:
            tee = None
            name = _spoken_tee_name(tee_name)

        try:
            round_ = Round.create(course, name, tee, holes_played, starting_hole)
        except ScorecardError as e:
            raise ToolError(str(e)) from e

        context.userdata.round = round_
        context.userdata.status = "in_progress"
        await context.userdata.push()
        logger.info(
            "round set up: %s holes at %s from %s, starting on %s",
            holes_played,
            course.name,
            name,
            starting_hole,
        )
        # The hole-by-hole agent starts with a fresh chat history; the round
        # context it needs is in its instructions. It asks about the first hole
        # itself, so this agent's last reply only acknowledges the setup.
        return (
            HoleByHoleAgent(round_),
            f"Round set up: {holes_played} holes at {course.name} from the {name} "
            f"tees, starting on hole {starting_hole}. Acknowledge it in a few words, "
            f'for example "Got it, the {name} tees." Don\'t ask about any hole; '
            "that comes next.",
        )
