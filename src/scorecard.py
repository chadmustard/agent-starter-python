"""Pure scorecard model: play order, hole recording and validation, summary
stats, and the JSON payload sent to the frontend. No I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from golf_api import CourseDetail, Tee

ShotResult = Literal["hit", "left", "right", "short", "long"]
SHOT_RESULTS: tuple[str, ...] = ("hit", "left", "right", "short", "long")
MISS_DIRECTIONS: tuple[str, ...] = ("left", "right", "short", "long")
RoundStatus = Literal["setup", "in_progress", "complete"]


class ScorecardError(ValueError):
    """Raised for invalid scorecard input. Messages are short, plain
    sentences meant to be read aloud by the agent.
    """


def play_order(
    holes_played: int, starting_hole: int, course_holes: int = 18
) -> list[int]:
    """holes_played must be 9 or 18 and <= course_holes; starting_hole in
    1..course_holes. Wraps around: (18, 10) -> [10..18, 1..9];
    (9, 10) -> [10..18]; (9, 14) -> [14..18, 1..4].
    """
    if holes_played not in (9, 18):
        raise ScorecardError("A round must be nine or eighteen holes.")
    if holes_played > course_holes:
        raise ScorecardError(f"This course only has {course_holes} holes.")
    if not (1 <= starting_hole <= course_holes):
        raise ScorecardError(
            f"The starting hole must be between one and {course_holes}."
        )
    return [((starting_hole - 1 + i) % course_holes) + 1 for i in range(holes_played)]


@dataclass
class HoleSpec:
    number: int
    par: int | None
    yardage: int | None
    handicap: int | None


@dataclass
class HoleScore:
    strokes: int
    putts: int
    fairway: ShotResult | None  # always None on par 3s
    green: ShotResult


@dataclass
class Round:
    course: CourseDetail
    tee_name: str
    tee: Tee | None  # None when the course has no tee data
    holes_played: int
    starting_hole: int
    holes: list[HoleSpec]  # in play order
    scores: dict[int, HoleScore] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        course: CourseDetail,
        tee_name: str,
        tee: Tee | None,
        holes_played: int,
        starting_hole: int,
    ) -> Round:
        """Build HoleSpecs from play_order(holes_played, starting_hole,
        course.holes) and course.hole_info (par, handicap, yardage for
        tee_name.lower(); None when missing).
        """
        order = play_order(holes_played, starting_hole, course.holes)
        info_by_number = {info.number: info for info in course.hole_info}
        tee_key = tee_name.lower()

        holes = []
        for number in order:
            info = info_by_number.get(number)
            if info is None:
                holes.append(
                    HoleSpec(number=number, par=None, yardage=None, handicap=None)
                )
            else:
                holes.append(
                    HoleSpec(
                        number=number,
                        par=info.par,
                        yardage=info.yardages.get(tee_key),
                        handicap=info.handicap,
                    )
                )

        return cls(
            course=course,
            tee_name=tee_name,
            tee=tee,
            holes_played=holes_played,
            starting_hole=starting_hole,
            holes=holes,
        )

    def hole(self, number: int) -> HoleSpec:
        for spec in self.holes:
            if spec.number == number:
                return spec
        raise ScorecardError(
            f"Hole {number} isn't part of this round. This round covers "
            f"holes {self.holes[0].number} through {self.holes[-1].number}."
        )

    def record_hole(
        self,
        number: int,
        strokes: int,
        putts: int,
        green: str,
        fairway: str | None = None,
        par: int | None = None,
    ) -> HoleScore:
        """Upsert (re-recording a hole replaces it)."""
        spec = self.hole(number)

        if par is not None:
            if not (3 <= par <= 6):
                raise ScorecardError("Par must be between three and six.")
            spec.par = par

        if spec.par is None:
            raise ScorecardError(f"What's the par for hole {number}?")

        if not (1 <= strokes <= 20):
            raise ScorecardError("Strokes must be between one and twenty.")
        if not (0 <= putts <= 10):
            raise ScorecardError("Putts must be between zero and ten.")
        if putts > strokes - 1:
            raise ScorecardError("Putts can't be that high for that many strokes.")

        if green not in SHOT_RESULTS:
            raise ScorecardError(
                "The green result must be hit, left, right, short, or long."
            )

        if spec.par == 3:
            fairway_result = None
        else:
            if fairway not in SHOT_RESULTS:
                raise ScorecardError(
                    "I need a fairway result for that hole: hit, left, right, "
                    "short, or long."
                )
            fairway_result = fairway

        score = HoleScore(
            strokes=strokes, putts=putts, fairway=fairway_result, green=green
        )
        self.scores[number] = score
        return score

    def next_hole(self) -> HoleSpec | None:
        for spec in self.holes:
            if spec.number not in self.scores:
                return spec
        return None

    def missing_holes(self) -> list[int]:
        return [spec.number for spec in self.holes if spec.number not in self.scores]

    @property
    def is_complete(self) -> bool:
        return not self.missing_holes()

    def _nine(self, low: int, high: int) -> dict | None:
        in_range = [spec for spec in self.holes if low <= spec.number <= high]
        if not in_range:
            return None

        recorded = [
            (spec, self.scores[spec.number])
            for spec in in_range
            if spec.number in self.scores
        ]
        return {
            "strokes": sum(score.strokes for _, score in recorded),
            "par": sum(spec.par for spec, _ in recorded),
            "putts": sum(score.putts for _, score in recorded),
        }

    def summary(self) -> dict:
        recorded = [
            (spec, self.scores[spec.number])
            for spec in self.holes
            if spec.number in self.scores
        ]

        total_strokes = 0
        total_par = 0
        total_putts = 0
        fairways_possible = 0
        fairways_hit = 0
        greens_hit = 0
        fairway_misses = dict.fromkeys(MISS_DIRECTIONS, 0)
        green_misses = dict.fromkeys(MISS_DIRECTIONS, 0)

        for spec, score in recorded:
            total_strokes += score.strokes
            total_par += spec.par
            total_putts += score.putts

            if spec.par >= 4:
                fairways_possible += 1
                if score.fairway == "hit":
                    fairways_hit += 1
                elif score.fairway in MISS_DIRECTIONS:
                    fairway_misses[score.fairway] += 1

            if score.green == "hit":
                greens_hit += 1
            elif score.green in MISS_DIRECTIONS:
                green_misses[score.green] += 1

        return {
            "holes_completed": len(recorded),
            "total_strokes": total_strokes,
            "total_par": total_par,
            "score_to_par": total_strokes - total_par,
            "total_putts": total_putts,
            "fairways_hit": fairways_hit,
            "fairways_possible": fairways_possible,
            "greens_hit": greens_hit,
            "greens_possible": len(recorded),
            "fairway_misses": fairway_misses,
            "green_misses": green_misses,
            "front_nine": self._nine(1, 9),
            "back_nine": self._nine(10, 18),
        }


def score_name(strokes: int, par: int) -> str:
    """strokes == 1 -> "hole in one" (checked first). Otherwise by
    diff = strokes - par.
    """
    if strokes == 1:
        return "hole in one"

    diff = strokes - par
    if diff <= -3:
        return "albatross"
    if diff == -2:
        return "eagle"
    if diff == -1:
        return "birdie"
    if diff == 0:
        return "par"
    if diff == 1:
        return "bogey"
    if diff == 2:
        return "double bogey"
    if diff == 3:
        return "triple bogey"
    return f"{diff} over par"


def build_payload(
    status: RoundStatus, course: CourseDetail | None, round_: Round | None
) -> dict:
    """Return the version-1 payload exactly as in the plan's schema."""
    course_payload = None
    if course is not None:
        course_payload = {
            "id": course.id,
            "name": course.name,
            "city": course.city,
            "state": course.state,
        }

    tee_payload = None
    holes_played = None
    starting_hole = None
    holes_payload: list[dict] = []
    summary_payload = None

    if round_ is not None:
        if round_.tee is not None:
            tee_payload = {
                "name": round_.tee.name,
                "gender": round_.tee.gender,
                "course_rating": round_.tee.course_rating,
                "slope": round_.tee.slope,
                "yardage": round_.tee.yardage,
            }

        holes_played = round_.holes_played
        starting_hole = round_.starting_hole

        for spec in round_.holes:
            score = round_.scores.get(spec.number)
            if score is None:
                holes_payload.append(
                    {
                        "number": spec.number,
                        "par": spec.par,
                        "yardage": spec.yardage,
                        "handicap": spec.handicap,
                        "strokes": None,
                        "putts": None,
                        "fairway": None,
                        "green": None,
                        "score_to_par": None,
                    }
                )
            else:
                holes_payload.append(
                    {
                        "number": spec.number,
                        "par": spec.par,
                        "yardage": spec.yardage,
                        "handicap": spec.handicap,
                        "strokes": score.strokes,
                        "putts": score.putts,
                        "fairway": score.fairway,
                        "green": score.green,
                        "score_to_par": score.strokes - spec.par,
                    }
                )

        summary_payload = round_.summary()

    return {
        "version": 1,
        "status": status,
        "course": course_payload,
        "tee": tee_payload,
        "holes_played": holes_played,
        "starting_hole": starting_hole,
        "holes": holes_payload,
        "summary": summary_payload,
    }
