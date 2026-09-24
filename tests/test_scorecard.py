import dataclasses
import json
from pathlib import Path

import pytest

from golf_api import CourseDetail, Tee, parse_course
from scorecard import (
    MISS_DIRECTIONS,
    HoleScore,
    Round,
    ScorecardError,
    build_payload,
    play_order,
    score_name,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


COURSE_FIXTURE = _load("blue_ash_course.json")


@pytest.fixture
def course() -> CourseDetail:
    return parse_course(COURSE_FIXTURE)


@pytest.fixture
def gold_tee(course: CourseDetail) -> Tee:
    return next(tee for tee in course.tees if tee.name == "Gold")


def _empty_hole_info_course(course: CourseDetail) -> CourseDetail:
    """A course with the same identity but no per-hole data, to exercise the
    unknown-par path.
    """
    return CourseDetail(
        id=course.id,
        name=course.name,
        city=course.city,
        state=course.state,
        par=course.par,
        holes=course.holes,
        tees=course.tees,
        hole_info=[],
    )


# --- play_order --------------------------------------------------------------


def test_play_order_full_round_no_wrap():
    assert play_order(18, 1) == list(range(1, 19))


def test_play_order_wraps_18_from_10():
    assert play_order(18, 10) == [
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
    ]


def test_play_order_9_from_10_no_wrap():
    assert play_order(9, 10) == [10, 11, 12, 13, 14, 15, 16, 17, 18]


def test_play_order_9_from_14_wraps():
    assert play_order(9, 14) == [14, 15, 16, 17, 18, 1, 2, 3, 4]


def test_play_order_rejects_bad_holes_played():
    with pytest.raises(ScorecardError):
        play_order(12, 1)


def test_play_order_rejects_starting_hole_zero():
    with pytest.raises(ScorecardError):
        play_order(18, 0)


def test_play_order_rejects_starting_hole_beyond_course():
    with pytest.raises(ScorecardError):
        play_order(18, 19)


def test_play_order_rejects_18_holes_on_9_hole_course():
    with pytest.raises(ScorecardError):
        play_order(18, 1, course_holes=9)


# --- Round.create --------------------------------------------------------------


def test_round_create_builds_holes_in_play_order(course, gold_tee):
    round_ = Round.create(course, "Gold", gold_tee, 18, 10)

    assert [h.number for h in round_.holes] == [
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
    ]
    first = round_.holes[0]
    assert first.number == 10
    assert first.par == 5
    assert first.yardage == 511
    last = round_.holes[-1]
    assert last.number == 9


def test_round_create_holes_missing_from_course_get_none_fields():
    course = CourseDetail(
        id="x",
        name="Sparse",
        city=None,
        state=None,
        par=None,
        holes=18,
        tees=[],
        hole_info=[],
    )
    round_ = Round.create(course, "White", None, 9, 1)
    for spec in round_.holes:
        assert spec.par is None
        assert spec.yardage is None
        assert spec.handicap is None


# --- record_hole --------------------------------------------------------------


def test_record_hole_happy_path(course, gold_tee):
    round_ = Round.create(course, "Gold", gold_tee, 18, 1)
    score = round_.record_hole(1, strokes=5, putts=2, green="short", fairway="hit")

    assert score == HoleScore(strokes=5, putts=2, fairway="hit", green="short")
    assert round_.scores[1] == score


def test_record_hole_upsert_replaces(course, gold_tee):
    round_ = Round.create(course, "Gold", gold_tee, 18, 1)
    round_.record_hole(1, strokes=5, putts=2, green="short", fairway="hit")
    round_.record_hole(1, strokes=4, putts=1, green="hit", fairway="hit")

    assert round_.scores[1] == HoleScore(strokes=4, putts=1, fairway="hit", green="hit")


def test_record_hole_par_3_drops_fairway(course, gold_tee):
    round_ = Round.create(course, "Gold", gold_tee, 18, 1)
    # hole 2 is a par 3 on this course
    score = round_.record_hole(2, strokes=3, putts=2, green="hit", fairway="hit")
    assert score.fairway is None


def test_record_hole_par_4_without_fairway_errors(course, gold_tee):
    round_ = Round.create(course, "Gold", gold_tee, 18, 1)
    with pytest.raises(ScorecardError):
        round_.record_hole(1, strokes=5, putts=2, green="hit")


def test_record_hole_putts_greater_than_or_equal_strokes_errors(course, gold_tee):
    round_ = Round.create(course, "Gold", gold_tee, 18, 1)
    with pytest.raises(ScorecardError):
        round_.record_hole(1, strokes=4, putts=4, green="hit", fairway="hit")


def test_record_hole_hole_in_one_on_par_3(course, gold_tee):
    round_ = Round.create(course, "Gold", gold_tee, 18, 1)
    score = round_.record_hole(2, strokes=1, putts=0, green="hit")
    assert score.strokes == 1
    assert score.putts == 0
    assert score.fairway is None


def test_record_hole_not_in_round_errors(course, gold_tee):
    round_ = Round.create(course, "Gold", gold_tee, 9, 1)
    with pytest.raises(ScorecardError):
        round_.record_hole(12, strokes=5, putts=2, green="hit", fairway="hit")


def test_record_hole_unknown_par_without_override_errors(course, gold_tee):
    sparse = _empty_hole_info_course(course)
    round_ = Round.create(sparse, "Gold", gold_tee, 18, 1)
    with pytest.raises(ScorecardError):
        round_.record_hole(1, strokes=5, putts=2, green="hit", fairway="hit")


def test_record_hole_unknown_par_with_override_succeeds(course, gold_tee):
    sparse = _empty_hole_info_course(course)
    round_ = Round.create(sparse, "Gold", gold_tee, 18, 1)
    score = round_.record_hole(1, strokes=5, putts=2, green="hit", fairway="hit", par=4)
    assert score.strokes == 5
    assert round_.hole(1).par == 4


def test_record_hole_invalid_strokes_errors(course, gold_tee):
    round_ = Round.create(course, "Gold", gold_tee, 18, 1)
    with pytest.raises(ScorecardError):
        round_.record_hole(1, strokes=21, putts=2, green="hit", fairway="hit")


def test_record_hole_invalid_par_override_errors(course, gold_tee):
    round_ = Round.create(course, "Gold", gold_tee, 18, 1)
    with pytest.raises(ScorecardError):
        round_.record_hole(1, strokes=5, putts=2, green="hit", fairway="hit", par=2)


def test_record_hole_failed_call_with_par_override_leaves_par_unchanged(
    course, gold_tee
):
    round_ = Round.create(course, "Gold", gold_tee, 18, 1)
    with pytest.raises(ScorecardError):
        round_.record_hole(1, strokes=5, putts=9, green="hit", fairway="hit", par=5)
    assert round_.hole(1).par == 4
    assert 1 not in round_.scores


def test_record_hole_failed_call_leaves_existing_score_and_par(course, gold_tee):
    round_ = Round.create(course, "Gold", gold_tee, 18, 1)
    round_.record_hole(1, strokes=5, putts=2, green="short", fairway="hit")
    with pytest.raises(ScorecardError):
        # par 3 override would drop the fairway, but the green is invalid
        round_.record_hole(1, strokes=4, putts=2, green="nowhere", par=3)
    assert round_.hole(1).par == 4
    assert round_.scores[1] == HoleScore(
        strokes=5, putts=2, fairway="hit", green="short"
    )


def test_record_hole_fairway_checked_against_overridden_par(course, gold_tee):
    round_ = Round.create(course, "Gold", gold_tee, 18, 1)
    # hole 2 is a par 3; overriding to par 4 makes the fairway required
    with pytest.raises(ScorecardError):
        round_.record_hole(2, strokes=4, putts=2, green="hit", par=4)
    assert round_.hole(2).par == 3


def test_record_hole_failed_call_on_unknown_par_leaves_par_unknown(course, gold_tee):
    sparse = _empty_hole_info_course(course)
    round_ = Round.create(sparse, "Gold", gold_tee, 18, 1)
    with pytest.raises(ScorecardError):
        round_.record_hole(1, strokes=0, putts=0, green="hit", fairway="hit", par=4)
    assert round_.hole(1).par is None


def test_record_hole_invalid_green_errors(course, gold_tee):
    round_ = Round.create(course, "Gold", gold_tee, 18, 1)
    with pytest.raises(ScorecardError):
        round_.record_hole(1, strokes=5, putts=2, green="somewhere", fairway="hit")


# --- next_hole / missing_holes / is_complete -----------------------------------


def test_progress_tracking(course, gold_tee):
    round_ = Round.create(course, "Gold", gold_tee, 9, 1)
    assert round_.next_hole().number == 1
    assert round_.missing_holes() == list(range(1, 10))
    assert not round_.is_complete

    for number in range(1, 10):
        par = round_.hole(number).par
        round_.record_hole(
            number,
            strokes=par,
            putts=2,
            green="hit",
            fairway=None if par == 3 else "hit",
        )

    assert round_.next_hole() is None
    assert round_.missing_holes() == []
    assert round_.is_complete


# --- summary --------------------------------------------------------------


def test_summary_all_fields(course, gold_tee):
    round_ = Round.create(course, "Gold", gold_tee, 18, 1)
    # hole 1: par 4, fairway hit, green short, 5 strokes, 2 putts
    round_.record_hole(1, strokes=5, putts=2, green="short", fairway="hit")
    # hole 2: par 3, green hit, 3 strokes, 2 putts
    round_.record_hole(2, strokes=3, putts=2, green="hit")
    # hole 5: par 5, fairway left, green hit, 5 strokes, 2 putts
    round_.record_hole(5, strokes=5, putts=2, green="hit", fairway="left")

    summary = round_.summary()

    assert summary["holes_completed"] == 3
    assert summary["total_strokes"] == 13
    assert summary["total_par"] == 12
    assert summary["score_to_par"] == 1
    assert summary["total_putts"] == 6
    assert summary["fairways_possible"] == 2
    assert summary["fairways_hit"] == 1
    assert summary["greens_possible"] == 3
    assert summary["greens_hit"] == 2
    assert summary["fairway_misses"] == {"left": 1, "right": 0, "short": 0, "long": 0}
    assert summary["green_misses"] == {"left": 0, "right": 0, "short": 1, "long": 0}
    # front nine has holes present and some recorded
    assert summary["front_nine"] == {"strokes": 13, "par": 12, "putts": 6}
    # back nine has holes present (this is an 18 hole round) but none recorded
    assert summary["back_nine"] == {"strokes": 0, "par": 0, "putts": 0}


def test_summary_nine_null_when_round_excludes_it(course, gold_tee):
    round_ = Round.create(course, "Gold", gold_tee, 9, 10)
    assert round_.summary()["front_nine"] is None
    round_.record_hole(10, strokes=5, putts=2, green="hit", fairway="hit")
    assert round_.summary()["back_nine"] == {"strokes": 5, "par": 5, "putts": 2}


def test_summary_empty_round():
    course = CourseDetail(
        id="x",
        name="Empty",
        city=None,
        state=None,
        par=None,
        holes=18,
        tees=[],
        hole_info=[],
    )
    round_ = Round.create(course, "White", None, 18, 1)
    summary = round_.summary()
    assert summary["holes_completed"] == 0
    assert summary["total_strokes"] == 0
    assert summary["total_par"] == 0
    assert summary["score_to_par"] == 0
    assert summary["total_putts"] == 0
    assert summary["fairways_possible"] == 0
    assert summary["fairways_hit"] == 0
    assert summary["greens_possible"] == 0
    assert summary["greens_hit"] == 0
    assert summary["fairway_misses"] == dict.fromkeys(MISS_DIRECTIONS, 0)
    assert summary["green_misses"] == dict.fromkeys(MISS_DIRECTIONS, 0)
    assert summary["front_nine"] == {"strokes": 0, "par": 0, "putts": 0}
    assert summary["back_nine"] == {"strokes": 0, "par": 0, "putts": 0}


# --- score_name --------------------------------------------------------------


@pytest.mark.parametrize(
    ("strokes", "par", "expected"),
    [
        (1, 4, "hole in one"),
        (1, 3, "hole in one"),
        (2, 5, "albatross"),
        (3, 5, "eagle"),
        (3, 4, "birdie"),
        (4, 4, "par"),
        (5, 4, "bogey"),
        (6, 4, "double bogey"),
        (7, 4, "triple bogey"),
        (8, 4, "4 over par"),
    ],
)
def test_score_name(strokes, par, expected):
    assert score_name(strokes, par) == expected


# --- build_payload --------------------------------------------------------------


def test_build_payload_setup_no_course():
    payload = build_payload("setup", None, None)
    assert payload["version"] == 1
    assert payload["status"] == "setup"
    assert payload["course"] is None
    assert payload["tee"] is None
    assert payload["holes_played"] is None
    assert payload["starting_hole"] is None
    assert payload["holes"] == []
    assert payload["summary"] is None
    json.dumps(payload)


def test_build_payload_setup_with_course(course):
    payload = build_payload("setup", course, None)
    assert payload["course"] == {
        "id": course.id,
        "name": course.name,
        "city": course.city,
        "state": course.state,
    }
    assert payload["tee"] is None
    assert payload["holes"] == []
    assert payload["summary"] is None
    json.dumps(payload)


def test_build_payload_round_without_tee_data_uses_spoken_tee_name(course):
    sparse = dataclasses.replace(course, tees=[])
    round_ = Round.create(sparse, "Blue", None, 18, 1)

    payload = build_payload("in_progress", sparse, round_)

    assert payload["tee"] == {
        "name": "Blue",
        "gender": None,
        "course_rating": None,
        "slope": None,
        "yardage": None,
    }
    json.dumps(payload)


def test_build_payload_in_progress(course, gold_tee):
    round_ = Round.create(course, "Gold", gold_tee, 9, 1)
    round_.record_hole(1, strokes=5, putts=2, green="hit", fairway="hit")

    payload = build_payload("in_progress", course, round_)

    assert len(payload["holes"]) == 9
    recorded = payload["holes"][0]
    assert recorded["number"] == 1
    assert recorded["strokes"] == 5
    assert recorded["putts"] == 2
    assert recorded["fairway"] == "hit"
    assert recorded["green"] == "hit"
    assert recorded["score_to_par"] == 1

    unrecorded = payload["holes"][1]
    assert unrecorded["number"] == 2
    assert unrecorded["strokes"] is None
    assert unrecorded["putts"] is None
    assert unrecorded["fairway"] is None
    assert unrecorded["green"] is None
    assert unrecorded["score_to_par"] is None

    assert payload["tee"] == {
        "name": gold_tee.name,
        "gender": gold_tee.gender,
        "course_rating": gold_tee.course_rating,
        "slope": gold_tee.slope,
        "yardage": gold_tee.yardage,
    }
    assert payload["holes_played"] == 9
    assert payload["starting_hole"] == 1
    assert payload["summary"] is not None
    json.dumps(payload)
