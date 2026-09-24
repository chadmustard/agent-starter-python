"""Shared fakes and fixtures for the caddie agent tests.

Nothing here touches the network: the golf API is faked from the fixture files
and published scorecard payloads are captured in memory.
"""

import json
from pathlib import Path

import pytest
from dotenv import load_dotenv

from caddie import CaddieData
from golf_api import (
    CourseDetail,
    CourseSummary,
    GolfAPIError,
    find_tee,
    parse_course,
    parse_search,
)
from scorecard import Round

# LLM tests call LiveKit Inference with the project credentials.
load_dotenv(Path(__file__).parent.parent / ".env.local")

FIXTURES = Path(__file__).parent / "fixtures"

BLUE_ASH_TEE = "Gold"


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


class FakeGolfAPI:
    """Serves the Blue Ash fixtures and records every call made to it."""

    def __init__(self) -> None:
        self.search_payload = load_fixture("blue_ash_search.json")
        self.course = parse_course(load_fixture("blue_ash_course.json"))
        self.calls: list[tuple[str, dict]] = []

    async def search_courses(
        self, query: str, state: str | None = None, limit: int = 5
    ) -> list[CourseSummary]:
        self.calls.append(
            ("search_courses", {"query": query, "state": state, "limit": limit})
        )
        return parse_search(self.search_payload)[:limit]

    async def get_course(self, course_id: str) -> CourseDetail:
        self.calls.append(("get_course", {"course_id": course_id}))
        if course_id != self.course.id:
            raise GolfAPIError("course not found")
        return self.course


class RecordingPublisher:
    """Async callable that stores every payload it is given."""

    def __init__(self) -> None:
        self.payloads: list[dict] = []

    async def __call__(self, payload: dict) -> None:
        self.payloads.append(payload)

    @property
    def last(self) -> dict | None:
        return self.payloads[-1] if self.payloads else None


@pytest.fixture
def blue_ash_course() -> CourseDetail:
    return parse_course(load_fixture("blue_ash_course.json"))


@pytest.fixture
def fake_golf_api() -> FakeGolfAPI:
    return FakeGolfAPI()


@pytest.fixture
def publisher() -> RecordingPublisher:
    return RecordingPublisher()


def build_caddie_data(
    course: CourseDetail,
    publisher: RecordingPublisher,
    *,
    holes_played: int = 18,
    starting_hole: int = 1,
    golf_api: FakeGolfAPI | None = None,
) -> CaddieData:
    """A CaddieData with a Round already set up for Blue Ash from the Gold tee."""
    tee = find_tee(course.tees, BLUE_ASH_TEE)
    round_ = Round.create(course, BLUE_ASH_TEE, tee, holes_played, starting_hole)
    return CaddieData(
        golf_api=golf_api or FakeGolfAPI(),
        publish=publisher,
        course=course,
        round=round_,
        status="in_progress",
    )


@pytest.fixture
def make_caddie_data(blue_ash_course: CourseDetail, publisher: RecordingPublisher):
    """Factory fixture: make_caddie_data(holes_played=18, starting_hole=1)."""

    def _make(*, holes_played: int = 18, starting_hole: int = 1) -> CaddieData:
        return build_caddie_data(
            blue_ash_course,
            publisher,
            holes_played=holes_played,
            starting_hole=starting_hole,
        )

    return _make
