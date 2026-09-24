import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from golf_api import (
    AmbiguousTeeError,
    CourseSummary,
    GolfAPIError,
    OpenGolfAPI,
    core_course_name,
    find_tee,
    parse_course,
    parse_search,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


SEARCH_FIXTURE = _load("blue_ash_search.json")
COURSE_FIXTURE = _load("blue_ash_course.json")


# --- parse_search -----------------------------------------------------------


def test_parse_search():
    results = parse_search(SEARCH_FIXTURE)
    assert results == [
        CourseSummary(
            id="283935aa-424d-42c4-8613-8b7b4e739cf1",
            name="Blue Ash Golf Course",
            city="Blue Ash",
            state="OH",
            par=72,
            holes=18,
        )
    ]


def test_parse_search_skips_entries_without_id_or_name():
    payload = {
        "courses": [
            {"id": None, "course_name": "No Id Golf Course"},
            {"course_name": "Missing Id Golf Course"},
            {"id": "no-name", "course_name": None},
            {"id": "blank-name", "course_name": ""},
            *SEARCH_FIXTURE["courses"],
        ]
    }

    assert parse_search(payload) == parse_search(SEARCH_FIXTURE)


# --- parse_course ------------------------------------------------------------


def test_parse_course():
    course = parse_course(COURSE_FIXTURE)

    assert len(course.hole_info) == 18
    assert [h.number for h in course.hole_info] == list(range(1, 19))

    hole1 = course.hole_info[0]
    assert hole1.par == 4
    assert hole1.handicap == 1
    assert hole1.yardages["gold"] == 374

    assert len(course.tees) == 6
    white_genders = {t.gender for t in course.tees if t.name == "White"}
    assert white_genders == {"male", "female"}

    black = next(t for t in course.tees if t.name == "Black")
    assert black.yardage == 6657
    assert black.course_rating == 72.4
    assert black.slope == 133


def test_parse_course_minimal_payload_defaults():
    course = parse_course(
        {"id": "x", "course_name": "Tiny", "tees": None, "holes_data": None}
    )

    assert course.tees == []
    assert course.hole_info == []
    assert course.holes == 18


def test_parse_course_zero_handicap_index_is_none():
    payload = {
        "id": "x",
        "course_name": "Tiny",
        "holes_data": [{"number": 1, "par": 4, "handicap_index": 0, "yardages": {}}],
    }

    course = parse_course(payload)

    assert course.hole_info[0].handicap is None


def test_parse_course_without_a_name_raises_golf_api_error():
    with pytest.raises(GolfAPIError):
        parse_course({"id": "x", "course_name": None})
    with pytest.raises(GolfAPIError):
        parse_course({"id": "x"})


def test_parse_course_without_an_id_raises_golf_api_error():
    with pytest.raises(GolfAPIError):
        parse_course({"course_name": "Tiny"})


def test_parse_course_skips_tees_without_a_name():
    payload = {
        "id": "x",
        "course_name": "Tiny",
        "tees": [
            {"tee_name": None, "gender": "Male", "yardage": 6000},
            {"gender": "Female", "yardage": 5000},
            {"tee_name": "", "gender": "Male"},
            {"tee_name": "Blue", "gender": "Male", "yardage": 6200},
        ],
    }

    course = parse_course(payload)

    assert [tee.name for tee in course.tees] == ["Blue"]


# --- core_course_name ---------------------------------------------------------


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Blue Ash Golf Club", "blue ash"),
        ("The Golf Club at Blue Ash", "blue ash"),
        ("Pebble Beach Golf Links", "pebble beach"),
    ],
)
def test_core_course_name(query, expected):
    assert core_course_name(query) == expected


# --- find_tee ------------------------------------------------------------------


@pytest.fixture
def tees():
    return parse_course(COURSE_FIXTURE).tees


def test_find_tee_by_name(tees):
    tee = find_tee(tees, "gold")
    assert tee.name == "Gold"


def test_find_tee_strips_tees_suffix_and_filters_by_gender(tees):
    tee = find_tee(tees, "White Tees", gender="female")
    assert tee.name == "White"
    assert tee.gender == "female"


def test_find_tee_ambiguous_without_gender(tees):
    with pytest.raises(AmbiguousTeeError):
        find_tee(tees, "white")


def test_find_tee_not_found_lists_available_names(tees):
    with pytest.raises(LookupError) as exc_info:
        find_tee(tees, "purple")

    assert "Black" in str(exc_info.value)


# --- OpenGolfAPI (real local aiohttp server, no network) ----------------------


@asynccontextmanager
async def running_api(app: web.Application):
    server = TestServer(app)
    await server.start_server()
    api = OpenGolfAPI(base_url=str(server.make_url("")).rstrip("/"))
    try:
        yield api
    finally:
        await api.aclose()
        await server.close()


@pytest.mark.asyncio
async def test_search_courses_falls_back_to_core_name():
    calls = []

    async def handler(request):
        calls.append(dict(request.query))
        if request.query.get("q") == "blue ash golf club":
            return web.json_response({"courses": [], "total": 0})
        return web.json_response(SEARCH_FIXTURE)

    app = web.Application()
    app.router.add_get("/courses/search", handler)

    async with running_api(app) as api:
        results = await api.search_courses("blue ash golf club")

    assert [c["q"] for c in calls] == ["blue ash golf club", "blue ash"]
    assert results == parse_search(SEARCH_FIXTURE)


@pytest.mark.asyncio
async def test_search_courses_does_not_repeat_a_query_that_is_already_core():
    calls = []

    async def handler(request):
        calls.append(dict(request.query))
        return web.json_response({"courses": [], "total": 0})

    app = web.Application()
    app.router.add_get("/courses/search", handler)

    async with running_api(app) as api:
        results = await api.search_courses("Blue Ash")

    assert [c["q"] for c in calls] == ["Blue Ash"]
    assert results == []


@pytest.mark.asyncio
async def test_search_courses_timeout_raises_golf_api_error():
    async def handler(request):
        await asyncio.sleep(1)
        return web.json_response(SEARCH_FIXTURE)

    app = web.Application()
    app.router.add_get("/courses/search", handler)

    server = TestServer(app)
    await server.start_server()
    api = OpenGolfAPI(base_url=str(server.make_url("")).rstrip("/"), timeout=0.1)
    try:
        with pytest.raises(GolfAPIError, match="timed out"):
            await api.search_courses("blue ash")
    finally:
        await api.aclose()
        await server.close()


@pytest.mark.asyncio
async def test_search_courses_sends_state_uppercased():
    captured = {}

    async def handler(request):
        captured.update(request.query)
        return web.json_response(SEARCH_FIXTURE)

    app = web.Application()
    app.router.add_get("/courses/search", handler)

    async with running_api(app) as api:
        await api.search_courses("blue ash", state="oh")

    assert captured["state"] == "OH"


@pytest.mark.asyncio
async def test_search_courses_http_error_raises_golf_api_error():
    async def handler(request):
        return web.Response(status=500)

    app = web.Application()
    app.router.add_get("/courses/search", handler)

    async with running_api(app) as api:
        with pytest.raises(GolfAPIError):
            await api.search_courses("blue ash")


@pytest.mark.asyncio
async def test_get_course_success():
    async def handler(request):
        return web.json_response(COURSE_FIXTURE)

    app = web.Application()
    app.router.add_get("/courses/{course_id}", handler)

    async with running_api(app) as api:
        course = await api.get_course("283935aa-424d-42c4-8613-8b7b4e739cf1")

    assert course.name == "Blue Ash Golf Course"
    assert len(course.tees) == 6


@pytest.mark.asyncio
async def test_get_course_404_raises_golf_api_error():
    async def handler(request):
        raise web.HTTPNotFound()

    app = web.Application()
    app.router.add_get("/courses/{course_id}", handler)

    async with running_api(app) as api:
        with pytest.raises(GolfAPIError, match="not found"):
            await api.get_course("unknown-id")


@pytest.mark.asyncio
async def test_aclose_does_not_close_injected_session():
    async with aiohttp.ClientSession() as session:
        api = OpenGolfAPI(http_session=session)
        await api.aclose()
        assert not session.closed
