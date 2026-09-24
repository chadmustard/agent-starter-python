"""Async client for the free OpenGolfAPI, plus pure parsing and tee-matching
helpers that the scorecard agents build on.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass

import aiohttp

logger = logging.getLogger("golf_api")

BASE_URL = "https://api.opengolfapi.org/api/v1"

_GENERIC_WORDS = {
    "the",
    "golf",
    "course",
    "club",
    "country",
    "cc",
    "gc",
    "links",
    "at",
    "of",
    "and",
    "&",
}


class GolfAPIError(Exception):
    """Raised for any failure talking to the OpenGolfAPI: HTTP errors, timeouts,
    non-JSON responses, and 404s.
    """


@dataclass(frozen=True)
class CourseSummary:
    id: str
    name: str
    city: str | None
    state: str | None
    par: int | None
    holes: int | None


@dataclass(frozen=True)
class Tee:
    name: str
    gender: str | None
    course_rating: float | None
    slope: int | None
    par: int | None
    yardage: int | None


@dataclass(frozen=True)
class HoleInfo:
    number: int
    par: int | None
    handicap: int | None
    yardages: dict[str, int]


@dataclass(frozen=True)
class CourseDetail:
    id: str
    name: str
    city: str | None
    state: str | None
    par: int | None
    holes: int
    tees: list[Tee]
    hole_info: list[HoleInfo]


def parse_search(payload: dict) -> list[CourseSummary]:
    """Entries with no id or no course name are skipped: they can't be
    selected or read back to the golfer.
    """
    courses = payload.get("courses") or []
    return [
        CourseSummary(
            id=course["id"],
            name=course.get("course_name"),
            city=course.get("city"),
            state=course.get("state"),
            par=course.get("par"),
            holes=course.get("holes"),
        )
        for course in courses
        if course.get("id") and course.get("course_name")
    ]


def _normalize_gender(gender: str | None) -> str | None:
    return gender.lower() if gender else None


def parse_course(payload: dict) -> CourseDetail:
    """Raises GolfAPIError when the course has no id or name. Tees with no
    name are skipped, since the golfer can't pick them.
    """
    if not payload.get("id") or not payload.get("course_name"):
        raise GolfAPIError("OpenGolfAPI returned a course with no id or name")

    tees_data = payload.get("tees") or []
    tees = [
        Tee(
            name=tee.get("tee_name"),
            gender=_normalize_gender(tee.get("gender")),
            course_rating=tee.get("course_rating"),
            slope=tee.get("slope"),
            par=tee.get("par"),
            yardage=tee.get("yardage"),
        )
        for tee in tees_data
        if tee.get("tee_name")
    ]

    holes_data = payload.get("holes_data") or []
    hole_info = [
        HoleInfo(
            number=hole["number"],
            par=hole.get("par"),
            handicap=hole.get("handicap_index") or None,
            yardages={
                name.lower(): yards
                for name, yards in (hole.get("yardages") or {}).items()
            },
        )
        for hole in holes_data
    ]
    hole_info.sort(key=lambda hole: hole.number)

    holes = payload.get("holes")
    if holes is None:
        holes = len(holes_data) if holes_data else 18

    return CourseDetail(
        id=payload["id"],
        name=payload.get("course_name"),
        city=payload.get("city"),
        state=payload.get("state"),
        par=payload.get("par"),
        holes=holes,
        tees=tees,
        hole_info=hole_info,
    )


def core_course_name(query: str) -> str:
    """Lowercase, strip punctuation, drop generic words, and collapse
    whitespace. "The Golf Club at Blue Ash" -> "blue ash".
    """
    stripped = re.sub(r"[^\w\s]", " ", query.lower())
    words = [word for word in stripped.split() if word not in _GENERIC_WORDS]
    return " ".join(words)


class AmbiguousTeeError(LookupError):
    """Raised by find_tee when more than one tee matches and gender was not
    given to disambiguate.
    """


def _strip_tee_suffix(name: str) -> str:
    name = name.strip().lower()
    for suffix in (" tees", " tee"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def find_tee(tees: list[Tee], name: str, gender: str | None = None) -> Tee:
    """Case-insensitive match on Tee.name (also accepting the name with a
    trailing " tees"/" tee" stripped). If gender is given, filter by it.
    """
    target = _strip_tee_suffix(name)
    matches = [tee for tee in tees if tee.name.lower() == target]
    if gender is not None:
        matches = [tee for tee in matches if tee.gender == gender.lower()]

    if not matches:
        available = ", ".join(sorted({tee.name for tee in tees}))
        raise LookupError(f"No tee named {name!r} found. Available tees: {available}")
    if len(matches) > 1:
        available = ", ".join(sorted({tee.name for tee in tees}))
        raise AmbiguousTeeError(
            f"Multiple tees named {name!r} found; specify gender. "
            f"Available tees: {available}"
        )
    return matches[0]


class OpenGolfAPI:
    def __init__(
        self,
        *,
        base_url: str = BASE_URL,
        timeout: float = 8.0,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._session = http_session
        self._owns_session = http_session is None

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self._session

    async def _get(self, path: str, params: dict[str, str] | None = None) -> dict:
        session = self._get_session()
        url = f"{self._base_url}{path}"
        try:
            async with session.get(
                url,
                params=params,
                timeout=aiohttp.ClientTimeout(total=self._timeout),
            ) as response:
                if response.status == 404:
                    raise GolfAPIError("course not found")
                if response.status >= 400:
                    raise GolfAPIError(
                        f"OpenGolfAPI request failed with status {response.status}"
                    )
                try:
                    return await response.json()
                except (aiohttp.ContentTypeError, ValueError) as exc:
                    raise GolfAPIError(
                        "OpenGolfAPI returned a non-JSON response"
                    ) from exc
        # asyncio.TimeoutError, not the builtin: they are only the same class
        # from Python 3.11 on, and this project supports 3.10.
        except asyncio.TimeoutError as exc:
            logger.warning("OpenGolfAPI request to %s timed out", url)
            raise GolfAPIError("OpenGolfAPI request timed out") from exc
        except aiohttp.ClientError as exc:
            logger.warning("OpenGolfAPI request to %s failed: %s", url, exc)
            raise GolfAPIError(f"OpenGolfAPI request failed: {exc}") from exc

    async def search_courses(
        self, query: str, state: str | None = None, limit: int = 5
    ) -> list[CourseSummary]:
        params: dict[str, str] = {"q": query, "limit": str(limit)}
        if state:
            params["state"] = state.upper()

        payload = await self._get("/courses/search", params=params)
        results = parse_search(payload)

        if not results:
            core = core_course_name(query)
            if core and core != query.strip().lower():
                params["q"] = core
                payload = await self._get("/courses/search", params=params)
                results = parse_search(payload)

        return results

    async def get_course(self, course_id: str) -> CourseDetail:
        payload = await self._get(f"/courses/{course_id}")
        return parse_course(payload)

    async def aclose(self) -> None:
        if self._owns_session and self._session is not None:
            await self._session.close()
