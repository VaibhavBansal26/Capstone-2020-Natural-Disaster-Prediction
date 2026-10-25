"""Geocoding and nearby emergency services from OpenStreetMap (Nominatim + Overpass).

Replaces the Google Maps Places calls the legacy app made from the browser with keys
embedded in the HTML. Both services are free and need no key, but ask for a descriptive
User-Agent and modest request rates.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from ndp import __version__
from ndp.services import ServiceError

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
# Public Overpass instances are often at capacity; try them in order with a short timeout each.
OVERPASS_URLS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
)
OVERPASS_TIMEOUT = 8.0
USER_AGENT = f"natural-disaster-prediction/{__version__} (github.com/VaibhavBansal26)"

KINDS = {"hospital": "Hospital", "police": "Police", "doctors": "Doctor", "clinic": "Clinic"}


@dataclass(frozen=True)
class Place:
    name: str
    lat: float
    lon: float


@dataclass(frozen=True)
class NearbyPlace:
    name: str
    kind: str
    lat: float
    lon: float
    address: str
    phone: str


def geocode(query: str) -> Place:
    try:
        with httpx.Client(timeout=15.0, headers={"User-Agent": USER_AGENT}) as client:
            resp = client.get(NOMINATIM_URL, params={"q": query, "format": "jsonv2", "limit": 1})
            resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise ServiceError(f"Geocoding failed: {exc}") from exc
    results = resp.json()
    if not results:
        raise ServiceError(f"No location found for {query!r}.")
    hit = results[0]
    return Place(name=hit.get("display_name", query), lat=float(hit["lat"]), lon=float(hit["lon"]))


NOMINATIM_REVERSE_URL = "https://nominatim.openstreetmap.org/reverse"


def reverse_geocode(lat: float, lon: float) -> Place:
    """Name for a coordinate; falls back to the coordinate itself so alerts never block on it."""
    fallback = Place(name=f"{lat:.3f}, {lon:.3f}", lat=lat, lon=lon)
    try:
        with httpx.Client(timeout=10.0, headers={"User-Agent": USER_AGENT}) as client:
            resp = client.get(
                NOMINATIM_REVERSE_URL,
                params={"lat": lat, "lon": lon, "format": "jsonv2", "zoom": 10},
            )
            resp.raise_for_status()
            name = resp.json().get("display_name")
    except (httpx.HTTPError, ValueError):
        return fallback
    return Place(name=name, lat=lat, lon=lon) if name else fallback


def _address(tags: dict) -> str:
    parts = [
        tags.get("addr:housenumber"),
        tags.get("addr:street"),
        tags.get("addr:suburb"),
        tags.get("addr:city"),
        tags.get("addr:postcode"),
    ]
    return ", ".join(p for p in parts if p)


def _overpass(query: str) -> dict:
    last_error: Exception | None = None
    with httpx.Client(timeout=OVERPASS_TIMEOUT, headers={"User-Agent": USER_AGENT}) as client:
        for url in OVERPASS_URLS:
            try:
                resp = client.post(url, data={"data": query})
                resp.raise_for_status()
                return resp.json()
            except (httpx.HTTPError, ValueError) as exc:
                last_error = exc
    raise ServiceError(
        "OpenStreetMap's Overpass service is busy right now; please retry in a minute "
        f"(last error: {last_error})."
    )


def _nominatim_fallback(lat: float, lon: float, radius_km: float, limit: int) -> list[NearbyPlace]:
    """Nominatim understands 'hospital near ...' style queries inside a bounding box.
    Used only when every Overpass mirror is unavailable."""
    delta = radius_km / 111.0
    viewbox = f"{lon - delta},{lat + delta},{lon + delta},{lat - delta}"
    places: list[NearbyPlace] = []
    answered = False
    with httpx.Client(timeout=15.0, headers={"User-Agent": USER_AGENT}) as client:
        for amenity, kind in (("hospital", "Hospital"), ("police", "Police")):
            try:
                resp = client.get(
                    NOMINATIM_URL,
                    params={
                        "q": amenity,
                        "format": "jsonv2",
                        "limit": limit // 2,
                        "viewbox": viewbox,
                        "bounded": 1,
                    },
                )
                resp.raise_for_status()
            except httpx.HTTPError:
                continue
            answered = True
            for hit in resp.json():
                places.append(
                    NearbyPlace(
                        name=hit.get("name") or hit.get("display_name", "").split(",")[0],
                        kind=kind,
                        lat=float(hit["lat"]),
                        lon=float(hit["lon"]),
                        address=", ".join(hit.get("display_name", "").split(",")[1:4]).strip(),
                        phone="",
                    )
                )
    if not answered:
        raise ServiceError(
            "OpenStreetMap's search services are busy right now; please retry in a minute."
        )
    return places  # may be empty: an offshore or remote area really can have nothing nearby


def nearby_services(lat: float, lon: float, radius_km: float, limit: int = 40) -> list[NearbyPlace]:
    radius_m = int(max(0.5, min(radius_km, 50)) * 1000)
    kinds = "|".join(KINDS)
    query = f"""
    [out:json][timeout:{int(OVERPASS_TIMEOUT)}];
    (
      node["amenity"~"^({kinds})$"](around:{radius_m},{lat},{lon});
      way["amenity"~"^({kinds})$"](around:{radius_m},{lat},{lon});
    );
    out center tags {limit * 2};
    """
    try:
        payload = _overpass(query)
    except ServiceError:
        return _nominatim_fallback(lat, lon, radius_km, limit)[:limit]

    places: list[NearbyPlace] = []
    for el in payload.get("elements", []):
        tags = el.get("tags", {})
        if "name" not in tags:
            continue
        centre = el.get("center", el)
        if "lat" not in centre:
            continue
        places.append(
            NearbyPlace(
                name=tags["name"],
                kind=KINDS.get(tags.get("amenity", ""), "Other"),
                lat=float(centre["lat"]),
                lon=float(centre["lon"]),
                address=_address(tags),
                phone=tags.get("phone") or tags.get("contact:phone") or "",
            )
        )
    places.sort(key=lambda p: (p.lat - lat) ** 2 + (p.lon - lon) ** 2)
    return places[:limit]
