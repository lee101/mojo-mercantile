"""A Mojo-backed, source-compatible subset of :mod:`mercantile` 1.2.1."""

from __future__ import annotations

import math
import operator
import warnings
from collections import namedtuple
from collections.abc import Sequence
from functools import reduce

import numpy as np

from ._lib import addr, f64_pairs, i64_triples, lib

__version__ = "0.1.0"
__all__ = [
    "Bbox", "LngLat", "LngLatBbox", "Tile", "bounding_tile", "bounds", "children",
    "feature", "geojson_bounds", "lnglat", "minmax", "neighbors", "parent", "quadkey",
    "quadkey_to_tile", "simplify", "tile", "tiles", "truncate_lnglat", "ul", "xy",
    "xy_bounds", "xy_many", "lnglat_many", "tile_many", "bounds_many",
]

R2D = 180 / math.pi
RE = 6378137.0
CE = 2 * math.pi * RE
EPSILON = 1e-14
LL_EPSILON = 1e-11


class MercantileError(Exception):
    pass


class InvalidLatitudeError(MercantileError):
    pass


class InvalidZoomError(MercantileError):
    pass


class ParentTileError(MercantileError):
    pass


class QuadKeyError(MercantileError):
    pass


class TileArgParsingError(MercantileError):
    pass


class TileError(MercantileError):
    pass


class Tile(namedtuple("Tile", ["x", "y", "z"])):
    def __new__(cls, x, y, z):
        lo, hi = minmax(z)
        if not lo <= x <= hi or not lo <= y <= hi:
            warnings.warn(
                "Mercantile 2.0 will require tile x and y to be within the range (0, 2 ** zoom)",
                FutureWarning,
            )
        return tuple.__new__(cls, [x, y, z])


LngLat = namedtuple("LngLat", ["lng", "lat"])
LngLatBbox = namedtuple("LngLatBbox", ["west", "south", "east", "north"])
Bbox = namedtuple("Bbox", ["left", "bottom", "right", "top"])


def _parse_tile_arg(*args):
    if len(args) == 1:
        args = args[0]
    if len(args) == 3:
        return Tile(*args)
    raise TileArgParsingError(
        "the tile argument may have 1 or 3 values. Note that zoom is a keyword-only argument"
    )


def minmax(zoom):
    try:
        if int(zoom) != zoom or zoom < 0:
            raise InvalidZoomError("zoom must be a positive integer")
    except ValueError:
        raise InvalidZoomError("zoom must be a positive integer")
    return 0, 2 ** zoom - 1


def truncate_lnglat(lng, lat):
    return min(180.0, max(-180.0, lng)), min(90.0, max(-90.0, lat))


def xy(lng, lat, truncate=False):
    if truncate:
        lng, lat = truncate_lnglat(lng, lat)
    if lat <= -90:
        return RE * math.radians(lng), float("-inf")
    if lat >= 90:
        return RE * math.radians(lng), float("inf")
    return RE * math.radians(lng), RE * math.log(math.tan(math.pi * 0.25 + 0.5 * math.radians(lat)))


def lnglat(x, y, truncate=False):
    lng, lat = x * R2D / RE, (math.pi * 0.5 - 2.0 * math.atan(math.exp(-y / RE))) * R2D
    if truncate:
        lng, lat = truncate_lnglat(lng, lat)
    return LngLat(lng, lat)


def _xy(lng, lat, truncate=False):
    if truncate:
        lng, lat = truncate_lnglat(lng, lat)
    x = lng / 360.0 + 0.5
    sinlat = math.sin(math.radians(lat))
    try:
        y = 0.5 - 0.25 * math.log((1.0 + sinlat) / (1.0 - sinlat)) / math.pi
    except (ValueError, ZeroDivisionError):
        raise InvalidLatitudeError(f"Y can not be computed: lat={lat!r}")
    return x, y


def tile(lng, lat, zoom, truncate=False):
    x, y = _xy(lng, lat, truncate)
    z2 = math.pow(2, zoom)
    return Tile(
        0 if x <= 0 else int(z2 - 1) if x >= 1 else int(math.floor((x + EPSILON) * z2)),
        0 if y <= 0 else int(z2 - 1) if y >= 1 else int(math.floor((y + EPSILON) * z2)),
        zoom,
    )


def bounds(*tile):
    x, y, z = _parse_tile_arg(*tile)
    # Mojo's native shift is defined only for non-negative shifts smaller
    # than the signed machine-word width. Keep the scalar API compatible
    # with mercantile for every other numeric zoom by using its formula.
    if x != int(x) or y != int(y) or z != int(z) or not 0 <= z <= 62:
        z2 = math.pow(2, z)
        return LngLatBbox(
            x / z2 * 360.0 - 180.0,
            math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / z2)))),
            (x + 1) / z2 * 360.0 - 180.0,
            math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / z2)))),
        )
    dst = np.empty(4, dtype=np.float64)
    lib().mm_bounds(x, y, z, addr(dst))
    return LngLatBbox(*dst)


def ul(*tile):
    west, _, _, north = bounds(*tile)
    return LngLat(west, north)


def xy_bounds(*tile):
    x, y, z = _parse_tile_arg(*tile)
    tile_size = CE / math.pow(2, z)
    left = x * tile_size - CE / 2
    top = CE / 2 - y * tile_size
    return Bbox(left, top - tile_size, left + tile_size, top)


def quadkey(*tile):
    x, y, z = _parse_tile_arg(*tile)
    digits = []
    for level in range(z, 0, -1):
        mask = 1 << (level - 1)
        digits.append(str((1 if x & mask else 0) + (2 if y & mask else 0)))
    return "".join(digits)


def quadkey_to_tile(qk):
    if len(qk) == 0:
        return Tile(0, 0, 0)
    x = y = 0
    for i, digit in enumerate(reversed(qk)):
        mask = 1 << i
        if digit == "1": x |= mask
        elif digit == "2": y |= mask
        elif digit == "3": x |= mask; y |= mask
        elif digit != "0":
            warnings.warn("QuadKeyError will not derive from ValueError in mercantile 2.0.", DeprecationWarning)
            raise QuadKeyError("Unexpected quadkey digit: %r" % digit)
    return Tile(x, y, i + 1)


def neighbors(*tile, **kwargs):
    x, y, z = _parse_tile_arg(*tile)
    _, hi = minmax(z)
    return [Tile(x + i, y + j, z) for i in (-1, 0, 1) for j in (-1, 0, 1)
            if (i or j) and 0 <= x + i <= hi and 0 <= y + j <= hi and z >= 0]


def parent(*tile, **kwargs):
    x, y, z = _parse_tile_arg(*tile)
    if z == 0:
        return None
    zoom = kwargs.get("zoom")
    if zoom is not None and (z <= zoom or zoom != int(zoom)):
        raise InvalidZoomError("zoom must be an integer and less than that of the input tile")
    if x != int(x) or y != int(y) or z != int(z):
        raise ParentTileError("the parent of a non-integer tile is undefined")
    target = z - 1 if zoom is None else zoom
    return Tile(x >> (z - target), y >> (z - target), target)


def children(*tile, **kwargs):
    x, y, z = _parse_tile_arg(*tile)
    zoom = kwargs.get("zoom")
    if zoom is not None and (z > zoom or zoom != int(zoom)):
        raise InvalidZoomError("zoom must be an integer and greater than that of the input tile")
    target = z + 1 if zoom is None else zoom
    result = [Tile(x, y, z)]
    while result[0].z < target:
        current = result.pop(0)
        result += [
            Tile(current.x * 2, current.y * 2, current.z + 1),
            Tile(current.x * 2 + 1, current.y * 2, current.z + 1),
            Tile(current.x * 2 + 1, current.y * 2 + 1, current.z + 1),
            Tile(current.x * 2, current.y * 2 + 1, current.z + 1),
        ]
    return result


def tiles(west, south, east, north, zooms, truncate=False):
    if truncate:
        west, south = truncate_lnglat(west, south)
        east, north = truncate_lnglat(east, north)
    bboxes = [(-180.0, south, east, north), (west, south, 180.0, north)] if west > east else [(west, south, east, north)]
    zooms = [zooms] if not isinstance(zooms, Sequence) else zooms
    for w, s, e, n in bboxes:
        w, s, e, n = max(-180.0, w), max(-85.051129, s), min(180.0, e), min(85.051129, n)
        for z in zooms:
            a, b = tile(w, n, z), tile(e - LL_EPSILON, s + LL_EPSILON, z)
            for x in range(a.x, b.x + 1):
                for y in range(a.y, b.y + 1):
                    yield Tile(x, y, z)


def simplify(tiles):
    roots = set()
    for current in sorted(tiles, key=operator.itemgetter(2)):
        if not any(parent(current, zoom=z) in roots for z in range(current.z)):
            roots.add(current)
    changed = True
    while changed:
        grouped, changed = {}, False
        for current in roots:
            grouped.setdefault(parent(current), set()).add(current)
        roots = []
        for ancestor, descendants in grouped.items():
            if len(descendants) == 4:
                roots.append(ancestor); changed = True
            else:
                roots += list(descendants)
    return roots


def bounding_tile(*bbox, **kwds):
    if len(bbox) == 2:
        bbox += bbox
    w, s, e, n = bbox
    if kwds.get("truncate"):
        w, s = truncate_lnglat(w, s); e, n = truncate_lnglat(e, n)
    try:
        a, b = tile(w, n, 32), tile(e - LL_EPSILON, s + LL_EPSILON, 32)
    except InvalidLatitudeError:
        return Tile(0, 0, 0)
    for z in range(28):
        mask = 1 << (32 - z - 1)
        if (a.x & mask) != (b.x & mask) or (a.y & mask) != (b.y & mask):
            return Tile(a.x >> (32 - z), a.y >> (32 - z), z) if z else Tile(0, 0, 0)
    return Tile(a.x >> 4, a.y >> 4, 28)


def feature(tile, fid=None, props=None, projected="geographic", buffer=None, precision=None):
    west, south, east, north = bounds(tile)
    if projected == "mercator":
        west, south = xy(west, south); east, north = xy(east, north)
    if buffer:
        west -= buffer; south -= buffer; east += buffer; north += buffer
    if precision and precision >= 0:
        west, south, east, north = (round(v, precision) for v in (west, south, east, north))
    geometry = {"type": "Polygon", "coordinates": [[[west, south], [west, north], [east, north], [east, south], [west, south]]]}
    result = {"type": "Feature", "bbox": [min(west, east), min(south, north), max(west, east), max(south, north)],
              "id": str(tile), "geometry": geometry, "properties": {"title": "XYZ tile %s" % str(tile)}}
    if props: result["properties"].update(props)
    if fid is not None: result["id"] = fid
    return result


def _coords(obj):
    coordinates = obj if isinstance(obj, (tuple, list)) else ([f["geometry"]["coordinates"] for f in obj["features"]] if "features" in obj else obj.get("coordinates", obj.get("geometry", obj).get("coordinates")))
    for element in coordinates:
        if isinstance(element, (float, int)):
            yield tuple(coordinates); break
        yield from _coords(element)


def geojson_bounds(obj):
    return LngLatBbox(*reduce(lambda box, point: (min(box[0], point[0]), min(box[1], point[1]), max(box[2], point[0]), max(box[3], point[1])), _coords(obj), (180.0, 90.0, -180.0, -90.0)))


def xy_many(coords, truncate=False):
    src = f64_pairs(coords)
    lng, lat = src[:, 0], src[:, 1]
    if truncate:
        lng, lat = np.clip(lng, -180.0, 180.0), np.clip(lat, -90.0, 90.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.column_stack((RE * np.radians(lng), RE * np.log(np.tan(np.pi * 0.25 + 0.5 * np.radians(lat)))))


def lnglat_many(coords, truncate=False):
    src = f64_pairs(coords)
    dst = np.column_stack((src[:, 0] * R2D / RE, (np.pi * 0.5 - 2.0 * np.arctan(np.exp(-src[:, 1] / RE))) * R2D))
    return np.clip(dst, (-180.0, -90.0), (180.0, 90.0)) if truncate else dst


def tile_many(coords, zoom, truncate=False):
    src = f64_pairs(coords)
    if isinstance(zoom, bool) or int(zoom) != zoom or not 0 <= zoom <= 62:
        raise InvalidZoomError("batch zoom must be an integer in the range 0 through 62")
    if not np.isfinite(src).all():
        raise ValueError("coordinates must be finite")
    # At these exact poles mercantile's scalar logarithm raises instead of
    # returning a tile; do the same before entering a no-error C ABI.
    if np.any(np.abs(np.sin(np.radians(src[:, 1]))) == 1.0):
        raise InvalidLatitudeError("Y can not be computed for a pole latitude")
    dst = np.empty((len(src), 2), dtype=np.int64)
    if not len(src):
        return dst
    lib().mm_tile_many(addr(src), len(src), int(zoom), int(truncate), addr(dst))
    return dst


def bounds_many(tile_values):
    src = i64_triples(tile_values)
    if src.size and (np.any(src[:, 2] < 0) or np.any(src[:, 2] > 62)):
        raise InvalidZoomError("batch tile zooms must be in the range 0 through 62")
    dst = np.empty((len(src), 4), dtype=np.float64)
    if not len(src):
        return dst
    lib().mm_bounds_many(addr(src), len(src), addr(dst))
    return dst
