"""Web Mercator coordinate and XYZ tile kernels exposed through a C ABI."""

from std.math import atan, exp, floor, log, sin, sinh, tan


comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]

comptime RE: Float64 = 6378137.0
comptime R2D: Float64 = 57.2957795130823208768
comptime PI: Float64 = 3.14159265358979323846
comptime EPSILON: Float64 = 0.00000000000001


def clamp_lnglat(lng: Float64, lat: Float64) -> Tuple[Float64, Float64]:
    var clipped_lng = lng
    var clipped_lat = lat
    if clipped_lng > 180.0:
        clipped_lng = 180.0
    elif clipped_lng < -180.0:
        clipped_lng = -180.0
    if clipped_lat > 90.0:
        clipped_lat = 90.0
    elif clipped_lat < -90.0:
        clipped_lat = -90.0
    return (clipped_lng, clipped_lat)


def accurate_sin(x: Float64) -> Float64:
    var x2 = x * x
    var poly = 1.0 / 51090942171709440000.0
    poly = -1.0 / 121645100408832000.0 + x2 * poly
    poly = 1.0 / 355687428096000.0 + x2 * poly
    poly = -1.0 / 1307674368000.0 + x2 * poly
    poly = 1.0 / 6227020800.0 + x2 * poly
    poly = -1.0 / 39916800.0 + x2 * poly
    poly = 1.0 / 362880.0 + x2 * poly
    poly = -1.0 / 5040.0 + x2 * poly
    poly = 1.0 / 120.0 + x2 * poly
    poly = -1.0 / 6.0 + x2 * poly
    return x * (1.0 + x2 * poly)


def mercator_xy(lng: Float64, lat: Float64, truncate: Int, dst: FPtr):
    var use_lng = lng
    var use_lat = lat
    if truncate != 0:
        use_lng, use_lat = clamp_lnglat(lng, lat)
    dst[0] = RE * use_lng * PI / 180.0
    var sinlat = accurate_sin(use_lat * PI / 180.0)
    dst[1] = RE * 0.5 * log((1.0 + sinlat) / (1.0 - sinlat))


def mercator_lnglat(x: Float64, y: Float64, truncate: Int, dst: FPtr):
    var lng = x * R2D / RE
    var lat = (PI * 0.5 - 2.0 * atan(exp(-y / RE))) * R2D
    if truncate != 0:
        lng, lat = clamp_lnglat(lng, lat)
    dst[0] = lng
    dst[1] = lat


def tile_index(lng: Float64, lat: Float64, zoom: Int, truncate: Int, dst: IPtr):
    var use_lng = lng
    var use_lat = lat
    if truncate != 0:
        use_lng, use_lat = clamp_lnglat(lng, lat)
    var x = use_lng / 360.0 + 0.5
    var sinlat = sin(use_lat * PI / 180.0)
    var y = 0.5 - 0.25 * log((1.0 + sinlat) / (1.0 - sinlat)) / PI
    var z2 = Float64(1 << zoom)
    var xtile: Int
    var ytile: Int
    if x <= 0.0:
        xtile = 0
    elif x >= 1.0:
        xtile = Int(z2 - 1.0)
    else:
        xtile = Int(floor((x + EPSILON) * z2))
    if y <= 0.0:
        ytile = 0
    elif y >= 1.0:
        ytile = Int(z2 - 1.0)
    else:
        ytile = Int(floor((y + EPSILON) * z2))
    dst[0] = Int64(xtile)
    dst[1] = Int64(ytile)


def tile_bounds(x: Int64, y: Int64, zoom: Int64, dst: FPtr):
    var z2 = Float64(1 << Int(zoom))
    var west = Float64(x) / z2 * 360.0 - 180.0
    var north = atan(sinh(PI * (1.0 - 2.0 * Float64(y) / z2))) * R2D
    var east = Float64(x + 1) / z2 * 360.0 - 180.0
    var south = atan(sinh(PI * (1.0 - 2.0 * Float64(y + 1) / z2))) * R2D
    dst[0] = west
    dst[1] = south
    dst[2] = east
    dst[3] = north


@export("mm_xy")
def mm_xy(lng: Float64, lat: Float64, truncate: Int, dst_addr: Int) abi("C"):
    mercator_xy(lng, lat, truncate, FPtr(unsafe_from_address=dst_addr))


@export("mm_lnglat")
def mm_lnglat(x: Float64, y: Float64, truncate: Int, dst_addr: Int) abi("C"):
    mercator_lnglat(x, y, truncate, FPtr(unsafe_from_address=dst_addr))


@export("mm_tile")
def mm_tile(lng: Float64, lat: Float64, zoom: Int, truncate: Int, dst_addr: Int) abi("C"):
    tile_index(lng, lat, zoom, truncate, IPtr(unsafe_from_address=dst_addr))


@export("mm_bounds")
def mm_bounds(x: Int64, y: Int64, zoom: Int64, dst_addr: Int) abi("C"):
    tile_bounds(x, y, zoom, FPtr(unsafe_from_address=dst_addr))


@export("mm_xy_many")
def mm_xy_many(src_addr: Int, n: Int, truncate: Int, dst_addr: Int) abi("C"):
    var src = FPtr(unsafe_from_address=src_addr)
    var dst = FPtr(unsafe_from_address=dst_addr)
    for i in range(n):
        mercator_xy(src[2 * i], src[2 * i + 1], truncate, dst + 2 * i)


@export("mm_lnglat_many")
def mm_lnglat_many(src_addr: Int, n: Int, truncate: Int, dst_addr: Int) abi("C"):
    var src = FPtr(unsafe_from_address=src_addr)
    var dst = FPtr(unsafe_from_address=dst_addr)
    for i in range(n):
        mercator_lnglat(src[2 * i], src[2 * i + 1], truncate, dst + 2 * i)


@export("mm_tile_many")
def mm_tile_many(src_addr: Int, n: Int, zoom: Int, truncate: Int, dst_addr: Int) abi("C"):
    var src = FPtr(unsafe_from_address=src_addr)
    var dst = IPtr(unsafe_from_address=dst_addr)
    for i in range(n):
        tile_index(src[2 * i], src[2 * i + 1], zoom, truncate, dst + 2 * i)


@export("mm_bounds_many")
def mm_bounds_many(src_addr: Int, n: Int, dst_addr: Int) abi("C"):
    var src = IPtr(unsafe_from_address=src_addr)
    var dst = FPtr(unsafe_from_address=dst_addr)
    for i in range(n):
        tile_bounds(src[3 * i], src[3 * i + 1], src[3 * i + 2], dst + 4 * i)
