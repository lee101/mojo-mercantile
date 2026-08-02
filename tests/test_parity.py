import math

import mercantile as reference
import mojo_mercantile as mojo
import numpy as np
import pytest


def test_coordinate_scalars_match_upstream():
    points = [(-180.0, -85.0), (-77.0365, 38.8977), (0.0, 0.0), (179.9, 84.9)]
    for point in points:
        assert mojo.xy(*point) == pytest.approx(reference.xy(*point), abs=1e-9)
        assert mojo.lnglat(*mojo.xy(*point)) == pytest.approx(reference.lnglat(*reference.xy(*point)), abs=1e-12)
    assert math.isinf(mojo.xy(0, 90)[1]) and math.isinf(mojo.xy(0, -90)[1])


def test_tile_bounds_and_quadkey_match_upstream_for_random_tiles():
    rng = np.random.default_rng(7)
    for zoom in (0, 1, 5, 12, 22):
        n = 1 << zoom
        for x, y in rng.integers(0, n, size=(40, 2)):
            current = reference.Tile(int(x), int(y), zoom)
            assert mojo.bounds(current) == pytest.approx(reference.bounds(current), abs=1e-12)
            assert mojo.ul(current) == pytest.approx(reference.ul(current), abs=1e-12)
            assert mojo.xy_bounds(current) == pytest.approx(reference.xy_bounds(current), abs=1e-9)
            assert mojo.quadkey(current) == reference.quadkey(current)
            assert mojo.quadkey_to_tile(mojo.quadkey(current)) == current


def test_tile_indices_match_upstream_at_multiple_zooms():
    rng = np.random.default_rng(8)
    points = np.column_stack((rng.uniform(-180, 180, 500), rng.uniform(-85, 85, 500)))
    for zoom in (0, 1, 8, 16, 24):
        assert [mojo.tile(lng, lat, zoom) for lng, lat in points] == [reference.tile(lng, lat, zoom) for lng, lat in points]


def test_scalar_tile_does_not_cross_the_batch_ffi(monkeypatch):
    def fail():
        raise AssertionError("scalar tile should not allocate an FFI output buffer")

    monkeypatch.setattr(mojo, "lib", fail)
    assert mojo.tile(-77.0365, 38.8977, 14) == reference.tile(-77.0365, 38.8977, 14)


def test_tile_collection_helpers_match_upstream():
    current = reference.Tile(486, 332, 10)
    assert mojo.parent(current) == reference.parent(current)
    assert mojo.parent(current, zoom=4) == reference.parent(current, zoom=4)
    assert mojo.children(current) == reference.children(current)
    assert mojo.children(current, zoom=12) == reference.children(current, zoom=12)
    assert mojo.neighbors(current) == reference.neighbors(current)
    assert mojo.simplify(reference.children(reference.Tile(1, 1, 2))) == reference.simplify(reference.children(reference.Tile(1, 1, 2)))


@pytest.mark.parametrize("bbox", [
    (-77.1, 38.8, -77.0, 38.9),
    (170.0, -10.0, -170.0, 10.0),
    (-180.0, -85.051129, 180.0, 85.051129),
])
def test_bbox_operations_match_upstream(bbox):
    assert list(mojo.tiles(*bbox, [2, 5, 9])) == list(reference.tiles(*bbox, [2, 5, 9]))
    assert mojo.bounding_tile(*bbox) == reference.bounding_tile(*bbox)


def test_feature_and_geojson_bounds_match_upstream():
    current = reference.Tile(486, 332, 10)
    assert mojo.feature(current, fid="example", props={"source": "test"}, precision=6) == reference.feature(current, fid="example", props={"source": "test"}, precision=6)
    geometry = {"type": "Polygon", "coordinates": [[[-77, 38], [-76, 38], [-76, 39], [-77, 38]]]}
    assert mojo.geojson_bounds(geometry) == reference.geojson_bounds(geometry)


def test_batch_mojo_kernels_match_scalar_upstream():
    rng = np.random.default_rng(9)
    coords = np.column_stack((rng.uniform(-180, 180, 10_000), rng.uniform(-85, 85, 10_000)))
    tiles = mojo.tile_many(coords, 14)
    expected_tiles = np.array([(t.x, t.y) for t in (reference.tile(lng, lat, 14) for lng, lat in coords)], dtype=np.int64)
    assert np.array_equal(tiles, expected_tiles)
    triples = np.column_stack((tiles[:200], np.full(200, 14, dtype=np.int64)))
    expected_bounds = np.array([reference.bounds(*item) for item in triples])
    assert mojo.bounds_many(triples) == pytest.approx(expected_bounds, abs=1e-12)


def test_batch_coordinate_compatibility_matches_upstream():
    coords = np.array([[-77.0365, 38.8977], [0.0, 0.0], [12.3, -45.6]])
    expected_xy = np.array([reference.xy(*point) for point in coords])
    assert mojo.xy_many(coords) == pytest.approx(expected_xy, abs=1e-9)
    expected_lnglat = np.array([reference.lnglat(*point) for point in expected_xy])
    assert mojo.lnglat_many(expected_xy) == pytest.approx(expected_lnglat, abs=1e-12)


def test_batch_ffi_boundary_validation_and_noncontiguous_inputs():
    coords = np.array([[-77.0365, 38.8977], [2.2945, 48.8584]], dtype=np.float64)
    padded = np.empty((2, 4), dtype=np.float64)
    padded[:, ::2] = coords
    assert np.array_equal(mojo.tile_many(padded[:, ::2], 14), mojo.tile_many(coords, 14))
    with pytest.raises(TypeError):
        mojo.bounds_many(np.array([[1.0, 2.0, 3.0]]))
    with pytest.raises(mojo.InvalidZoomError):
        mojo.tile_many(coords, 63)
    with pytest.raises(mojo.InvalidZoomError):
        mojo.bounds_many(np.array([[1, 2, 63]], dtype=np.int64))
    with pytest.raises(ValueError):
        mojo.tile_many(np.array([[np.nan, 0.0]]), 2)
    with pytest.raises(mojo.InvalidLatitudeError):
        mojo.tile_many(np.array([[0.0, 90.0]]), 2)


def test_empty_batch_arrays_do_not_cross_the_ffi(monkeypatch):
    monkeypatch.setattr(mojo, "lib", lambda: (_ for _ in ()).throw(AssertionError("unexpected FFI call")))
    assert mojo.tile_many(np.empty((0, 2), dtype=np.float64), 14).shape == (0, 2)
    assert mojo.bounds_many(np.empty((0, 3), dtype=np.int64)).shape == (0, 4)


def test_scalar_bounds_falls_back_for_large_integer_zooms():
    current = reference.Tile(2**70 - 1, 2**70 - 1, 70)
    assert mojo.bounds(current) == pytest.approx(reference.bounds(current), abs=1e-12)


def test_upstream_error_behaviour():
    with pytest.raises(mojo.InvalidZoomError):
        mojo.minmax(-1)
    with pytest.raises(mojo.InvalidLatitudeError):
        mojo.tile(0, 90, 3)
    with pytest.warns(DeprecationWarning), pytest.raises(mojo.QuadKeyError):
        mojo.quadkey_to_tile("123x")


def test_value_helpers_match_upstream():
    assert mojo.minmax(12) == reference.minmax(12)
    assert mojo.truncate_lnglat(200, -100) == reference.truncate_lnglat(200, -100)
    assert isinstance(mojo.tile(0, 0, 1), mojo.Tile)
    assert isinstance(mojo.lnglat(0, 0), mojo.LngLat)
    assert isinstance(mojo.bounds(0, 0, 1), mojo.LngLatBbox)
    assert isinstance(mojo.xy_bounds(0, 0, 1), mojo.Bbox)
