# mojo-mercantile

`mojo-mercantile` is a Mojo-accelerated port of [mercantile](https://github.com/mapbox/mercantile)'s Web Mercator XYZ tile mathematics. It provides a separate Python module, `mojo_mercantile`, for applications that opt in explicitly.

The project validates against the real conda-forge `mercantile` 1.2.1 package. The native kernels make repeated point-to-tile and tile-to-bounds work practical without Python-loop overhead.

## Coverage

The compatibility layer implements and tests these mercantile 1.2.1 functions: `xy`, `lnglat`, `tile`, `bounds`, `ul`, `xy_bounds`, `quadkey`, `quadkey_to_tile`, `tiles`, `parent`, `children`, `neighbors`, `simplify`, `bounding_tile`, `feature`, `geojson_bounds`, `minmax`, and `truncate_lnglat`, plus the `Tile`, `LngLat`, `LngLatBbox`, and `Bbox` return types. The tests compare normal results and selected errors with the installed upstream package.

`xy_many`, `lnglat_many`, `tile_many`, and `bounds_many` are additional NumPy array APIs. Mojo acceleration applies to `tile_many` and `bounds_many`; scalar `tile`, `xy`, and `lnglat` remain Python implementations. The upstream CLI, package metadata, and undocumented internals are not included. Batch inputs are copied to C-contiguous buffers when needed. Coordinate arrays must be shaped `(n, 2)` and tile arrays `(n, 3)`; `bounds_many` requires integer tiles and both native batch kernels support zooms 0 through 62.

## Install and use

```bash
pixi install
pixi run build
```

The Pixi activation adds `python/` to `PYTHONPATH`.

```python
import numpy as np
import mojo_mercantile as mercantile

tile = mercantile.tile(-77.0365, 38.8977, 14)
print(tile)                           # Tile(x=4685, y=6267, z=14)
print(mercantile.bounds(tile))

points = np.array([[-77.0365, 38.8977], [2.2945, 48.8584]])
print(mercantile.tile_many(points, 14))
```

Run the verified suite and benchmark with:

```bash
pixi run test
pixi run bench
```

## Benchmark

Measured on 2026-08-02 by `pixi run bench`, using Mojo `1.0.0b3.dev2026072406`, Python 3.13, and mercantile 1.2.1. Values are the best of three Mojo runs and one upstream Python-loop run. Results are machine-dependent; the benchmark task takes a machine-wide flock.

| kernel | mojo-mercantile | mercantile 1.2.1 | speedup |
| --- | ---: | ---: | ---: |
| tile_many, 1M coordinates at z14 | 81.54 ms | 3665.14 ms | 44.95x |
| tile, 100K coordinates at z14 | 342.83 ms | 340.14 ms | 0.99x |
| bounds_many, 1M z14 tiles | 172.43 ms | 17696.98 ms | 102.64x |

## How it works

`src/mercantile.mojo` builds to `dist/libmojo-mercantile.so`. The ctypes wrapper normalizes and retains NumPy buffers for the duration of each synchronous call, then passes their addresses and row counts to a small C ABI. Mojo writes only to wrapper-owned output arrays: coordinate inputs are `float64 (n, 2)`, tile inputs `int64 (n, 3)`, and bounds outputs `float64 (n, 4)`. There is no GPU path.
