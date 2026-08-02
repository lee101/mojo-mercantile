"""Measure the native batch kernels against mercantile's scalar Python API."""

import os
import sys
import time

import mercantile
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))
import mojo_mercantile as mojo  # noqa: E402


def measure(fn, reps=3):
    best = float("inf")
    for _ in range(reps):
        started = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - started)
    return best


def row(name, native, upstream):
    print(f"| {name} | {native * 1e3:.2f} ms | {upstream * 1e3:.2f} ms | {upstream / native:.2f}x |")


def main():
    rng = np.random.default_rng(0)
    n = 1_000_000
    coords = np.column_stack((rng.uniform(-180, 180, n), rng.uniform(-85, 85, n)))
    tile_values = mojo.tile_many(coords, 14)
    triples = np.column_stack((tile_values, np.full(n, 14, dtype=np.int64)))

    print("| kernel | mojo-mercantile | mercantile 1.2.1 | speedup |")
    print("| --- | ---: | ---: | ---: |")
    native = measure(lambda: mojo.tile_many(coords, 14))
    upstream = measure(lambda: [mercantile.tile(lng, lat, 14) for lng, lat in coords], reps=1)
    row("tile_many, 1M coordinates at z14", native, upstream)
    native = measure(lambda: [mojo.tile(lng, lat, 14) for lng, lat in coords[:100_000]])
    upstream = measure(lambda: [mercantile.tile(lng, lat, 14) for lng, lat in coords[:100_000]], reps=1)
    row("tile, 100K coordinates at z14", native, upstream)
    native = measure(lambda: mojo.bounds_many(triples))
    upstream = measure(lambda: [mercantile.bounds(*item) for item in triples], reps=1)
    row("bounds_many, 1M z14 tiles", native, upstream)


if __name__ == "__main__":
    main()
