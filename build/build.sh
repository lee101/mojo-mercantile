#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$root/dist"
mojo build --emit shared-lib "$root/src/mercantile.mojo" -o "$root/dist/libmojo-mercantile.so"
