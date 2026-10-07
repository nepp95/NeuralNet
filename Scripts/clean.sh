#!/bin/sh

set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)

if [ ! -f "$ROOT/premake5.lua" ] || [ ! -d "$ROOT/NeuralNetwork" ]; then
    echo "Refusing to clean unexpected repository root '$ROOT'." >&2
    exit 1
fi

rm -rf -- \
    "$ROOT/build" \
    "$ROOT/.tools" \
    "$ROOT/.vcpkg-cache"

find "$ROOT" -type d -name __pycache__ -prune -exec rm -rf -- {} \;
find "$ROOT" -type f \( \
    -name '*.sln' -o \
    -name '*.slnx' -o \
    -name '*.vcxproj' -o \
    -name '*.vcxproj.filters' -o \
    -name '*.ninja' -o \
    -name '.ninja_deps' -o \
    -name '.ninja_log' -o \
    -name 'compile_commands.json' -o \
    -name '*.pyc' -o \
    -name '*.pyo' \
\) -delete

echo "Removed all generated build artifacts."
