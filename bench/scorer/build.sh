#!/bin/sh
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
mkdir -p "$root/build/scorer"
${CC:-cc} -std=c11 -D_POSIX_C_SOURCE=200809L -O2 -fPIC -shared \
  -I"$root/src" "$root/bench/scorer/reference.c" \
  "$root/src/sym.c" "$root/src/sym_tree_refine.c" "$root/src/sym_tree_io.c" \
  "$root/src/sym_family_io.c" "$root/src/wl_api.c" \
  -lm -o "$root/build/scorer/reference.so"
