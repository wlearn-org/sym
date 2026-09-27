#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DIST_DIR="${PROJECT_DIR}/dist"

NAME=$(node -e "const p=require('${PROJECT_DIR}/package.json'); console.log(p.name.split('/').pop())")
EXPORTS=$(node -e '
  const fs = require("fs")
  const path = require("path")
  const src = fs.readFileSync(path.join("'"${PROJECT_DIR}"'", "src", "index.js"), "utf8")
  const noComments = src.replace(/\/\*[\s\S]*?\*\//g, "").replace(/(^|[^:])\/\/.*$/gm, (_, p) => p)
  const match = noComments.match(/module\.exports\s*=\s*\{([\s\S]*?)\}/m)
  if (!match) throw new Error("Could not find module.exports object in src/index.js")
  const names = match[1].split(",").map(s => s.trim()).filter(Boolean).map(s => s.split(":")[0].trim())
  console.log(names.join(","))
')

mkdir -p "$DIST_DIR"

COMMON_FLAGS=(
  --bundle
  --platform=browser
  --minify
  --alias:node:fs=./scripts/empty.js
  --alias:fs=./scripts/empty.js
  --alias:node:crypto=./scripts/empty.js
  --alias:node:path=./scripts/empty.js
  --alias:ws=./scripts/empty.js
  --define:__dirname='""'
  --define:__filename='""'
)

npx esbuild "${PROJECT_DIR}/src/index.js" \
  "${COMMON_FLAGS[@]}" \
  --format=iife \
  --global-name="${NAME}" \
  --outfile="${DIST_DIR}/${NAME}.js"

INTERNAL="__${NAME}"
npx esbuild "${PROJECT_DIR}/src/index.js" \
  "${COMMON_FLAGS[@]}" \
  --format=iife \
  --global-name="${INTERNAL}" \
  --outfile="${DIST_DIR}/${NAME}.mjs"

IFS=',' read -ra KEYS <<< "$EXPORTS"
DESTRUCTURE=$(IFS=','; echo "${KEYS[*]}")
EXPORT_LINE=$(IFS=','; echo "${KEYS[*]}")
echo "var {${DESTRUCTURE}}=${INTERNAL};export{${EXPORT_LINE}};" >> "${DIST_DIR}/${NAME}.mjs"

ls -lh "${DIST_DIR}/${NAME}.js" "${DIST_DIR}/${NAME}.mjs"
