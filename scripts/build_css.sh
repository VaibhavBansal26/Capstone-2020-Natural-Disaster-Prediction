#!/usr/bin/env sh
# Compile Tailwind to a static stylesheet so pages do not ship the 300 KB in-browser compiler.
# Run after changing templates or static JS that use new utility classes. Needs Node (npx).
set -eu
cd "$(dirname "$0")/.."
npx --yes tailwindcss@3.4.17 \
  -c tailwind.config.js \
  -i src/ndp/static/tailwind.input.css \
  -o src/ndp/static/tailwind.css \
  --minify
ls -l src/ndp/static/tailwind.css
