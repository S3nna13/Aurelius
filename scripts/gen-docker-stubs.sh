#!/bin/sh
# Generate stub package.json files for all workspace crates.
# Needed when building Docker images with npm workspaces.
# Usage: gen-docker-stubs.sh

for dir in "$@"; do
  name=$(basename "$dir" | sed 's/^/aurelius-/')
  mkdir -p "$dir"
  if [ ! -f "$dir/package.json" ]; then
    printf '{"name":"%s","private":true}\n' "$name" > "$dir/package.json"
  fi
done
