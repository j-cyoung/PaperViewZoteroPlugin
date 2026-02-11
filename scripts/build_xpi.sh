#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out_file="$repo_root/paperview-query.xpi"
package_items=(
  "manifest.json"
  "bootstrap.js"
  "chrome.manifest"
  "locale"
  "skin"
  "service"
)

cd "$repo_root"

for item in "${package_items[@]}"; do
  if [[ ! -e "$item" ]]; then
    echo "Missing required package item: $item" >&2
    exit 1
  fi
done

rm -f "$out_file"
zip -rFS "$out_file" "${package_items[@]}" \
  -x "__MACOSX/*" \
  -x ".DS_Store" \
  -x "*/__pycache__/*" \
  -x "*.pyc"

echo "Built: $out_file"
