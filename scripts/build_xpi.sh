#!/bin/bash
set -euo pipefail

plugin_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../zotero-plugin" && pwd)"
out_file="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/paperview-query.xpi"

if [[ ! -d "$plugin_dir" ]]; then
  echo "Plugin directory not found: $plugin_dir" >&2
  exit 1
fi

cd "$plugin_dir"
# macOS zip includes __MACOSX, so exclude it
zip -rFS "$out_file" . \
  -x "__MACOSX/*" \
  -x ".DS_Store" \
  -x "*/__pycache__/*" \
  -x "*.pyc"

echo "Built: $out_file"
