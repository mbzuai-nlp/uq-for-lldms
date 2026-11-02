#!/usr/bin/env bash
# Set up the Fast-dLLM checkout that Dream-7B support requires.
#
# llada/load_dream_cached.py imports Dream's cache-aware model code from
#   <repo>/Fast-dLLM/dream/model/modeling_dream.py
# which is not vendored in this repo. Run this once before evaluating Dream:
#   bash scripts/setup_dream.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$REPO_ROOT/Fast-dLLM"
FASTDLLM_URL="https://github.com/NVlabs/Fast-dLLM.git"
FASTDLLM_COMMIT="af42f511f705a2db0d16aa63ebcfcbc26f29181d"

if [ -e "$DEST/dream/model/modeling_dream.py" ]; then
    echo "Fast-dLLM already present at $DEST — nothing to do."
    exit 0
fi

echo "Cloning Fast-dLLM into $DEST ..."
git clone "$FASTDLLM_URL" "$DEST"
git -C "$DEST" checkout "$FASTDLLM_COMMIT"
echo "Done. Dream can now be loaded via llada/load_dream_cached.py."
