#!/usr/bin/env bash
# Fetch the earlier eeld-cf code base that the method comparison imports (repaired procedures and
# EELD-adapted counterfactual generators). It is pinned to the commit used for the stored runs.
set -euo pipefail
cd "$(dirname "$0")/.."
DEST=${EELD_LEGACY_REPO:-external/eeld-cf}
COMMIT=07e983a
if [ ! -d "$DEST/.git" ]; then
  git clone --quiet https://github.com/epereirasg/eeld-cf "$DEST"
fi
git -C "$DEST" checkout --quiet "$COMMIT"
echo "eeld-cf at $(git -C "$DEST" rev-parse --short HEAD) in $DEST"
