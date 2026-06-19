#!/usr/bin/env bash
# Reproducible de-leak of the v4 structural hard negatives against the FROZEN
# external validation set. A structural negative is a composition-matched shuffle
# of a training positive; if one coincidentally lands >=80% identical to an external
# sequence (esp. an external POSITIVE), training on it would leak the external test.
#
# CD-HIT-EST-2D enforces a hard threshold floor of 0.80 (short-word constraint);
# 0.80 is also the exact threshold used to build external_deleaked.fasta, so this
# keeps the de-leak consistent end-to-end.
#
# Result (2026-06-10): 185 -> 184 kept; dropped parent=316 di-shuffle (82.0% ident
# to ext|fabrini2024|C_tilde, a 39-nt designed RNA whose shuffle collided by composition).
set -euo pipefail
cd "$(dirname "$0")/.."
CDHIT=/opt/homebrew/Caskroom/mambaforge/base/envs/rnaphaseek/bin/cd-hit-est-2d
EXT=Data/raw/multispecies/external/external_deleaked.fasta
RAW=Data/raw/multispecies/strict_struct_negatives_v4.raw.fasta
OUT=Data/raw/multispecies/strict_struct_negatives_v4.fasta
W=Data/raw/multispecies/external/deleak_work
mkdir -p "$W"

"$CDHIT" -i "$EXT" -i2 "$RAW" -o "$W/structneg_kept.fasta" \
         -c 0.80 -n 5 -M 0 -T 0 -d 0 > "$W/cdhit.log" 2>&1

cp "$W/structneg_kept.fasta" "$OUT"
echo "raw   : $(grep -c '>' "$RAW")"
echo "kept  : $(grep -c '>' "$OUT")  (-> $OUT)"
echo "dropped clusters (external + negative co-membership):"
grep -B1 hardneg_struct "$W/structneg_kept.fasta.clstr" | grep -E "label=pos|label=neg" || true
