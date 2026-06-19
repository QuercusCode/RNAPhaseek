#!/usr/bin/env bash
# Archive the FULL project to the external LaCie, then slim the internal copy by
# removing ONLY the regenerable FEGS cache (Data/processed, ~278G) and symlinking it
# to the external archive. Internal keeps the model + code + raw FASTAs + splits + docs
# (~14G) so the model stays always-usable; the external holds the complete archive for
# future training/reproduction.
#
# SAFETY: the internal Data/processed is deleted ONLY after the external archive passes
# (a) file-count match, (b) size match, and (c) sample-checksum — and even then it is
# regenerable from the internal FASTAs. Any failure aborts WITHOUT deleting anything.
set -u
SRC=/Users/synbaiteam/Documents/RNAPhaseek_scripts
DEST=/Volumes/LaCie/RNAPhaseek_scripts
LOG=$SRC/split_to_external.log
exec >> "$LOG" 2>&1
echo "================ START $(date) ================"

[ -d /Volumes/LaCie ] || { echo "FATAL: /Volumes/LaCie not mounted — abort"; exit 1; }
mkdir -p "$DEST" || { echo "FATAL: cannot create $DEST — abort"; exit 1; }

echo "[1/4] rsync full project -> external (hours; exFAT-friendly flags) ..."
rsync -rltD --modify-window=2 --no-perms --no-owner --no-group --delete \
      --exclude '__pycache__' --exclude '.DS_Store' --exclude 'split_to_external.log' \
      "$SRC/" "$DEST/"
RS=$?
echo "[1/4] rsync exit=$RS"
[ $RS -eq 0 ] || { echo "FATAL: rsync nonzero exit ($RS) — NOT deleting anything"; exit 1; }

echo "[2/4] verify file counts of Data/processed ..."
SRC_N=$(find "$SRC/Data/processed" -type f 2>/dev/null | wc -l | tr -d ' ')
DST_N=$(find "$DEST/Data/processed" -type f 2>/dev/null | wc -l | tr -d ' ')
echo "      internal=$SRC_N  external=$DST_N"
[ "$SRC_N" -gt 0 ] && [ "$SRC_N" = "$DST_N" ] || { echo "FATAL: file-count mismatch — NOT deleting"; exit 1; }

echo "[2b] verify total size (du, GiB) ..."
SRC_SZ=$(du -sk "$SRC/Data/processed" | awk '{print int($1/1048576)}')
DST_SZ=$(du -sk "$DEST/Data/processed" | awk '{print int($1/1048576)}')
echo "      internal=${SRC_SZ}G  external=${DST_SZ}G"
DIFF=$(( SRC_SZ > DST_SZ ? SRC_SZ-DST_SZ : DST_SZ-SRC_SZ ))
[ "$DIFF" -le 3 ] || { echo "FATAL: size differs by ${DIFF}G (>3G) — NOT deleting"; exit 1; }

echo "[3/4] sample-checksum 6 npz across the cache ..."
FAIL=0
for f in $(find "$SRC/Data/processed" -name '*.npz' 2>/dev/null | awk 'NR%50000==1' | head -6); do
  rel=${f#"$SRC"/}
  a=$(shasum "$f" 2>/dev/null | awk '{print $1}')
  b=$(shasum "$DEST/$rel" 2>/dev/null | awk '{print $1}')
  if [ -n "$a" ] && [ "$a" = "$b" ]; then echo "      OK  $rel"; else echo "      MISMATCH $rel"; FAIL=1; fi
done
[ $FAIL -eq 0 ] || { echo "FATAL: checksum mismatch — NOT deleting"; exit 1; }

echo "[4/4] verification PASSED. Slimming internal: remove regenerable Data/processed + symlink to archive ..."
rm -rf "$SRC/Data/processed"
ln -s "$DEST/Data/processed" "$SRC/Data/processed"
echo "      internal Data/processed -> symlink to $DEST/Data/processed"
echo "FREED:"; df -h "$SRC" | awk 'NR==2{print "  internal now "$4" free ("$5" used)"}'
echo "================ DONE $(date) ================"
