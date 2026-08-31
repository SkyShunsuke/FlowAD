#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: $0 'YYYY-MM-DD HH:MM'"
  echo "Example: $0 '2026-08-30 06:00'"
  exit 1
fi

TARGET_TIME="$1"

# 指定時刻をUNIX時間に変換
TARGET_EPOCH=$(date -d "$TARGET_TIME" +%s 2>/dev/null || true)

if [ -z "${TARGET_EPOCH}" ]; then
  echo "ERROR: 時刻の形式が不正です: $TARGET_TIME"
  echo "Example: '2026-08-30 06:00'"
  exit 1
fi

NOW_EPOCH=$(date +%s)

if [ "$TARGET_EPOCH" -le "$NOW_EPOCH" ]; then
  echo "ERROR: 指定時刻は現在時刻より未来にしてください: $TARGET_TIME"
  exit 1
fi

SLEEP_SEC=$((TARGET_EPOCH - NOW_EPOCH))

echo "Target time : $TARGET_TIME"
echo "Current time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Sleep       : ${SLEEP_SEC} sec"
echo "User        : ${USER}"
echo

sleep "$SLEEP_SEC"

echo "==== $(date '+%Y-%m-%d %H:%M:%S') ===="
echo "Checking PBS jobs for user: ${USER}"

# qselect が使える場合は qselect を優先
if command -v qselect >/dev/null 2>&1; then
  mapfile -t JOB_IDS < <(qselect -u "$USER" 2>/dev/null || true)
else
  # qselect がない環境用: qstat の表から自分のジョブIDだけを抽出
  mapfile -t JOB_IDS < <(
    qstat 2>/dev/null | awk -v user="$USER" '
      $1 ~ /^[0-9]/ && $3 == user { print $1 }
    '
  )
fi

if [ "${#JOB_IDS[@]}" -eq 0 ]; then
  echo "No jobs found."
  exit 0
fi

echo "Jobs to delete:"
printf '  %s\n' "${JOB_IDS[@]}"
echo

for jid in "${JOB_IDS[@]}"; do
  echo "qdel $jid"
  qdel "$jid" || echo "WARNING: failed to delete $jid"
done

echo
echo "Done."