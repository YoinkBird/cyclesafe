#!/usr/bin/env bash
# Purpose: quick and dirty test to verify files generated where expected

set -eu
workdir=$1; shift

file_list="$(ls -1 ${workdir})"
expected_files=(
  "gps_input_route.json"
  "gps_scored_route.json"
  "human_read_dectree.pkl"
)

echo "TEST: Verify Presence of Files after Running Model"
set -e
grep_opt_quiet=""
grep_opt_quiet="-q"
for efile in ${expected_files[@]}; do
  echo "${file_list}" | grep ${grep_opt_quiet} "${efile}"
done

echo "PASS"
