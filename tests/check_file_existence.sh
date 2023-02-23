#!/usr/bin/env bash
# Purpose: check for expected model output files
# CAVEAT: functionality currently duplicated in another script in this dir

set -eu

echo "TEST: Verify Presence of Files expected to be generated after independently Running Model"
# options
workspace=${1:-"E: pass in path to workspace"}; shift;

# verify
expected_files=(
  "gps_scored_route.json"  # --dumpresponse="gps_scored_route.json"
  "human_read_dectree.pkl" # see modelmanager/model.py
)

if [[ ${#} -ne 0 ]]; then
  echo "TEST: override default file list"
  expected_files=(${@})
fi

set -e
for efile in ${expected_files[@]}; do
  test -e "${workspace}/${efile}"
done

echo "PASS"
