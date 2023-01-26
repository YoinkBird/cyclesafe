#!/usr/bin/env bash
# Purpose: verify that model generates expected output files

set -eu

# options
workspace=$1; shift;

# run from top-level
scriptdir="$(dirname "${BASH_SOURCE[0]}")"
cd="$(dirname "${scriptdir}")"

# run model
# override workspace, override path to input route file from non-workspace path (e.g. for testing purposes)
python3 ./modelmanager/model.py --workspace="${workspace}" --routefile=./tests/route_json/gps_generic.json --dumpresponse="gps_scored_route.json" --datafile ./data/txdot_cris_20170427.csv

# verify
file_list="$(ls -1 ${workspace})"
expected_files=(
  "gps_scored_route.json"  # --dumpresponse="gps_scored_route.json"
  "human_read_dectree.pkl" # see modelmanager/model.py
)

echo "TEST: Verify Presence of Files after Running Model"
set -e
for efile in ${expected_files[@]}; do
  test -e "${workspace}/${efile}"
done

echo "PASS"
