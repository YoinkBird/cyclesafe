#!/usr/bin/env bash
# Purpose: verify that model generates expected output files

set -eu

echo "TEST: Run Model, Verify Presence of Generated Files"
# options
workspace=${1:-"E: pass in path to workspace"}; shift;

# run from top-level
scriptdir="$(dirname "${BASH_SOURCE[0]}")"
cd="$(dirname "${scriptdir}")"

# run model
# override workspace, override path to input route file from non-workspace path (e.g. for testing purposes)
python3 ./modelmanager/model.py --workspace="${workspace}" --routefile=./tests/route_json/gps_generic.json --dumpresponse="gps_scored_route.json" --datafile ./data/txdot_cris_20170427.csv

./tests/check_file_existence.sh "${workspace}"
