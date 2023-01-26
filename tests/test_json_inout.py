import json
import sys
sys.exit("STUB FILE")

'''
diff tests/route_json/gps_scored_eerc_to_klane.json output/gps_scored_route.json 
'''
filepath1 = "tests/route_json/gps_scored_eerc_to_klane.json"
filepath2 = "output/gps_scored_route.json"

loaded1a = {}
with open ( filepath1, 'r' ) as infile:
    loaded1 = json.load(infile)

loaded2a = {}
with open ( filepath2, 'r' ) as infile:
    loaded2 = json.load(infile)

import pprint as pp
pp.pprint(loaded1)

pp.pprint(loaded2)

# TODO: load in pandas, drop the scores, and diff
