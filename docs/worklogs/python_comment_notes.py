

"""
txdot_parse.py
"""
# miscellaneous
'''
# look into dictvectorizer dv.get_feature_names http://stackoverflow.com/a/34194521
'''
'''
pandas tricks
filtering
http://stackoverflow.com/a/11872393
# select data with average_daily_traffic_amount but intersecting_street_name null
# => busy roads without an intersection
data[~data['average_daily_traffic_amount'].isnull() & data['intersecting_street_name'].isnull()]

# select intersection_related == 'Non Intersection' and intersecting_street_name null
# => verify whether intersecting_street_name==null indicates that there is no intersection
# => then only display the columns pertaining to street names
data[(data['intersection_related'] == 'Non Intersection') & data['intersecting_street_name'].isnull()][['street_name','intersecting_street_name','intersection_related']]

data[(data['intersection_related'] == 'Non Intersection') & data['intersecting_street_name'].isnull()][colgrps['intersection']]
'''
# if total person count is needed
#  print("-I-: creating total involved count")
#  individual_counts = [
#    'crash_death_count',
#    'crash_incapacitating_injury_count',
#    'crash_non-incapacitating_injury_count'
#    'crash_not_injured_count',
#    'crash_possible_injury_count',
#  ]
#  total_count = pd.Series(index=mapdf.index,dtype=int)
#  for icount in individual_counts:
#    total_count += mapdf[icount]
##    mapdf['person_number'] = mapdf['person_number'] + mapdf[icount]
