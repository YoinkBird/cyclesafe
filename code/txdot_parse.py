
from helpers import *
from feature_definitions import *
from sklearn.ensemble import RandomForestClassifier
from sklearn import (metrics, model_selection, linear_model, preprocessing, ensemble, neighbors, decomposition)
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import numpy as np
import pandas as pd
import pprint as pp
import re
#import xgboost as xgb
import os,sys

if(__name__ == '__main__'):
    # import the "crash" data
    datafile = "../data/txdot_2010_2017.csv"
    # import the "crash" data
    curdir=os.path.split(__file__)[0]
    datadir=os.path.split(curdir)[0] + "/data"
    datafile = "my_map_grid.csv"
    datafile = os.path.join(datadir, datafile)

def print_total(df, value):
#    print(value)
    # https://stackoverflow.com/a/26641085
    mask = df.select_dtypes([np.object]).apply(lambda x: x.str.contains(r"%s" % value, case=False, na=False))
    # name important for appending. src: https://stackoverflow.com/a/31292395
    ser = pd.Series(name=value)
    # find/count occurrences of string. src: https://stackoverflow.com/a/20076611
    for col in mask:
        # print(col)
        #print(mask[col].value_counts())
        # print(df[mask[col]][col].value_counts())
 #       print(df[mask[col]][col].count())
        tmpser = pd.Series(df[mask[col]][col].count(), index=[ col ], name=value)
        ser = ser.append(tmpser)
        ser.append(pd.Series(df[mask[col]][col].count(), index=[ col ]))
    return ser


def replace_with_npnan(data , printout=False):
    # convert to 'nan'
    nan_report_df = pd.DataFrame()

    verbose=0
    if(1):
        # standardise
        #------
        re_missing='no.*data'
        re_replace='no_data'
        if( verbose > 0):
            print("r'%s'" % re_missing)
        nan_report_df = nan_report_df.append(print_total(data, re_missing))
        for cat in data.columns:
            data[cat] = data[cat].apply(lambda x: x if type(x) != str else re.sub(r"%s" % re_missing, re_replace, x, flags=re.IGNORECASE))
        data.replace(to_replace=re_replace, value=np.nan, inplace=True)
        #---
        re_missing='unknown'
        re_replace='unknown'
        if( verbose > 0):
            print("r'%s'" % re_missing)
        nan_report_df = nan_report_df.append(print_total(data, re_missing))
        for cat in data.columns:
            data[cat] = data[cat].apply(lambda x: x if type(x) != str else re.sub(r"%s" % re_missing, re_replace, x, flags=re.IGNORECASE))
        data.replace(to_replace=re_replace, value=np.nan, inplace=True)
        #---
        re_missing='not.*reported'
        re_replace='not_reported'
        if( verbose > 0):
            print("r'%s'" % re_missing)
        nan_report_df = nan_report_df.append(print_total(data, re_missing))
        for cat in data.columns:
            data[cat] = data[cat].apply(lambda x: x if type(x) != str else re.sub(r"%s" % re_missing, re_replace, x, flags=re.IGNORECASE))
        data.replace(to_replace=re_replace, value=np.nan, inplace=True)
        #---
        re_missing='not.*applicable'
        re_replace='not_applicable'
        if( verbose > 0):
            print("r'%s'" % re_missing)
        nan_report_df = nan_report_df.append(print_total(data, re_missing))
        for cat in data.columns:
            data[cat] = data[cat].apply(lambda x: x if type(x) != str else re.sub(r"%s" % re_missing, re_replace, x, flags=re.IGNORECASE))
        data.replace(to_replace=re_replace, value=np.nan, inplace=True)
        #---
        #------

    data['speed_limit'].replace(0,np.nan,inplace=True) # speed_limit - np.nan is faster
    data['speed_limit'].replace(-1,np.nan,inplace=True) # speed_limit - np.nan is faster

    if(printout):
        print("txdot data undefined values report:")
        orig_val_expand_frame_repr = pd.get_option('display.expand_frame_repr')
        pd.set_option('display.expand_frame_repr', False)
        print(nan_report_df.transpose())
        pd.set_option('display.expand_frame_repr', orig_val_expand_frame_repr)
    return (data, nan_report_df)

# ensure all data is machine-readable
def clean_data(datafile, source='txdot', verbose=0):
    data = pd.read_csv(datafile,header=10)
    # feature definitions and properties
    featdef = get_feature_defs()
    if(verbose):
        print(list(data.columns))

    # try something interesting - track the "categories" of columns
    colgrps = {
        # relevant info for about intersections
        'intersection' : ['street_name','intersecting_street_name','intersection_related'],
      }
    # preprocessing
    # for column names: remove punctuation, then lowercase (src: http://stackoverflow.com/a/38931854)
    def process_cols(df):
      return df.columns.str.replace('[,\s()]+','_').str.lower()
    data.columns = process_cols(data)
    regexpunct = '[,\s()-]+'
    regexpunct_gps_safe = '[,\s()]+'
    def process_data_punctuation(df):
        # improved - convert if string (adapted from src: https://stackoverflow.com/a/22245305 )
        # for col in data: print(data[col].apply(lambda x: x if type(x) != str else x.lower()).unique())
        for category in list(featdef[featdef.type == 'str'].index):
           df[category] = df[category].str.replace(regexpunct,'_').str.lower()
        for category in list(featdef[featdef.type == 'street'].index):
           df[category] = df[category].str.replace(regexpunct,'_')
        return df
    data = process_data_punctuation(data)
    # validate string values - standardise to lowercase and remove punctuation
    #TODO: featdef_or_regex? see vvvbelowvvv for context# for cat in list(featdef[featdef.type != 'street'].index):
    for cat in data.columns:
        # exception for 'street name'; not used for model and values may rely on punctuation to remain unique
        #TODO: featdef_or_regex? see ^^^above^^^ for context#
        if ( re.match(r"%s" % ".*street_name.*", cat)):
            if(verbose):
                print("skipping %s" % cat)
            continue 
        if(verbose):
            print("category: %s" % (cat));
        # lowercase, replace things with '_'. can't use the var "regexpunct" because the '-' breaks gps vals as gps is encoded with '+'/'-'<value>
        #TODO-useTheVar: data[cat] = data[cat].apply(lambda x: x if type(x) != str else re.sub(regexpunct_gps_safe,'_',x).lower())
        data[cat] = data[cat].apply(lambda x: x if type(x) != str else re.sub('[,\s()]+','_',x).lower())
#        print( data[cat].map(lambda x: "" if type(x) != str else x).unique())
        if(verbose > 4):
             print( data[cat].value_counts() )
    # convert to 'nan'
    (data, nan_report_df) = replace_with_npnan(data, True)
    # special cases
    data.columns = data.columns.str.replace('crash_i_d', 'crash_id')
    # GPS coordinates were initially read in as string because missing entries were called 'No Data'
    data['latitude'] = data['latitude'].astype(float)
    data['longitude'] = data['longitude'].astype(float)

    # debugging info
    if(verbose > 2):
        for col in data.columns:
            if col in list(featdef[featdef.type == 'street'].index):
                print("skipping %s" % col)
                continue
            if ( re.match(r"%s" % ".*street_name.*", col)):
                print("skipping %s" % col)
                continue 
            print("column: %s" % (col));
            print(data[col].unique())

    return(data,featdef)

# catch-all pre-processing function
# e.g. to only have to call one function once from within the model
def preprocess_data(datafile, source='txdot', verbose=0):
    # first clean the data
    (data,featdef) = clean_data(datafile, source='txdot', verbose=0)
    # <crash_time>
    # convert integer crashtime to datetime with year
    data['crash_datetime'] = create_datetime_series(data)
    featdef = add_feature(featdef, 'crash_datetime', {'type':'datetime','regtype':'categorical'})
    # todo: plot by year, then by time.
    data['crash_time_dec'] = data.crash_datetime.apply(time_base10)
    featdef = add_feature(featdef, 'crash_time_dec', {'type':'float','regtype':'continuous'})
    featdef.set_value('crash_time_dec', 'origin', 'crash_datetime')
    # todo: figure out how to hist() with grouping by year, then group by time within each year
    # data.crash_time_dec.hist(bins=48)
    # if(showplt): 
    #   plt.show()
    data['crash_time_30m'] = data.crash_datetime.apply(time_round30min)
    featdef = add_feature(featdef, 'crash_time_30m', {'type':'24h','regtype':'categorical','origin':'crash_datetime'})
    # </crash_time>
    return(data,featdef)

# add categorical data
def preproc_add_bin_categories(data, featdef, verbose=0):
    if(1):
        # factorizable data
        # convert 'Wet' 'Dry' to '1' '0'
        data['surface_condition'] = data['surface_condition'].factorize()[0]
        # DOC: rename col http://stackoverflow.com/a/11346337
        data.rename(columns={'surface_condition':'surface_wet'})
        # print number of unique entries
        if(verbose > 0):
            for colname in data.columns:
                # np method uses 'size': print("% 4d : %s" % (data[colname].unique().size, colname))
                print("% 4d : %s" % (len(data[colname].unique()), colname))
        # remove data which is has no importance
        # better to drop cols with all NaN and convert "unimportant" data to NaN
        #  - can't universally decide to drop col just based on uniqueness
        # e.g. all of object_struck is 'Not Applicable' and useless, but if surface_condition had only one value "dry" this would be important
        # ? for colname in data.columns:
        # colname = 'object_struck'
        # if(len(data[colname].unique()) == 1):
        #   print("-I-: dropping %s for having all homogenous values %s", (colname, data[colname].unique()[0]))
        #   data.drop(colname,axis=1,inplace=True)

    # binary consolidate features for data exploration.
    # choose binary categories for categorical data to facilitate certain regression techniques
    # this reduces the dimensionality while maintaining control over the features
    # features from weak binary-categories get discarded, and the binary category is not used beyond data exploration!
    # in theory, only some categories will produce strong features
    # however, expanding all categories with e.g. get_dummies would lead to feature explosion, thus increasing the complexity of the dataset
    # this allows the general direction to be found, i.e. which categories can be ignored due to low correlation
    # the strong categories can then be expanded via get_dummies
    # tradeoff/caveat:
    # 1. if one category has one strong feature but many weak features, this technique can hide that feature (correlation will be low, but not zero)
    #    antidote: with enough time, model can be run with each omitted feature
    # 2. the binary re-classification will incorrectly bias the resulting model, and could potentially hide strong features 
    #    this should absolutely only be used for feature selection
    if(1):
        regexpunct = '[,\s()-]+'

        # Explanation: 'Crash Severity' contains the severity of injuries sustained in the crash
        # able to walk away: binary crash severity - was the person badly injured, or could they "walk away"
        category = 'crash_severity'
        bin_category = (category, "bin_%s" % category)
        data[bin_category[1]] = data[bin_category[0]].str.replace('[,\s()-]+','_').str.lower()
        # group ['non_incapacitating_injury','possible_injury','not_injured'] as 1, group ['incapacitating_injury','killed'] as 0, ignore 'unknown' as np.nan, 
        data[bin_category[1]].replace(['non_incapacitating_injury','possible_injury','not_injured'], 1, inplace=True)
        data[bin_category[1]].replace(['incapacitating_injury','killed'], 0, inplace=True)
        data[bin_category[1]].replace(['unknown'], np.nan, inplace=True)
        # track new feature
        featattr = dict(featdef.ix[bin_category[0]])
        featdef = add_feature(featdef, bin_category[1], {'type':'int','regtype':'bin_cat','target':featattr['target']})
        featdef.set_value(bin_category[1], 'origin', bin_category[0])

        # Explanation: 'Day of Week' is often thought of as "work week" + "weekend"
        # 1. combine Mo-Th for week, and Fr-Sun for weekend
        # 2. combine Mo-Fr@1600 for week, and Fr@1600-Sun for weekend
        # STUB - not sure if this one makes sense, Sunday night is not particularly wild

        # Explanation: 'Intersection Related' : categorize intersecton by how defensive a cyclist would have to ride
        # 'intersection' and 'intersection_related' are accidents resulting from intersection. in theory, cyclist could be defensive
        # 'non_intersection', 'driveway_access' are accidents without intersection or from a driveway.
        # non_intersection will likely be fault of motorized vehicle (cyclist could, of course, do a bad turn)
        # although driveway could be considered intersection, e.g. cyclist can slow down and watch for car at every driveway,
        # this is still likely the motorists fault -and- a cyclist cannot reasonably avoid a sideways collision
        # TODO: create category of "reasonable" intersection accidents - permute direction, turn, etc
        # 'not_reported' is to be ignored. only present once, and deciding which way undefinite data goes is a bad idea!
        # ['intersection_related', 'intersection',];['non_intersection', 'driveway_access',];['not_reported']
        # categorisation
        bin_category = 'bin_intersection_related'
        category = 'intersection_related'
        # 1 :
        bin_true = ['intersection_related', 'intersection',]
        # 0 :
        bin_false = ['non_intersection', 'driveway_access',]
        # nan :
        bin_znan = ['not_reported']
        data[bin_category] = data['intersection_related'].str.replace(regexpunct,'_').str.lower()
        data[bin_category].replace(bin_true,  1, inplace=True)
        data[bin_category].replace(bin_false, 0, inplace=True)
        data[bin_category].replace(bin_znan, np.nan, inplace=True)
        # track new feature
        bin_category = (category, "bin_%s" % category)
        featattr = dict(featdef.ix[bin_category[0]])
        featdef = add_feature(featdef, bin_category[1], {'type':'int','regtype':'bin_cat','target':featattr['target']})
        featdef.set_value(bin_category[1], 'origin', bin_category[0])

        # Explanation: 'Light Condition' can be reduce to "good visibility", "bad visibility"
        # ['dark_lighted', 'dark_not_lighted', 'dusk', 'dark_unknown_lighting', 'dawn',];['unknown',];['daylight']
        # categorisation
        # 0 :
        bin_category = 'bin_light_condition'
        category = 'light_condition'
        binfalse = ['dark_lighted', 'dark_not_lighted', 'dusk', 'dark_unknown_lighting', 'dawn',]
        #   note: this hides the effects of good lighting at night, but the definition of 'dark, not lighted' is not clear, 
        #   and it would require more investigation to determine which cyclists have adequate lighting
        # 1 :
        bintrue = ['daylight']
        # nan :
        binznan = ['unknown',]
        data[bin_category] = data['light_condition'].str.replace(regexpunct,'_').str.lower()
        data[bin_category].replace(bintrue,  1, inplace=True)
        data[bin_category].replace(binfalse, 0, inplace=True)
        data[bin_category].replace(binznan, np.nan, inplace=True)
        # track new feature
        bin_category = (category, "bin_%s" % category)
        featattr = dict(featdef.ix[bin_category[0]])
        featdef = add_feature(featdef, bin_category[1], {'type':'int','regtype':'bin_cat','target':featattr['target']})
        featdef.set_value(bin_category[1], 'origin', bin_category[0])

        # Explanation: Manner of Collision - direction of Units involved
        # motorist fault likelihood higher for "non changing" situations, e.g. if going straight
        # categorisation
        category = 'manner_of_collision'
        bin_category = (category, "bin_%s" % category)
        # 1 :
        bin_true = [
                 'one_motor_vehicle_going_straight',
                 'angle_both_going_straight',
                 'one_motor_vehicle_other',
                 'opposite_direction_one_straight_one_left_turn',
                 'same_direction_one_straight_one_stopped'
                 ]
        # 0 :
        bin_false = [
                 'one_motor_vehicle_backing',
                 'same_direction_both_going_straight_rear_end',
                 'opposite_direction_both_going_straight',
                 'one_motor_vehicle_turning_left',
                 'one_motor_vehicle_turning_right',
                 ]
        # nan :
        bin_znan =  ['']
        data[bin_category[1]] = data[bin_category[0]].str.replace('[,\s()-]+','_').str.lower()
        data[bin_category[1]].replace(bin_true,  1, inplace=True)
        data[bin_category[1]].replace(bin_false, 0, inplace=True)
        data[bin_category[1]].replace(bin_znan, np.nan, inplace=True)
        # track new feature
        featattr = dict(featdef.ix[bin_category[0]])
        featdef = add_feature(featdef, bin_category[1], {'type':'int','regtype':'bin_cat','target':featattr['target']})
        featdef.set_value(bin_category[1], 'origin', bin_category[0])

    return(data,featdef)

'''
approach:
    0. work with pre-processed data (txdot_parse.preprocess_data)
    1. identify all entries for intersections (street_name, intersecting_street_name) with no speed_limit
        => data2
    2. get all available data for these intersections
        => df3
    3. 
'''
'''
profiling:
    IPython CPU timings (estimated):
      User   :       4.62 s.
      System :       0.28 s.
    Wall time:       5.53 s.
'''
# assume already processed
def impute_mph(data, verbose=0):
    verbose3 = 0
    if(verbose > 2):
      verbose3 = 1
    if(verbose):
        print("-I-: using verbosity %d" % (verbose))
    colgrps = {
        # relevant entries for intersections
        'intersection' : ['street_name','intersecting_street_name'],
        'inter_mph' : ['speed_limit','crash_year','street_name','intersecting_street_name'],
        # speed limit changes over the years
        'inter_mph_uniq' : ['speed_limit','crash_year','street_name','intersecting_street_name'],
      }
    # impute missing speed limits

    if(verbose):
        print("total missing speed limit data:\n %s" % (data[~(data['speed_limit']).isnull()].shape[0]))
    # df of all intersections and relvant data - keep only attributes which identify unique intersections and speed limits
    df_inter = data[(~ data.intersecting_street_name.isnull())][colgrps['inter_mph_uniq']]
    num_inter = df_inter.shape[0]
    if(verbose):
        print("total intersections:\nin : %s\nout: %s" % (data.shape[0], df_inter.shape[0]))
    df_inter.drop_duplicates(subset=colgrps['inter_mph_uniq'], inplace=True)
    if(verbose):
        print("deduped:\nout: %s" % (df_inter.shape[0]))

    # df of intersections without speed limits, to be used as an "index" to examine each intersection
    df_inter_nomph = df_inter[(df_inter.speed_limit.isnull())][colgrps['intersection']].drop_duplicates()
    if(verbose):
        print("intersections without speed_limit:\nout: %s" % (df_inter_nomph.shape[0]))
    df_inter_nomph.reset_index(drop=True, inplace=True)

    # TODO: include crash id
    # stub: data[(data.street_name == ser['street_name']) & (data.intersecting_street_name == ser['intersecting_street_name'])][colgrps['inter_mph']]
    # loop through list of intersections with a missing entry for speed_limit
    # get all entries for these intersections
    # impute
    # TODO: is this a case for a pivot_table ?
    for i,ser in df_inter_nomph.iterrows():
      if(verbose > 1):  #verbose>=2
          print("%d / %d ..." % (i, df_inter_nomph.shape[0]))
      #if (i != 3): continue # skip first 2, they are confirmed boring
      if(verbose3):
          print("%s | %s | %s" % (i, ser.street_name, ser.intersecting_street_name))
      #print(data[(data['street_name'] == ser['street_name'])]) 
      # sometimes the updates to 'dftmp' happen to 'data', something about slices vs copies, blah
      # A value is trying to be set on a copy of a slice from a DataFrame
      # See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      # disable with: (not a good idea!)
      #pd.set_option('mode.chained_assignment',None)
      #pd.set_option('mode.chained_assignment','raise')
      # 
      dftmp = data[(data['street_name'] == ser['street_name']) & (data['intersecting_street_name'] == ser['intersecting_street_name'])].copy()
      if(verbose3):
          print("before:")
          print(dftmp['speed_limit'].unique())
          print(dftmp[colgrps['inter_mph']])
      # sequence: fill back (misses a missing "last" entry), then forward (in order to get the missing "last" entry
      # backwards fill - BIAS towards future speed limits (often higher!)
      dftmp['speed_limit'].fillna(method='bfill',inplace=True)
      # forwards fill - BIAS towards past speed limits (often lower!)
      dftmp['speed_limit'].fillna(method='ffill',inplace=True)
      if(verbose3):
          print("after:")
          print(dftmp[colgrps['inter_mph']])
      # write
      for jq in dftmp.index:
        if(verbose3):
            print("index %d" % jq)
            print(data.ix[jq].speed_limit)
        tmplim = dftmp.ix[jq].speed_limit
        if(np.isnan(data.ix[jq].speed_limit)): # | (data.ix[jq].speed_limit == -1) ):
            if(not np.isnan(tmplim) ):
                data.set_value(jq,'speed_limit',dftmp.ix[jq].speed_limit)
        if(verbose3):
            print(data.ix[jq].speed_limit)
      #if (i == 6): break # quit after 6 loops
    if(verbose):
        print("total new missing speed limit data:\n %s" % (data[data['speed_limit'].isnull()].shape[0]))
    return data

# def get_abt(data):
# data quality report
def get_data_quality_report(data):
    # TEMP
    # 3.1 data quality report
    # 'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max' included in DataFrame.describe()
    # '%mising', 'cardinality' not in DataFrame.describe()
    # pandas=>book equivalents: '25%' equ '1st qrt', '50%' ?equ median?, '75%' equ '3rd qrt', 'max' is_same
    #
    # pandas order + book vals:
    # 'count', '%mising', 'cardinality', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'
    # book order:
    # 'count',  '%mising',  'cardinality',  'min',  '25%',  'mean',  '50% (median)',  '75%',  'max',  'std', 
    print("data quality report for %s" % 'txdot crash data')
    df_quality_rpt = data.describe(exclude=['datetime64']).transpose()
    # adding things
    # eg. df_quality_rpt.ix['speed_limit'].append(pd.Series('hi'), index="new"))
    newvals=pd.DataFrame()
    for row in df_quality_rpt.index:
        # print(row);
        percent_missing = (100 * data[row].isnull().sum()) / data.shape[0]
        # print(percent_missing)
        tmpser = (pd.Series({'%missing': percent_missing ,'cardinality': data[row].unique().size}))
        tmpser.name=row
        newvals = newvals.append(tmpser)
        # df_quality_rpt.append(tmpser)
    # cardinality is integer
    newvals['cardinality'] = newvals['cardinality'].astype('int')
    # column order:
    # 0th is 'count' followed by newvals (missing,cardinality) 
    # 1th+ is mean,std, etc
    colorder = [df_quality_rpt.columns[0]] + newvals.columns.values.tolist() + df_quality_rpt.columns[1:].tolist()
    colorder = ['count','%missing','cardinality','mean','std','min','25%','50%','75%','max','freq','top'] # ,'unique'
    # add the new columns in correct order
    df_quality_rpt = df_quality_rpt.join(newvals)[colorder]
    return df_quality_rpt

def print_get_data_quality_report(df_quality_rpt):
    # df_quality_rpt = get_data_quality_report(data)
    orig_val_expand_frame_repr = pd.get_option('display.expand_frame_repr')
    pd.set_option('display.expand_frame_repr', False)
    # pd.set_option('max_colwidth', 15)
    pd.set_option('precision', 4)
    print("pandas Describe + %missing + cardinality")
    print(df_quality_rpt)
    pd.set_option('display.expand_frame_repr', orig_val_expand_frame_repr)

# run for testing purposes
if(__name__ == '__main__'):
    print("-I-: Self-Test: txdot_parse.py")
    print("-I-: should not see this from an import!")
    test_impute_mph = 1
    test_dummies = 1
    show_data_vis = 1
    show_data_vis_model = 1

    print("-I-: load file")
    # get data
    (data,featdef) = preprocess_data(datafile, verbose=1)

    print("-I-: binary categories")
    # add binary categories
    (data,featdef) = preproc_add_bin_categories(data, featdef, verbose=1)

    # data quality report
    df_quality_rpt = get_data_quality_report(data)
    print_get_data_quality_report(df_quality_rpt)

    # save to file
    with open ('data_quality_report.html', 'w') as outfile: df_quality_rpt.fillna('').to_html(outfile)



    # impute speed_limit mph
    if(test_impute_mph):
        print("########################################")
        # 35.0    638
        # 30.0    549
        # 45.0    187
        # 40.0    184
        # 55.0     53
        # 25.0     48
        # 50.0     41
        # 60.0     18
        # 65.0      8
        # 10.0      8
        # 5.0       6
        # 20.0      5
        # 15.0      5
        # 32.0      1
        # 36.0      1
        print(data.speed_limit.value_counts())
        totalmissing   = data[data['speed_limit'].isnull()].shape[0]
        missingpercent = totalmissing / data.shape[0]
        print("pre : total missing speed limit data:\n %s (%s of 1)" % (totalmissing, missingpercent))
        print(data.speed_limit.unique())
        data = impute_mph(data, verbose=0)
        totalmissing   = data[data['speed_limit'].isnull()].shape[0]
        missingpercent = totalmissing / data.shape[0]
        print("post: total missing speed limit data:\n %s (%s of 1)" % (totalmissing, missingpercent))
        print(data.speed_limit.unique())
        print(data.speed_limit.value_counts())
        print(data.info())
        print("########################################")
    print("-I-: End of Pre-Processing")
    print("-I-: Brief data exploration")


    if(show_data_vis):
        print("########################################")
        print(data.head())
        print(data.info())
        if(1):
          data.describe()
          data.hist()
          data.corr().plot() # TODO: seaborn
          plt.show()
        else:
          print("-I-: Skipping...")
        print("########################################")

    if(test_dummies):
        print("########################################")
        # list of vars which become dummie'd
        dummies_needed_list = list(featdef[featdef.dummies == 1].index)

        # dummies
        # http://stackoverflow.com/a/36285489 - use of columns
        data_dummies = pd.get_dummies(data, columns=dummies_needed_list)
        # no longer need to convert headers, already done in process_data_punctuation
        pp.pprint(list(data_dummies))

    if(show_data_vis_model):
        # alternative visualisation
        datapt = data.pivot_table(values=['crash_death_count','crash_incapacitating_injury_count','crash_non-incapacitating_injury_count'], index=['speed_limit','crash_time'])
        print(datapt)


        # inspect features with high covariance
        pairplot_bin_var_list = list(featdef[featdef['pairplot']].index)
        if(0):
            sns.pairplot(data, vars=pairplot_var_list)
            plt.show()
        print("########################################")

    print("-I-: End of File")


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
