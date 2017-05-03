
from helpers import *
from feature_definitions import *
from txdot_parse import *
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

# import the "crash" data
datafile = "my_map_grid.csv"

# get clean data
(data,featdef) = preprocess_data(datafile)

# add binary categories
(data,featdef) = preproc_add_bin_categories(data, featdef, verbose=1)


show_data_vis = 0
show_data_piv = 0
if(show_data_vis):
    print("########################################")
    print(data.head())
    print(data.info())
    if(1):
      data.describe()
      data.hist()
      data.corr().plot() # TODO: seaborn
      plt.show()
      # inspect features with high covariance
      pairplot_bin_var_list = list(featdef[featdef['pairplot']].index)
      if(0):
          sns.pairplot(data, vars=pairplot_var_list)
          plt.show()
    else:
      print("-I-: Skipping...")
    print("########################################")

if(show_data_piv):
    # alternative visualisation
    datapt = data.pivot_table(values=['crash_death_count','crash_incapacitating_injury_count','crash_non-incapacitating_injury_count'], index=['speed_limit','crash_time'])
    print(datapt)


if(0):
    # list of vars which become dummie'd
    dummies_needed_list = list(featdef[featdef.dummies == 1].index)

    # dummies
    # http://stackoverflow.com/a/36285489 - use of columns
    data_dummies = pd.get_dummies(data, columns=dummies_needed_list)
    # no longer need to convert headers, already done in process_data_punctuation
    pp.pprint(list(data_dummies))

# dummies - new method
(data_dummies,featdef) = featdef_get_dummies(data,featdef)

# verify
validpreds = len(list(featdef[(featdef.type == 'int') & (featdef.target != True)].index))
validtargs = len(list(featdef[(featdef.type == 'int') & (featdef.target == True)].index))
invalfeats = len(list(featdef[(featdef.type != 'int') & (featdef.dummies != True)].index))
alldummies = data_dummies.shape[1]
print("valid+string %d / %d total" % (validpreds+validtargs+invalfeats, alldummies))
# any non-dummy & integer type
if(0):
    print(data_dummies[list(featdef[(featdef.dummies == False) & (featdef.type == 'int')].index)].info())

# mainly integer data
data_int_list = list(featdef[(featdef.dummies == False) & (featdef.type == 'int')].index)
df_int = data_dummies[list(featdef[(featdef.dummies == False) & (featdef.type == 'int')].index)]
df_int_nonan = df_int.dropna()

# pca stub
# pca = decomposition.PCA(svd_solver='full')
# pca.fit(pd.get_dummies(data[dummies_needed_list])).transform(pd.get_dummies(data[dummies_needed_list]))

print("-I-: train-test split")

if(1):
    # TODO : create a df of featdef with these attributes
    predictors  = list(featdef[(featdef.regtype != False) & (featdef.type == 'int') & (featdef.target != True) & (featdef.dummies == False)].index)
    responsecls = list(featdef[(featdef.regtype != False) & (featdef.type == 'int') & (featdef.target == True) & (featdef.dummies == False) & (featdef.regtype == 'bin_cat')].index)

    if(1):
        print("predictors:")
        print(predictors)
        print("responsecls:")
        print(responsecls)
    testsize = 0.3
    # data_nonan = data[ predictors + responsecls ].dropna()
    data_nonan = df_int_nonan
    X_train, X_test, y_train, y_test = model_selection.train_test_split(data_nonan[predictors],data_nonan[responsecls], test_size=testsize)

    from sklearn import tree
    clf = tree.DecisionTreeClassifier() #max_depth = 5)
    #clf.fit(X_train,y_train)
    clf.fit(data_nonan[predictors],data_nonan[responsecls])

    # prediction and scoring
    print("-I-: cross_val_score on train (itself)")
    print(model_selection.cross_val_score(clf, X_train, y_train.values.ravel()))
    # TODO: how to use multioutput-multioutput?
    # vvv multiclass-multioutput is not supported vvv
    # print(model_selection.cross_val_score(clf, X_train, y_train))
    y_pred = clf.predict_proba(X_test)
    print("-I-: cross_val_score against test")
    print(model_selection.cross_val_score(clf, X_test, y_test.values.ravel()))

    # print important features
    print("-I-: most important features:")
    clf_imp_feats = print_model_feats_important(clf, predictors)
    # pie chart - not useful here
    # clf_imp_feats.value_counts().plot(kind='pie');plt.show()

    # bar chart
    ax = get_ax_bar(clf_imp_feats, title="DecisionTree Important Features")
    plt.show()
    # horizontal bar chart
    ax = get_ax_barh(clf_imp_feats, title="DecisionTree Important Features")
    plt.show()

    # plot important features
    alreadyseen = {}
    for i in np.argsort(clf.feature_importances_)[::-1]:
      feat = predictors[i]
      #feat = predictors[i].replace('bin_','')
      pltkind = 'pie'
      print("%s" % ( feat))
      if(featdef.ix[feat].origin):
          feat_orig = featdef.ix[predictors[i]].origin
          #print("testing %s - %s" % ( feat_orig, feat))
          if(not feat_orig in alreadyseen):
              alreadyseen[feat_orig] = 1
              data[feat_orig].value_counts().plot(kind=pltkind, title="%s - original values for %s" % (feat_orig, feat))
          #else:
              #print("%d for %s - skipping %s" % (alreadyseen[feat_orig], feat_orig, feat))
      else:
          data[feat].value_counts().plot(kind=pltkind, title="%s " % (feat))
      plt.show()



# predictors  = list(featdef[(featdef.regtype == 'bin_cat') & (featdef.target != True)].index)
# responsecls = list(featdef[(featdef.regtype == 'bin_cat') & (featdef.target == True)].index)
predictors = [
# 'crash_time',
# 'crash_time_dec',
 'bin_intersection_related',
 'bin_light_condition',
 'bin_manner_of_collision',
 ]
responsecls = [
 'bin_crash_severity'
 ]
testsize = 0.3
data_nonan = data[ predictors + responsecls ].dropna()
X_train, X_test, y_train, y_test = model_selection.train_test_split(data_nonan[predictors],data_nonan[responsecls], test_size=testsize)

from sklearn import tree
clf = tree.DecisionTreeClassifier() #max_depth = 5)
clf.fit(X_train,y_train)

# prediction and scoring
print("-I-: cross_val_score on train (itself)")
print(model_selection.cross_val_score(clf, X_train, y_train.values.ravel()))
y_pred = clf.predict_proba(X_test)
print("-I-: cross_val_score against test")
print(model_selection.cross_val_score(clf, X_test, y_test.values.ravel()))

# DOC: How to interpret decision trees' graph results and find most informative features?
# src: http://stackoverflow.com/a/34872454
print("-I-: most important features:")
for i in np.argsort(clf.feature_importances_)[::-1]:
  print("%f : %s" % (clf.feature_importances_[i],predictors[i]))

# plotting important features
for i in np.argsort(clf.feature_importances_)[::-1]:
  feat = predictors[i]
  feat = predictors[i].replace('bin_','')
  pltkind = 'pie'
  if(featdef.ix[feat].origin):
      feat_orig = featdef.ix[predictors[i]].origin
      data[feat].value_counts().plot(kind=pltkind, title="%s - original values for %s" % (feat_orig, feat))
  else:
      data[feat].value_counts().plot(kind=pltkind, title="%s " % (feat))
  plt.show()

print("time of day:")
ax_time = get_ax_time(
        interval = '24h',
        title = 'Frequency of Bike Crashes For Time of Day (2010-2017)',
        xlabel = 'Time of Day (24 hr)',
        ylabel = 'count',
        )
data.crash_time.hist(bins=48,ax=ax_time)
plt.show()
# data.crash_time_30m.value_counts(sort=False).plot(kind='pie');plt.show()
# /plotting important features

# display tree criteria
# src: http://scikit-learn.org/stable/modules/tree.html#classification
from IPython.display import Image
# pydot plus had to be installed as python -m pip
# src : http://stackoverflow.com/a/42469100
import pydotplus
dot_data = tree.export_graphviz(clf, out_file=None,
        feature_names=predictors,
        class_names=['0']+responsecls, # seems to require at least two class names
        rounded=True,
        filled=True,
        # proportion = True,  : bool, optional (default=False) When set to True, change the display of ‘values’ and/or ‘samples’ to be proportions and percentages respectively.

        )
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png() , retina=True)
print("-I-: if img doesn't show, run \n Image(pydotplus.graph_from_dot_data(dot_data).create_png() , retina=True)")
print("-I-: End of File")


# miscellaneous
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

'''
# look into dictvectorizer dv.get_feature_names http://stackoverflow.com/a/34194521
'''
# DOC
# feature importance and feature selection
# e.g. reducing complexity of a tree model
# https://www.analyticsvidhya.com/blog/2016/12/introduction-to-feature-selection-methods-with-an-example-or-how-to-select-the-right-variables/
# 
# automatically discarding low-importance features
# http://scikit-learn.org/stable/modules/feature_selection.html#feature-selection-using-selectfrommodel

# Interpreting Decision Tree in context of feature importances
# https://datascience.stackexchange.com/questions/16693/interpreting-decision-tree-in-context-of-feature-importances

# further reading - Random Forests
# http://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm

# xgboost on ubuntu
# manually link
#   https://askubuntu.com/a/764572
# conda install libcc
# https://github.com/dmlc/xgboost/issues/1043
