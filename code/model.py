
from helpers import *
from feature_definitions import *
from txdot_parse import *
from sklearn.ensemble import RandomForestClassifier
from sklearn import (metrics, model_selection, linear_model, preprocessing, ensemble, neighbors, decomposition)
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import numpy as np
import pandas as pd
import pprint as pp
import re
#import xgboost as xgb

import os,sys
# import the "crash" data
curdir=os.path.split(__file__)[0]
datadir=os.path.split(curdir)[0] + "/data"
datafile = "my_map_grid.csv"
datafile = os.path.join(datadir, datafile)
if(0):
  print(__file__)
  pp.pprint(sys.path)


# get clean data
(data,featdef) = preprocess_data(datafile)

# add binary categories
(data,featdef) = preproc_add_bin_categories(data, featdef, verbose=1)

def dectree_evaluate_cv_strategy(X_full, y_full):
  # Recursive Feature Elimination CV
  # http://scikit-learn.org/stable/auto_examples/feature_selection/plot_rfe_with_cross_validation.html#sphx-glr-auto-examples-feature-selection-plot-rfe-with-cross-validation-py
  from sklearn.model_selection import StratifiedKFold,GroupKFold
  from sklearn.feature_selection import RFECV
  from sklearn import tree
  clf = tree.DecisionTreeClassifier(random_state = 42) #max_depth = 5)

  # ### Choosing the Cross Validation iterator
  # #### GroupKFold Consideration
  # http://scikit-learn.org/stable/modules/cross_validation.html#group-k-fold
  # No need for GroupKFold:  "For example if the data is obtained from different subjects with several samples per-subject"
  # => only one entry per accident
  # cvFold = GroupKFold(3)

  # settling on ...
  cvFold = StratifiedKFold

  # ### Choosing the scoring parameter
  # The "accuracy" scoring is proportional to the number of correct classifications
  # 'accuracy' leads to only one feature
  rfecv = RFECV(estimator=clf, step=1, cv=cvFold(2), scoring='accuracy')
  # proof:
  scorer = 'accuracy'
  print("-I-: test scores for arbitrary depth using %s" % scorer)
  accuracy_scores = []
  for i in range(0,10):
    rfecv.fit(X_full,y_full.values.ravel())
    accuracy_scores.append(rfecv.n_features_)
    #print("Optimal number of features [depth:None][scoring:%s]: %d" % (scorer, rfecv.n_features_))
  print("[depth:00][scoring:%s] avg score: %f , std: %f, med: %f" % (scorer, np.mean(accuracy_scores), np.std(accuracy_scores), np.median(accuracy_scores)))

  # ### 'ROC AUC' more appropriate for classification
  scorer = 'roc_auc'
  print("-I-: test scores for different depths using %s" % scorer)
  roc_auc_scores = []
  for depth in range (1,51,5):
    clf = tree.DecisionTreeClassifier(random_state = 42, max_depth = depth)
    rfecv = RFECV(estimator=clf, step=1, cv=cvFold(2), scoring='roc_auc')
    # proof:
    for i in range(0,10):
      rfecv.fit(X_full,y_full.values.ravel())
      roc_auc_scores.append(rfecv.n_features_)
      #print("Optimal number of features [depth:%d][scoring:%s]: %d" % (depth,scorer, rfecv.n_features_))
    print("[depth:%02d][scoring:%s] avg score: %f , std: %f, med: %f" % (depth, scorer, np.mean(roc_auc_scores), np.std(roc_auc_scores), np.median(roc_auc_scores)))

  # "optimal"  number fluctuates wildly - 29,17,4, etc
  scorer = 'roc_auc'
  print("-I-: test scores for arbitrary depth using %s" % scorer)
  rfecv = RFECV(estimator=clf, step=1, cv=cvFold(2), scoring='roc_auc')
  # proof:
  roc_auc_scores = []
  for i in range(0,10):
    rfecv.fit(X_full,y_full.values.ravel())
    roc_auc_scores.append(rfecv.n_features_)
    #print("Optimal number of features [scoring:%s]: %d" % (scorer, rfecv.n_features_))
  print("[depth:00][scoring:%s] avg score: %f , std: %f, med: %f" % (scorer, np.mean(roc_auc_scores), np.std(roc_auc_scores), np.median(roc_auc_scores)))


def run_cross_val(data_dummies,featdef,dropfeatures=[]):
    print("-I-: creating new dataset without %s" % dropfeatures)
    # mainly integer data
    # TODO: evaluate whether to  move the validfeats higher up during the nonan phase
    # TODO - change crash_year to regtype == False

    # valid features - defined for regression (aka classification), are integers (because I'm really into that), and ain't supposed to be no dummy (entries meant to be encoded as dummies)
    # 'regtype ! = False' mainly for crash_id
    validfeats = featdef[(featdef.dummies == False) & (featdef.type == 'int')]
    # validfeats = validfeats.drop(['average_daily_traffic_amount','average_daily_traffic_year'])# inplace copy - , inplace=True)
    # validfeats = validfeats.drop(['crash_year'])# inplace copy - , inplace=True)
    if(len(dropfeatures) >0):
        validfeats = validfeats.drop(dropfeatures)
    ## print("-I-: verify validfeats")
    ## print(validfeats[validfeats.dummies])
    data_int_list = list(validfeats.index)
    df_int = data_dummies[list(validfeats.index)]
    # Avoid: ValueError: Input contains NaN, infinity or a value too large for dtype('float64').
    df_int_nonan = df_int.dropna()
    print("NaN handling: Samples: NaN data %d / %d fullset => %d newset" % ( (df_int.shape[0] - df_int_nonan.shape[0]) , df_int.shape[0] , df_int_nonan.shape[0]))
    if(df_int_nonan.shape[1] == df_int.shape[1]):
      print("NaN handling: no  feature reduction after dropna(): pre %d , post %d " % (df_int_nonan.shape[1] , df_int.shape[1]))
    else:
      print("NaN handling: !!! FEATURE REDUCTION after dropna(): pre %d , post %d " % (df_int_nonan.shape[1] , df_int.shape[1]))
    if(1):
        print("-I-: DecisionTree")
        # further prune valid features - mainly get rid of crash_id
        validfeats = validfeats[validfeats.regtype != False] # only invalid values are False
        predictors  = list(validfeats[(validfeats.target != True) & (validfeats.regtype != 'bin_cat')].index)
        responsecls = list(validfeats[(validfeats.target == True) & (validfeats.regtype == 'bin_cat')].index)
    print("-I-: DecisionTree - feature selection")
    from sklearn.model_selection import StratifiedKFold,GroupKFold
    from sklearn.feature_selection import RFECV
    from sklearn import tree
    clf = tree.DecisionTreeClassifier(random_state = 42) #max_depth = 5)
    # use full dataset for feature selection
    X_full = df_int_nonan[predictors]
    y_full = df_int_nonan[responsecls]

    print("-I-: DecisionTree - evaluating CV strategy")
    if(0):
      dectree_evaluate_cv_strategy(X_full, y_full)
    else:
      print("-I-: ... skipping")

    print("-I-: previously chosen CV: StratifiedKFold with roc_auc_score")
    # settling on ...
    cvFold = StratifiedKFold
    rfecv = RFECV(estimator=clf, step=1, cv=cvFold(2), scoring='roc_auc')
    rfecv.fit(X_full,y_full.values.ravel())

    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    # TODO: plot the feature names at 45deg angle under the numbers
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()
    print("Optimal number of features : %d" % rfecv.n_features_)

    # print important features
    print("-I-: most important features:")
    clf_imp_feats = print_model_feats_important(rfecv.estimator_, predictors, 0)
    ax = get_ax_barh(clf_imp_feats, title="DecisionTree Important Features")
    plt.show()

    print("-I-: examining most important features:")
    print("ratio  score   non-nan total feature")
    for i,feat in enumerate(clf_imp_feats.index):
        num_not_nan = data_dummies[~data_dummies[feat].isnull()].shape[0] # data_dummies[feat].count() wooudl work too
        print("%0.4f %0.4f %5d %5d %s" % (num_not_nan/ data_dummies.shape[0], clf_imp_feats[i], num_not_nan, data_dummies.shape[0], feat))
    return (predictors,responsecls)

# <def_generate_clf_scatter_plot>
def generate_clf_scatter_plot(featdef, data_dummies, target_feat):
    from sklearn import tree
    from sklearn.metrics import confusion_matrix
    print("simple scatter plot")
    validfeats = featdef[(featdef.regtype != False) & (featdef.type == 'int') & (featdef.dummies == False)]
    # define predictors and response
    # predictors:
    # note: removing some features which have too many missing entries: 'average_daily_traffic_amount' and 'average_daily_traffic_year'
    # TODO: hard-coding is bad - find a way to automatically remove features with too few values
    print("-I-: manually excluding some features which have too many missing entries: 'average_daily_traffic_amount' and 'average_daily_traffic_year'")
    predictors  = list(featdef[(featdef.regtype != False) & (featdef.target != True) & (featdef.dummies == False) & (featdef.regtype != 'bin_cat') & (featdef.type == 'int') & ((featdef.index != 'average_daily_traffic_amount') & (featdef.index != 'average_daily_traffic_year')) ].index)
    # response class:
    responsecls = list(featdef[(featdef.regtype != False) & (featdef.target == True) & (featdef.dummies == False) & (featdef.regtype != 'bin_cat') & (featdef.type == 'int') & (featdef.origin == target_feat)].index)
    #hack, not needed now: responsecls = ['crash_severity']
    if(1):
        print("##############")
        print("predictors:")
        print(predictors)
        print("responsecls:")
        print(responsecls)
        print("##############")

    print("-I-: train-test split")
    # def collapse_dummies(row):
    #     for cat in y_full.columns:
    #         if(row[cat] == 1):
    #             return cat
    testsize = 0.3
#    # data_nonan = data[ predictors + responsecls ].dropna()
#    data_nonan = df_int_nonan
    # get dummies for this particular case
    #(data_dummies,featdef) = featdef_get_dummies(
    #        data[ predictors + responsecls ].dropna(),
    #        featdef)
    data_dummies_firstplot = data_dummies[ predictors + responsecls ].dropna()
    #data_dummies_firstplot.drop(['average_daily_traffic_amount','average_daily_traffic_year'],axis=1,inplace=1)
    X_full = data_dummies_firstplot[predictors]
    y_full = data_dummies_firstplot[responsecls].idxmax(1)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X_full,y_full, test_size=testsize)


    print(X_full.columns)
    print(y_full)


    clf = tree.DecisionTreeClassifier() #max_depth = 5)
    # use full dataset for feature selection

    # raw
    #clf.fit(X_full,y_full)
    #cm = confusion_matrix(y_test,clf.predict(X_full))
    #helpers.plot_confusion_matrix(cm,classes=clf.classes_)

    # train/test
    clf.fit(X_train,y_train)

    cm = confusion_matrix(y_test,clf.predict(X_test))
    plot_confusion_matrix(cm,classes=clf.classes_)
    plt.show()
# </def_generate_clf_scatter_plot>

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

# basic analysis of target variable 'crash_severity'
print("################################################################################")
print("-I-: printing scatter plot for feature 'crash_severity'")
generate_clf_scatter_plot(featdef, data_dummies, 'crash_severity')
print("################################################################################")

# verify
print("################################################################################")
print("-I-:" + "data processing and verification")
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
# Avoid: ValueError: Input contains NaN, infinity or a value too large for dtype('float64').
df_int_nonan = df_int.dropna()
print("NaN handling: Samples: NaN data %d / %d fullset => %d newset" % ( (df_int.shape[0] - df_int_nonan.shape[0]) , df_int.shape[0] , df_int_nonan.shape[0]))
if(df_int_nonan.shape[1] == df_int.shape[1]):
  print("NaN handling: no  feature reduction after dropna(): pre %d , post %d " % (df_int_nonan.shape[1] , df_int.shape[1]))
else:
  print("NaN handling: !!! FEATURE REDUCTION after dropna(): pre %d , post %d " % (df_int_nonan.shape[1] , df_int.shape[1]))
print("################################################################################")

# pca stub
# pca = decomposition.PCA(svd_solver='full')
# pca.fit(pd.get_dummies(data[dummies_needed_list])).transform(pd.get_dummies(data[dummies_needed_list]))


# strategy:
# successively (eventually recursively?) get best predictors while increasing size of dataset
# i.e. initially many features also have many NaN so the dataset is smaller
# 
print("################################################################################")
print("-I-: " + "Determination of Strongest Features")
if(1):
    print("-I-: " + "skipping ...")
else:
    print(" ################################################################################")
    print("-I-: DecisionTree")
    print("-I-: First Run")
    (predictors, responsecls) = run_cross_val(data_dummies, featdef)
    print(" --------------------------------------------------------------------------------")
    print("-I-: result: average_daily_traffic_amount and average_daily_traffic_year are only a small portion of the dataset")
    '''
    ratio  score   non-nan total  feature
    0.1470 0.3880   328  2232 average_daily_traffic_amount
    1.0000 0.3227  2232  2232 crash_year
    0.1326 0.1768   296  2232 average_daily_traffic_year
    0.7424 0.1125  1657  2232 speed_limit
    '''
    print(" ################################################################################")
    print("-I-: Second Run") #creating new dataset without average_daily_traffic_year and average_daily_traffic_amount")
    (predictors, responsecls) = run_cross_val(data_dummies, featdef, ['average_daily_traffic_amount','average_daily_traffic_year'])
    #plt.bar(data_dummies.crash_year.value_counts().index,data_dummies.crash_year.value_counts().values) ; plt.show()
    print(" --------------------------------------------------------------------------------")
    print("-I-: result: crash_year factors in very heavily and warrants further analysis. however, this cuold also simply be due to the fact that a crash year is always associated with every accident record.")
    '''
    ratio  score   non-nan total  feature
    1.0000 0.2753  2232  2232 crash_year
    0.7424 0.1454  1657  2232 speed_limit
    1.0000 0.0460  2232  2232 day_of_week_thursday
    1.0000 0.0416  2232  2232 day_of_week_sunday
    1.0000 0.0401  2232  2232 light_condition_dark_not_lighted
    1.0000 0.0387  2232  2232 intersection_related_non_intersection
    0.9996 0.0384  2231  2232 bin_intersection_related
    0.9937 0.0384  2218  2232 bin_light_condition
    1.0000 0.0370  2232  2232 day_of_week_tuesday
    1.0000 0.0369  2232  2232 bin_manner_of_collision
    1.0000 0.0365  2232  2232 surface_condition
    1.0000 0.0325  2232  2232 day_of_week_monday
    1.0000 0.0320  2232  2232 intersection_related_driveway_access
    1.0000 0.0301  2232  2232 day_of_week_friday
    1.0000 0.0274  2232  2232 intersection_related_intersection
    1.0000 0.0274  2232  2232 day_of_week_saturday
    1.0000 0.0272  2232  2232 light_condition_dark_lighted
    1.0000 0.0270  2232  2232 intersection_related_not_reported
    1.0000 0.0196  2232  2232 day_of_week_wednesday
    1.0000 0.0027  2232  2232 intersection_related_intersection_related
    '''
    print(" ################################################################################")
    print("-I-: Third Run") #creating new dataset without average_daily_traffic_year and average_daily_traffic_amount")
    (predictors, responsecls) = run_cross_val(data_dummies, featdef, ['average_daily_traffic_amount','average_daily_traffic_year','crash_year'])
    print(" --------------------------------------------------------------------------------")
    print("-I-: result: the remaining factors are speed_limit and surface_condition. this makes intuitive sense. However, this result is subject to change on different runs, which is in line with the results seen while evaluating the CV strategy.")
    '''
    # without bin_cat
    ratio  score   non-nan total feature
    0.7424 0.6294  1657  2232 speed_limit
    1.0000 0.3706  2232  2232 surface_condition
    # old, with bin_cat
    ratio  score   non-nan total  feature
    0.7424 0.6389  1657  2232 speed_limit
    1.0000 0.3611  2232  2232 surface_condition
    '''
    print(" ################################################################################")
    print("-I-: Fourth Run") # creating new dataset without speed_limit and surface_condition")
    (predictors, responsecls) = run_cross_val(data_dummies, featdef, ['average_daily_traffic_amount','average_daily_traffic_year','crash_year','speed_limit','surface_condition'])
    print("-I-: result: the remaining factors are varied, but seem to settle around two categories: binary categories, or their counterparts. this makes intuitive sense, and the dataset should be re-run without the binary categories. this was a mistake")
    '''
    # without bin_cat
    ratio  score   non-nan total  feature
    1.0000 0.5124  2232  2232 day_of_week_monday
    1.0000 0.4876  2232  2232 day_of_week_friday
    # old, with bin_cat
    ratio  score   non-nan total  feature
    1.0000 0.4342  2232  2232 bin_manner_of_collision
    0.9996 0.4131  2231  2232 bin_intersection_related
    0.9937 0.1527  2218  2232 bin_light_condition
    ratio  score   non-nan total  feature
    1.0000 0.1739  2232  2232 day_of_week_tuesday
    1.0000 0.1159  2232  2232 day_of_week_wednesday
    1.0000 0.1103  2232  2232 day_of_week_monday
    1.0000 0.1047  2232  2232 bin_manner_of_collision
    1.0000 0.0923  2232  2232 day_of_week_friday
    0.9937 0.0855  2218  2232 bin_light_condition
    1.0000 0.0816  2232  2232 day_of_week_saturday
    1.0000 0.0814  2232  2232 day_of_week_sunday
    0.9996 0.0802  2231  2232 bin_intersection_related
    1.0000 0.0741  2232  2232 day_of_week_thursday
    '''
    print("################################################################################")

    # Next step: train-test split
    print("-I-: train-test split")
    testsize = 0.3
    # data_nonan = data[ predictors + responsecls ].dropna()
    data_nonan = df_int_nonan
    X_train, X_test, y_train, y_test = model_selection.train_test_split(data_nonan[predictors],data_nonan[responsecls], test_size=testsize)

    from sklearn import tree
    clf = tree.DecisionTreeClassifier(random_state = 42) #max_depth = 5)
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

    # plot important features
    print("-I-: most important features:")
    clf_imp_feats = print_model_feats_important(clf, predictors)
    ax = get_ax_barh(clf_imp_feats, title="DecisionTree Important Features")
    plt.show()

    # print strong categories+features
    strongest_cats = {}
    strongest_cats_list = []
    print("-I-:" + "strongest categories, values")
    # plotting important features
    for i in np.argsort(clf.feature_importances_)[::-1]:
      feat = predictors[i]
      feat_orig = feat
      if(featdef.ix[feat].origin):
          feat_orig = featdef.ix[predictors[i]].origin
      # store
      if(feat_orig not in strongest_cats):
          strongest_cats[feat_orig] = [feat]
          strongest_cats_list.append(feat_orig)
          #print("%s - strongest value: %s" % (feat_orig, feat))
      if(feat_orig in strongest_cats):
          strongest_cats[feat_orig].append(feat)
          continue
    print("")
    print("-I-:" + "all categories, values")
    for val in strongest_cats_list:
        print(val)
        pp.pprint(strongest_cats[val])
    # /print strong categories+features

    # print pie-charts for each important feature.
    #+ why, you ask? pie chart shows the values as a direct ratio with each other.
    #+ It also effectively highlights the total amount of values with fewer entries as it is simple to estimate them together as one 'slice'
    #+ the pie chart also confines the visualisation to a fixed area, thereby creating a simple overview.
    #+ a bar chart could also work, but the widths of the diagrams would vary with cardinality and 
    #+   somewhat hides the cumulation of the values with fewer entries as several small-ish bars
    print_imp_feats_piecharts(data,featdef, clf,predictors)

    print("-I-:" + "model accuracy:")
    y_pred = clf.predict(X_test)
    clf.fit(X_train,y_train)
    cm = confusion_matrix(y_test,clf.predict(X_test))
    plot_confusion_matrix(cm,classes=['fubar','aight'])
    plt.show()

#/end of determining strong features
print("################################################################################")



print("################################################################################")
print("-I-:" + "Simple DecisionTree for binary features")
# this model is for human-consumption by generating a human-readable decision tree
#+ as such, it focuses only on binary choices to reduce the complexity
#+ while this does reduce the effectiveness, so does adding even more features.
#+ currently, this includes all available binary features.
print("-I-: train-test split")
# vvv no longer needed, as far as I can tell. commenting out just in case I am overlooing something vvv
## predictors = [
## # 'crash_time',
## # 'crash_time_dec',
##  'bin_intersection_related',
##  'bin_light_condition',
##  'bin_manner_of_collision',
##  ]
## responsecls = [
##  'bin_crash_severity'
##  ]
# now handled by 'featdef'
print("automatic predictors + responsecls")
predictors  = list(featdef[(featdef.regtype == 'bin_cat') & (featdef.target != True)].index)
responsecls = list(featdef[(featdef.regtype == 'bin_cat') & (featdef.target == True)].index)
pp.pprint(predictors)
pp.pprint(responsecls)

testsize = 0.3
data_nonan = data[ predictors + responsecls ].dropna()
X_train, X_test, y_train, y_test = model_selection.train_test_split(data_nonan[predictors],data_nonan[responsecls], test_size=testsize)

from sklearn import tree
clf = tree.DecisionTreeClassifier(random_state = 42) #max_depth = 5)
clf.fit(X_train,y_train)

# prediction and scoring
print("-I-: cross_val_score on train (itself)")
print(model_selection.cross_val_score(clf, X_train, y_train.values.ravel()))
y_pred = clf.predict_proba(X_test)
print("-I-: cross_val_score against test")
print(model_selection.cross_val_score(clf, X_test, y_test.values.ravel()))
cm = confusion_matrix(y_test,clf.predict(X_test))
plot_confusion_matrix(cm,classes=['fubar','aight'])
plt.show()

# DOC: How to interpret decision trees' graph results and find most informative features?
# src: http://stackoverflow.com/a/34872454
print("################################################################################")
print("-I-: most important features:")
for i in np.argsort(clf.feature_importances_)[::-1]:
  print("%f : %s" % (clf.feature_importances_[i],predictors[i]))

# plotting important features
for i in np.argsort(clf.feature_importances_)[::-1]:
  feat = predictors[i]
  #feat = predictors[i].replace('bin_','')
  pltkind = 'pie'
  if(featdef.ix[feat].origin):
      feat_orig = featdef.ix[predictors[i]].origin
      data[feat_orig].value_counts().plot(kind=pltkind, title="%s - original values for %s" % (feat_orig, feat))
  else:
      data[feat].value_counts().plot(kind=pltkind, title="%s " % (feat))
  plt.show()

print("--------------------------------------------------------------------------------")
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
print("--------------------------------------------------------------------------------")
print("-I-: decision tree tree of binary features")
# src: http://scikit-learn.org/stable/modules/tree.html#classification
from IPython.display import (Image,display)
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
display(Image(graph.create_png() , retina=True))
# vvv resolved, thanks to src: https://stackoverflow.com/a/35210224 vvv
#print("-I-: if img doesn't show, run \n Image(pydotplus.graph_from_dot_data(dot_data).create_png() , retina=True)")
# /display tree criteria
print("################################################################################")
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
'''
# http://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection
from sklearn.feature_selection import SelectKBest,chi2,mutual_info_classif,f_classif
# removes all but the k highest scoring features, i.e. output array corresponds to which features to keep
SelectKBest(mutual_info_classif, k=2).fit_transform(X_full,y_full.values.ravel())
'''
# low variance
'''
http://scikit-learn.org/stable/modules/feature_selection.html#removing-features-with-low-variance
VarianceThreshold().fit_transform(X_full)
'''


# Interpreting Decision Tree in context of feature importances
# https://datascience.stackexchange.com/questions/16693/interpreting-decision-tree-in-context-of-feature-importances

# further reading - Random Forests
# http://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm

# xgboost on ubuntu
# manually link
#   https://askubuntu.com/a/764572
# conda install libcc
# https://github.com/dmlc/xgboost/issues/1043

# import importlib
# import helpers
# importlib.reload(helpers)

# pvalues
# from sklearn.feature_selection import SelectKBest,chi2,mutual_info_classif,f_classif
# tmpfilter.fit(X_full,y_full)
# tmpfilter.fit_transform(X_full,y_full)

# import ipynb
# http://nbviewer.jupyter.org/github/jupyter/notebook/blob/master/docs/source/examples/Notebook/Importing%20Notebooks.ipynb
