
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

def dectree_evaluate_cv_strategy(X_full, y_full):
  # Recursive Feature Elimination CV
  # http://scikit-learn.org/stable/auto_examples/feature_selection/plot_rfe_with_cross_validation.html#sphx-glr-auto-examples-feature-selection-plot-rfe-with-cross-validation-py
  from sklearn.model_selection import StratifiedKFold,GroupKFold
  from sklearn.feature_selection import RFECV
  from sklearn import tree
  clf = tree.DecisionTreeClassifier() #max_depth = 5)

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
    clf = tree.DecisionTreeClassifier(max_depth = depth)
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
# Avoid: ValueError: Input contains NaN, infinity or a value too large for dtype('float64').
df_int_nonan = df_int.dropna()
print("NaN handling: Samples: NaN data %d / %d fullset => %d newset" % ( (df_int.shape[0] - df_int_nonan.shape[0]) , df_int.shape[0] , df_int_nonan.shape[0]))
if(df_int_nonan.shape[1] == df_int.shape[1]):
  print("NaN handling: no  feature reduction after dropna(): pre %d , post %d " % (df_int_nonan.shape[1] , df_int.shape[1]))
else:
  print("NaN handling: !!! FEATURE REDUCTION after dropna(): pre %d , post %d " % (df_int_nonan.shape[1] , df_int.shape[1]))

# pca stub
# pca = decomposition.PCA(svd_solver='full')
# pca.fit(pd.get_dummies(data[dummies_needed_list])).transform(pd.get_dummies(data[dummies_needed_list]))


# strategy:
# successively (eventually recursively?) get best predictors while increasing size of dataset
# i.e. initially many features also have many NaN so the dataset is smaller
# 
if(1):
    print("-I-: DecisionTree")
    # TODO: move the validfeats higher up during the nonan phase
    # valid features - defined for regression (aka classification), are integers (because I'm really into that), and ain't supposed to be no dummy (entries meant to be encoded as dummies)
    # 'regtype ! = False' mainly for crash_id
    validfeats = featdef[(featdef.regtype != False) & (featdef.type == 'int') & (featdef.dummies == False)]
    # define predictors and response
    predictors  = list(featdef[(featdef.regtype != False) & (featdef.type == 'int') & (featdef.target != True) & (featdef.dummies == False)].index)
    responsecls = list(featdef[(featdef.regtype != False) & (featdef.type == 'int') & (featdef.target == True) & (featdef.dummies == False) & (featdef.regtype == 'bin_cat')].index)

    if(1):
        print("##############")
        print("predictors:")
        print(predictors)
        print("responsecls:")
        print(responsecls)
        print("##############")

    print("-I-: DecisionTree - feature selection")
    from sklearn.model_selection import StratifiedKFold,GroupKFold
    from sklearn.feature_selection import RFECV
    from sklearn import tree
    clf = tree.DecisionTreeClassifier() #max_depth = 5)
    # use full dataset for feature selection
    X_full = df_int_nonan[predictors]
    y_full = df_int_nonan[responsecls]

    print("-I-: DecisionTree - feature selection")
    if(0):
      dectree_evaluate_cv_strategy(X_full, y_full)
    else:
      print("-I-: ... skipping")

    print("-I-: previously chosen: StratifiedKFold with roc_auc_score")
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
    print("-I-: result: average_daily_traffic_amount and average_daily_traffic_year are only a small portion of the dataset")
    print(" ################################################################################")
    print("-I-: creating new dataset without average_daily_traffic_year and average_daily_traffic_amount")
    # mainly integer data
    validfeats = featdef[(featdef.dummies == False) & (featdef.type == 'int')]
    validfeats = validfeats.drop(['average_daily_traffic_amount','average_daily_traffic_year'])# inplace copy - , inplace=True)
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
        print("-I-: DecisionTree 2")
        # further prune valid features - mainly get rid of crash_id
        validfeats = validfeats[validfeats.regtype != False] # only invalid values are False
        predictors  = list(validfeats[(validfeats.target != True)].index)
        responsecls = list(validfeats[(validfeats.target == True) & (validfeats.regtype == 'bin_cat')].index)
    print("-I-: DecisionTree - feature selection")
    from sklearn.model_selection import StratifiedKFold,GroupKFold
    from sklearn.feature_selection import RFECV
    from sklearn import tree
    clf = tree.DecisionTreeClassifier() #max_depth = 5)
    # use full dataset for feature selection
    X_full = df_int_nonan[predictors]
    y_full = df_int_nonan[responsecls]

    print("-I-: DecisionTree - feature selection")
    if(0):
      dectree_evaluate_cv_strategy(X_full, y_full)
    else:
      print("-I-: ... skipping")

    print("-I-: previously chosen: StratifiedKFold with roc_auc_score")
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
    plt.bar(data_dummies.crash_year.value_counts().index,data_dummies.crash_year.value_counts().values) ; plt.show()
    print("-I-: result: crash_year factors in very heavily and warrants further analysis")
    print(" ################################################################################")
    print("-I-: creating new dataset without crash_year")
    # mainly integer data
    validfeats = featdef[(featdef.dummies == False) & (featdef.type == 'int')]
    validfeats = validfeats.drop(['average_daily_traffic_amount','average_daily_traffic_year'])# inplace copy - , inplace=True)
    validfeats = validfeats.drop(['crash_year'])# inplace copy - , inplace=True)
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
        print("-I-: DecisionTree 3")
        # further prune valid features - mainly get rid of crash_id
        validfeats = validfeats[validfeats.regtype != False] # only invalid values are False
        predictors  = list(validfeats[(validfeats.target != True)].index)
        responsecls = list(validfeats[(validfeats.target == True) & (validfeats.regtype == 'bin_cat')].index)
    print("-I-: DecisionTree - feature selection")
    from sklearn.model_selection import StratifiedKFold,GroupKFold
    from sklearn.feature_selection import RFECV
    from sklearn import tree
    clf = tree.DecisionTreeClassifier() #max_depth = 5)
    # use full dataset for feature selection
    X_full = df_int_nonan[predictors]
    y_full = df_int_nonan[responsecls]

    print("-I-: DecisionTree - feature selection")
    if(0):
      dectree_evaluate_cv_strategy(X_full, y_full)
    else:
      print("-I-: ... skipping")

    print("-I-: previously chosen: StratifiedKFold with roc_auc_score")
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
    print("-I-: result: the remaining factors are speed_limit and surface_condition. this makes intuitive sense")
    print(" ################################################################################")
    print("-I-: creating new dataset without speed_limit and surface_condition")
    # mainly integer data
    validfeats = featdef[(featdef.dummies == False) & (featdef.type == 'int')]
    validfeats = validfeats.drop(['average_daily_traffic_amount','average_daily_traffic_year'])# inplace copy - , inplace=True)
    validfeats = validfeats.drop(['crash_year'])# inplace copy - , inplace=True)
    validfeats = validfeats.drop(['speed_limit','surface_condition'])# inplace copy - , inplace=True)
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
        print("-I-: DecisionTree 3")
        # further prune valid features - mainly get rid of crash_id
        validfeats = validfeats[validfeats.regtype != False] # only invalid values are False
        predictors  = list(validfeats[(validfeats.target != True)].index)
        responsecls = list(validfeats[(validfeats.target == True) & (validfeats.regtype == 'bin_cat')].index)
    print("-I-: DecisionTree - feature selection")
    from sklearn.model_selection import StratifiedKFold,GroupKFold
    from sklearn.feature_selection import RFECV
    from sklearn import tree
    clf = tree.DecisionTreeClassifier() #max_depth = 5)
    # use full dataset for feature selection
    X_full = df_int_nonan[predictors]
    y_full = df_int_nonan[responsecls]

    print("-I-: DecisionTree - feature selection")
    if(0):
      dectree_evaluate_cv_strategy(X_full, y_full)
    else:
      print("-I-: ... skipping")

    print("-I-: previously chosen: StratifiedKFold with roc_auc_score")
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
    print("-I-: result: the remaining factors are varied")
    print(" ################################################################################")

    # Next step: train-test split
    print("-I-: train-test split")
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
    ax = get_ax_barh(clf_imp_feats, title="DecisionTree Important Features")
    plt.show()

    print_imp_feats_piecharts(data,featdef, clf,predictors)

    y_pred = clf.predict(X_test)
    clf.fit(X_train,y_train)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test,clf.predict(X_test))
    plot_confusion_matrix(cm,classes=['fubar','aight'])
    plt.show()




print("-I-: train-test split")
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
cm = confusion_matrix(y_test,clf.predict(X_test))
plot_confusion_matrix(cm,classes=['fubar','aight'])
plt.show()

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
