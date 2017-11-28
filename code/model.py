
from helpers import *
from feature_definitions import *
from txdot_parse import *
from sklearn.ensemble import RandomForestClassifier
from sklearn import (metrics, model_selection, linear_model, preprocessing, ensemble, neighbors, decomposition)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from scipy import interp
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import numpy as np
import pandas as pd
import pprint as pp
import re
#import xgboost as xgb

import os,sys

# hard-coded globals
resource_dir = "output"

if __name__ == '__main__':
    # global options
    options = {
            'graphics' : 0, # 0 - disable, 1 - enable
            'verbose' : 0, # -1 - absolutely silent 0 - minimal info, 1+ - increasing levels
            }

    # choose which model to run
    runmodels = {
            'dectree_evaluate_cv_strat' : 0, # loop through decision strategies - depends on manual_analyse_strongest_predictors
            'manual_analyse_strongest_predictors' : 0, # manual successive determination of strongest features
            'generate_human_readable_dectree' : 0, # human-readable binary decision-tree
            'score_manual_predef_route' : 0, # hard-coded route generated from input data, used for enabling scoring of new data
            'map_manual_analyse_strongest_predictors' : 0, # analyse map with manual successive determination of strongest features
            'map_generate_human_readable_dectree' : 1, # analyse map with human-readable binary decision-tree
            }

    # src: https://docs.python.org/3/howto/argparse.html#id1
    import argparse
    # if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graphics', type=int, default=0) # action="store_true", default=False)
    parser.add_argument('--verbose', type=int, default=0)
    args = parser.parse_args()
    # "args" defined with 'default=<>', no need for a conditional
    options['graphics'] = args.graphics
    options['verbose'] = args.verbose

#<def_model_prepare>
def model_prepare():
    # manual verification and verbose:
    #+ increased default verbosity during model prep
    #+ several steps are for manual verification, it must be printed in the normal "silent" case,
    #+ but should have option to completely silence, e.g. by setting options['verbose'] = -1
    verbose = options['verbose']

    if( options['verbose'] >= 1):
        print("################################################################################")
        print("-I-: " + "Prepare Data for Model")
        print("-I-: " + "running ...")
    ################################################################################
    # DATA : IMPORT , PROCESSING, FEATURE DEFINITION
    ################################################################################
    # IMPORT the crash data
    curdir=os.path.split(__file__)[0]
    datadir=os.path.split(curdir)[0] + "/data"
    datafile = "my_map_grid.csv"
    datafile = os.path.join(datadir, datafile)
    if(0):
      print(__file__)
      pp.pprint(sys.path)


    # PROCESSING: get clean data
    (data,featdef) = preprocess_data(datafile)

    # FEATURE DEFINITION: add binary categories
    if( verbose >= 1 ):
        print("-I-:" + " adding binary categories")
    (data,featdef) = preproc_add_bin_categories(data, featdef, verbose=verbose)
    ################################################################################
    # /DATA : IMPORT , PROCESSING, FEATURE DEFINITION
    ################################################################################

    ################################################################################
    # DATA : EXPLORATION (chapter 3.5)
    ################################################################################
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
    if (options['graphics'] == 1):
        print("################################################################################")
        print("-I-: printing scatter plot for feature 'crash_severity'")
        generate_clf_scatter_plot(featdef, data_dummies, 'crash_severity')
        print("################################################################################")

    # verify
    if(verbose >= 0): # manual verification
        print("################################################################################")
        print("-I-:" + "data processing and verification")
    validpreds = len(list(featdef[(featdef.type == 'int') & (featdef.target != True)].index))
    validtargs = len(list(featdef[(featdef.type == 'int') & (featdef.target == True)].index))
    invalfeats = len(list(featdef[(featdef.type != 'int') & (featdef.dummies != True)].index))
    alldummies = data_dummies.shape[1]
    if(verbose >= 0): # manual verification
        print("valid+string %d / %d total" % (validpreds+validtargs+invalfeats, alldummies))
    # any non-dummy & integer type
    if(0):
        print(data_dummies[list(featdef[(featdef.dummies == False) & (featdef.type == 'int')].index)].info())

    # mainly integer data
    data_int_list = list(featdef[(featdef.dummies == False) & (featdef.type == 'int')].index)
    df_int = data_dummies[list(featdef[(featdef.dummies == False) & (featdef.type == 'int')].index)]
    # Avoid: ValueError: Input contains NaN, infinity or a value too large for dtype('float64').
    df_int_nonan = df_int.dropna()
    if(verbose >= 0): # manual verification
        print("NaN handling: Samples: NaN data %d / %d fullset => %d newset" % ( (df_int.shape[0] - df_int_nonan.shape[0]) , df_int.shape[0] , df_int_nonan.shape[0]))
    if(df_int_nonan.shape[1] == df_int.shape[1]):
        if(verbose >= 0): # manual verification
          print("NaN handling: no  feature reduction after dropna(): pre %d , post %d " % (df_int_nonan.shape[1] , df_int.shape[1]))
    else:
      # if(verbose >= 0): # manual verification, but never silence - this is very important information
      print("NaN handling: !!! FEATURE REDUCTION after dropna(): pre %d , post %d " % (df_int_nonan.shape[1] , df_int.shape[1]))
    if( verbose >= 1):
        print("################################################################################")
        print("-I-:" + " END preparing data for model")
        print("################################################################################")
    return(data, data_dummies, df_int_nonan, featdef)
#</def_model_prepare>

################################################################################
# FUNCTIONS
################################################################################

################################################################################
# <def_dectree_evaluate_cv_strategy>
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
# </def_dectree_evaluate_cv_strategy>
################################################################################

################################################################################
# <def_run_cross_val>
# TODO: split into "find max feats" and "run_cross_val"
def run_cross_val(data_dummies,featdef,dropfeatures=[]):
    verbose = options['verbose']
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
    if(runmodels['dectree_evaluate_cv_strat'] == 1):
      dectree_evaluate_cv_strategy(X_full, y_full)
    else:
      print("-I-: ... skipping")

    print("-I-: previously chosen CV: StratifiedKFold with roc_auc_score")
    # settling on ...
    cvFold = StratifiedKFold
    rfecv = RFECV(estimator=clf, step=1, cv=cvFold(2), scoring='roc_auc')
    rfecv.fit(X_full,y_full.values.ravel())

    # Plot number of features VS. cross-validation scores
    if (options['graphics'] == 1):
        plt.figure()
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation score (nb of correct classifications)")
        # TODO: plot the feature names at 45deg angle under the numbers
        plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
        plt.show()
    print("Optimal number of features : %d" % rfecv.n_features_)

    # print important features
    if( verbose >= 1 ):
        print("-I-: most important features:")
    clf_imp_feats = print_model_feats_important(rfecv.estimator_, predictors, 0)
    if (options['graphics'] == 1):
        ax = get_ax_barh(clf_imp_feats, title="DecisionTree Important Features")
        plt.show()

    if(verbose >= 1):
        print("-I-: examining most important features:")
        print("ratio  score   non-nan total feature")
        for i,feat in enumerate(clf_imp_feats.index):
            num_not_nan = data_dummies[~data_dummies[feat].isnull()].shape[0] # data_dummies[feat].count() wooudl work too
            print("%0.4f %0.4f %5d %5d %s" % (num_not_nan/ data_dummies.shape[0], clf_imp_feats[i], num_not_nan, data_dummies.shape[0], feat))
    #--------------------------------------------------------------------------------------------------------------
    # ROC curve - currently independent of RFECV results
    # source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py
    #------

    cvFold = StratifiedKFold(n_splits=9)
    # use previously chosen important features from dataset
    print("-I: RFECV important predictors")
    pp.pprint(clf_imp_feats.index)
    X_full = df_int_nonan[clf_imp_feats.index]
    y_full = df_int_nonan[responsecls]
    # store multiple true-positive-rate
    tprs = []
    # store multiple area-under-curve
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    i = 0
    for train, test in cvFold.split(X_full, y_full.values.ravel()):
        # DOC: indexing src: https://stackoverflow.com/a/45116022
        #+ DOC: alternative - shuffle df.index -  src: https://stackoverflow.com/a/31247293
        proba_list_ = clf.fit(X_full.iloc[train], y_full.iloc[train]).predict_proba(X_full.iloc[test])
        # compute ROC curve and ROCAUC (area under ROC curve)
        #+ false-positive-rate, true-positive-rate
        fpr, tpr, thresholds = roc_curve(y_full.iloc[test], proba_list_[:,1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        if (options['graphics'] == 1):
            plt.plot( fpr, tpr, lw=1, alpha=0.3, label = 'ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        i += 1
    if (options['graphics'] == 1):
        plt.plot( [0, 1], [0, 1], linestyle='--', lw=2, color='r', label='50%', alpha=0.8)
    # plot average true-positive-rate vs average false-positive-rate
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    if ( options['graphics'] == 1):
        plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC= %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=0.8)
    # plot average false-positive-rate vs
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    if (options['graphics'] == 1):
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2, label=r'$\pm$ 1 std. dev.')

    if (options['graphics'] == 1):
        # options
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True  Positive Rate')
        plt.title('Receiver Operating Characteristic Curves')
        plt.legend(loc="lower right")
        # display
        plt.show()



    print("-I-: total predictors:")
    pp.pprint(predictors)
    print("-I: RFECV important predictors")
    pp.pprint(clf_imp_feats.index)
    # TODO: return clf_imp_feats instead of predictors
    #+ RFECV returns very small list of important features (only two for now). Judgement says this would be too few for real-world use, especially when keeping in mind that RFECV is also just a random process based on a seed. Best approach would likely be stacking a few models, or finding the "second best" number of features. For now, RFECV has done its job for eliminating the features with lowest number of values, don't use for determining optimal predictors though. Simply return predictors list, which is generated without the dropfeatures and therefore the caller is getting predictors from passed-in featdef - list of dropfeatures
    # return (clf_imp_feats.index,responsecls)
    # for now, return full list of predictors
    return (predictors,responsecls)
# </def_run_cross_val>
################################################################################

################################################################################
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
    if (options['graphics'] == 1):
        plt.show()
# </def_generate_clf_scatter_plot>
################################################################################

################################################################################
# /FUNCTIONS
################################################################################
################################################################################
# FUNCTIONS_FOR_API
################################################################################
# DUPLICATE from server.py
# dump json to file for consumption by whatever else needs it
def retrieve_json_file(filename):
    import json
    verbose = options['verbose']
    # for saving:
    #if ( verbose >= 1):
    #    print("# save to file")

    # TODO: convert his, want to force everything in one dir. tmp:
    # filepath=("%s/%s" % (resource_dir, filename))
    filepath = filename
    if( verbose >= 1):
        print("mock-response sending to : " + filepath)
    # for saving:
    # with open(filepath, 'w') as outfile:
    #    json.dump(response_json, outfile)

    # open file as json
    loadedjson = str()
    with open(filepath, 'r') as infile:
       loadedjson = json.load(infile)

    # return native object
    return loadedjson

# DUPLICATE from server.py
# save json to file for consumption by whatever else needs it
#+ in practice, not such a great idea, but for now it is what it is
#+ ultimately, the server needs to call the model anyway.
#+ will have to fix the encoding issues of converting to 2to3; not impossible, but super annoying
def save_json_file(response_json, filename): # ="gps_output_route.json"):
    import json
    verbose = options['verbose']
    if( verbose >= 1):
        print("# save to file")
    # tmp:
    filepath=("%s/%s" % (resource_dir, filename))
    # if ( quiet != 1):
    #     print("mock-response sending to : " + filepath)
    with open(filepath, 'w') as outfile:
       json.dump(response_json, outfile)

    return filename

# mock json request
def mock_receive_request_json():
    # NOTE: files may have to be symlinked first:
    # ln -s ../server/res/gps_input_route.json output/
    # tmp: - probably needs to be in a config
    filename = "gps_input_route.json"
    filepath=("%s/%s" % (resource_dir, filename))
    # testing:
    # filepath="t/route_json/gps_generic.json"
    return retrieve_json_file(filepath)


# dump json to file for consumption by whatever else needs it
# save, load, validate json
def mock_return_response_json(route):
    verbose = options['verbose']
    print("# save to file")
    # tmp:
    filepath="output/gps_scored_route.json"
    filepath="gps_scored_route.json"
    print("mock-response sending to : " + filepath)
    save_json_file(route, filepath)

    # verify
    loadedjson = str()
    filepath="output/gps_scored_route.json"
    loadedjson =  retrieve_json_file(filepath)

    loadedroute = json.loads(loadedjson)

    return_value = -1
    # verify
    if( response_json == loadedjson ):
        print("json string resurrected successfully")
        return_value = 1
    # compare the dict if possible?
    return return_value

#def mock_return_response_json(route):

    # not sure how args should look
################################################################################
# /FUNCTIONS_FOR_API
################################################################################

################################################################################
# MODELS - various
################################################################################
# pca stub
# pca = decomposition.PCA(svd_solver='full')
# pca.fit(pd.get_dummies(data[dummies_needed_list])).transform(pd.get_dummies(data[dummies_needed_list]))


################################################################################
# MODEL+EVALUATION - identify strong features
################################################################################
#<def_manual_analyse_strongest_predictors>
# strategy:
# successively (eventually recursively?) get best predictors while increasing size of dataset
# i.e. initially many features also have many NaN so the dataset is smaller
# 
def manual_analyse_strongest_predictors(data, data_dummies, df_int_nonan, featdef):
    verbose = options['verbose']
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
    # (predictors, responsecls) = run_cross_val(data_dummies, featdef, ['average_daily_traffic_amount','average_daily_traffic_year','crash_year','speed_limit','surface_condition'])
    run_cross_val(data_dummies, featdef, ['average_daily_traffic_amount','average_daily_traffic_year','crash_year','speed_limit','surface_condition'])
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
    print("-I-: cross_val_score on train (itself) with default, then with roc_auc")
    print(model_selection.cross_val_score(clf, X_train, y_train.values.ravel()))
    print(model_selection.cross_val_score(clf, X_train, y_train.values.ravel(), scoring='roc_auc'))
    # TODO: how to use multioutput-multioutput?
    # vvv multiclass-multioutput is not supported vvv
    # print(model_selection.cross_val_score(clf, X_train, y_train))
    y_pred = clf.predict_proba(X_test)
    print("-I-: cross_val_score against test with default, then with roc_auc")
    print(model_selection.cross_val_score(clf, X_test, y_test.values.ravel()))
    print(model_selection.cross_val_score(clf, X_test, y_test.values.ravel(), scoring='roc_auc'))

    # plot important features
    if( verbose >= 1 ):
        print("-I-: most important features:")
    clf_imp_feats = print_model_feats_important(clf, predictors, printout=verbose)
    if (options['graphics'] == 1):
        ax = get_ax_barh(clf_imp_feats, title="DecisionTree Important Features")
        plt.show()

    # print strong categories+features
    strongest_cats = {}
    strongest_cats_list = []
    if(verbose >= 1):
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
    if (options['graphics'] == 1):
        print_imp_feats_piecharts(data,featdef, clf,predictors)

    print("-I-:" + "model accuracy:")
    y_pred = clf.predict(X_test)
    clf.fit(X_train,y_train)
    if (options['graphics'] == 1):
        cm = confusion_matrix(y_test,clf.predict(X_test))
        plot_confusion_matrix(cm,classes=['fubar','aight'])
        plt.show()
    # return the model
    return (clf, predictors, responsecls)
#/end of determining strong features
#</def_manual_analyse_strongest_predictors>
################################################################################
# /MODEL+EVALUATION - identify strong features
################################################################################


################################################################################
# MODEL+EVALUATION - human readable
################################################################################
#<def_generate_human_readable_dectree>
def generate_human_readable_dectree(data, data_dummies, featdef):
    verbose = options['verbose']
    # this model is for human-consumption by generating a human-readable decision tree
    #+ as such, it focuses only on binary choices to reduce the complexity
    #+ while this does reduce the effectiveness, so does adding even more features.
    #+ currently, this includes all available binary features.
    if( verbose >= 1):
        print("################################################################################")
        print("-I-:" + "Simple DecisionTree for binary features")
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
    predictors  = list(featdef[(featdef.regtype == 'bin_cat') & (featdef.target != True)].index)
    responsecls = list(featdef[(featdef.regtype == 'bin_cat') & (featdef.target == True)].index)
    if(verbose >= 1):
        print("automatic predictors + responsecls")
        pp.pprint(predictors)
        pp.pprint(responsecls)

    testsize = 0.3
    data_nonan = data[ predictors + responsecls ].dropna()
    X_train, X_test, y_train, y_test = model_selection.train_test_split(data_nonan[predictors],data_nonan[responsecls], test_size=testsize)

    from sklearn import tree
    clf = tree.DecisionTreeClassifier(random_state = 42) #max_depth = 5)
    clf.fit(X_train,y_train)

    # prediction and scoring
    y_pred = clf.predict_proba(X_test)
    if(verbose >= 1):
        print("-I-: cross_val_score on train (itself)")
        print(model_selection.cross_val_score(clf, X_train, y_train.values.ravel()))
        print("-I-: cross_val_score against test")
        print(model_selection.cross_val_score(clf, X_test, y_test.values.ravel()))
    if (options['graphics'] == 1):
        cm = confusion_matrix(y_test,clf.predict(X_test))
        plot_confusion_matrix(cm,classes=['fubar','aight'])
        plt.show()

    # DOC: How to interpret decision trees' graph results and find most informative features?
    # src: http://stackoverflow.com/a/34872454
    if(verbose >= 1):
        print("################################################################################")
        print("-I-: most important features:")
        for i in np.argsort(clf.feature_importances_)[::-1]:
          print("%f : %s" % (clf.feature_importances_[i],predictors[i]))

    if (options['graphics'] == 1):
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

    # return the model
    return (clf, predictors, responsecls)
#<def_generate_human_readable_dectree>
################################################################################
# /MODEL+EVALUATION - human readable
################################################################################

################################################################################
# MODEL+EVALUATION - score manual-created user route
################################################################################
# <def_score_manual_predef_route>
def score_manual_predef_route(data, data_dummies, featdef):
    print("################################################################################")
    print("#                                      TODO                                     ")
    print("#                                      TODO                                     ")
    print("#                                      TODO                                     ")
    print("start here for the model fitting on pre-defined gps data                        ")
    print(" create an X_test_gps_coord of crash-data entries and then clf.predict(X_test_gps_coord)  ")
    print("may have to remove 'manner of collision' and other after-the fact features. see 'post-fact' in feature_definitions.py")
    print("################################################################################")

    print("################################################################################")
    print("-I-: " + "WORK_IN_PROGRESS - scoring manual user route <score_manual_predef_route>")
    print("################################################################################")
    print("#                                      NEXT:                                     ")
    print("#                                      NEXT:                                     ")
    print("#                                      NEXT:                                     ")
    print(" !! NOT YET !! fuzzy gps match")
    print("refer to the outline for more details")
    # dump out file which contains values whch can be provided by user/tool/etc
    #+ user is expected to plan their route by selecting intersections and deleting the remaining columns
    #+ note - this is a WIP, just to illustrate the type of data needed, still needs to be claned up.
    #+ e.g. user won't remove columns based on 'crash_year', or know the 'manner_of_collision', but these values could be useful yet
    #+ e.g. user can enter the time they are going to travel, but doesn't remove cols based on 'crash_time'.
    #+ in that sense, some cols are for conveying info to user, some are for user to input info
    #+ won't change this format though, since this is just a bootstrapping procedure to get this code to correlate user input with existing data

    # hard-code:
    # step 1: dump valid, then read in, score, etc
    # step 2: edit dump, feed into 'step 1'

    # organisation: relative importance roughly from left->right in the resulting csv
    #+ route most important, on the left
    #+ weather, time, etc easy to provide, next left
    #+ remaining unsure data, all on the right
    # Note: order etc subject to change based on the model. Starting with the basic model for now, moving on from there
    # TODO: this is hard-coded based on deciding which options are not needed. find a way to use featdef to get these features. May need to compare the columns to the following output to determine the pattern. 
    # +TODO: -or- BETTER IDEA -  only dump the predictors, which is the better idea
    temp_user_csv_name="route_planner_options.csv" # full csv
    my_likey = [
        'latitude' ,
        'longitude' ,
        'intersecting_street_name' ,  # required
        'street_name' ,               # required
        #----
        'crash_time' ,                # meant as 'time of travel' 
        'light_condition' ,           # user should know this, but ultimately it can be looked up based on time?
        'weather_condition' ,         # good to know
        'surface_condition' ,         # not always known, but maybe?
        'day_of_week' ,               # less relevant for immediate route planning
        #----
        'intersection_related' ,      # user won't know, but maybe model can
        'manner_of_collision' ,       # leaving here to indicate that model may be able to predict what to watch out for 
        'crash_datetime' ,            # enter exact time ... I think preproc can handle this conversion, actually
        'crash_time_30m',             # probably irrelevant? maybe easier for planning purposes
        'crash_year' ,                # irrelevant, may be needed later for granularity
        ]
    data[my_likey].to_csv(temp_user_csv_name)
    # ^^^ IGNORING FOR NOW ^^^^ 
    #+ the human-readable model only has three predictors, start with that and get complex later

    # limited UI-CSV, with only ~3 cols
    temp_user_csv_name="route_planner_options_limited.csv"
    ########################################
    # prepare model
    ########################################
    # current SIMPLE user input:
    #+ use the generate_human_readable_dectree model, which has only three features
    # get a copy of the complete model which was run on the entire dataset
    model_clf_simple, clf_simple_predictors, clf_simple_responsecls = generate_human_readable_dectree(data, data_dummies, featdef)

    # user-required data will be based on model's features
    #+ TODO: the model features are copy-pasted from generate_human_readable_dectree, make this flexible!
    #+ +TODO otherwise we have to manually update hard-coded values, no fun. 
    #notNeeded-ReturnedFromModelGeneratorNow# clf_simple_predictors  = list(featdef[(featdef.regtype == 'bin_cat') & (featdef.target != True)].index)
    #notNeeded-WontPredictOnResponseCLS# clf_simple_responsecls = list(featdef[(featdef.regtype == 'bin_cat') & (featdef.target == True)].index)

    ########################################
    # interact with user
    ########################################
    # generate the "user interface", a spreadsheet with all of the options
    # TODO: this is too bare-bones, only has the bin_cat with no street-names. will need to use the pre-processing to do this right! I.e. let the CSV contain street names etc but only use the required predictors once it's loaded in
    # limit to 80 rows
    data[ clf_simple_predictors + my_likey ][:80].to_csv(temp_user_csv_name)

    # drop NA inputs, can't assume user data is correct!
    #+ for better code clarity, could do this in two steps with dropna(inplace=True)
    user_route_data = pd.read_csv(temp_user_csv_name).dropna()

    # make user data useful
    # NOTE: phase1 - don't worry about this for this phase, this isn't the final input form
    #+ NEXT STEP:
    #+ focus on correlating intersections against the DB (fuzzy match of gps)
    # add binary categories - user not expected to know the breakdown, as definitions of bin_cat can change over time
    # (user_route_data,featdef) = preproc_add_bin_categories(user_route_data, featdef, verbose=1)

    ########################################
    # scoring the route, assume sanitised input
    ########################################
    # extract relevant data for scoring
    X_test = user_route_data[clf_simple_predictors]

    ## prediction and scoring
    #noNeedHere-onlyForModelGen# print("-I-: cross_val_score on train (itself)")
    #noNeedHere-onlyForModelGen# print(model_selection.cross_val_score(model_clf_simple, X_train, y_train.values.ravel()))

    # 2 cols, corresponding to range of response-class. I.e. binary 0,1 therefore 2 cols
    y_pred = model_clf_simple.predict_proba(X_test)
    y_pred_predict = model_clf_simple.predict(X_test)
    print("-I-: WARNING - author could be misinterpretting the function calls! /WARNING As per current understanding of sklearn, for the given route the chance of severe injury given an accident is:")
    print(np.average(y_pred[:,1])) # second column is chance of '1', i.e. severe injury
    # TODO: determine how to score using the prediction
    #PROBABLY-noNeedHere-onlyForModelGen# print("-I-: cross_val_score against test")
    #PROBABLY-noNeedHere-onlyForModelGen# print(model_selection.cross_val_score(model_clf_simple, X_test, y_test.values.ravel()))
    #PROBABLY-noNeedHere-onlyForModelGen# cm = confusion_matrix(y_test,model_clf_simple.predict(X_test))
    #PROBABLY-noNeedHere-onlyForModelGen# plot_confusion_matrix(cm,classes=['fubar','aight'])
    #PROBABLY-noNeedHere-onlyForModelGen# plt.show()
    print("################################################################################")
    print("-I-: " + "END - WORK_IN_PROGRESS - scoring manual user route </score_manual_predef_route>")
    print("################################################################################")
# <def_score_manual_predef_route>
################################################################################
# /MODEL+EVALUATION - score manual-created user route
################################################################################

################################################################################
# MODEL CACHING - generate or retrieve model+features
################################################################################
def retrieve_model(data, data_dummies, featdef):
    # <PICKLE>
    # https://stackoverflow.com/questions/10592605/save-classifier-to-disk-in-scikit-learn
    # path to pickle:
    import pickle
    path_saved_model = str()
    if( runmodels['map_generate_human_readable_dectree'] ):
        path_saved_model = "output/human_read_dectree.pkl"
    elif( runmodels['map_manual_analyse_strongest_predictors'] ):
        path_saved_model = "output/human_read_dectree.pkl"
    import os.path
    # model_clf_score_route, clf_score_predictors, clf_score_responsecls = (tree.DecisionTreeClassifier(), [], pd.DataFrame)
    model_clf_score_route, clf_score_predictors, clf_score_responsecls = ([],[],[])
    # load if exists
    loadpickle=1
    if (loadpickle  and os.path.exists(path_saved_model) and os.path.isfile(path_saved_model) ):
        print("-I-: retrieving model from pickle file")
        with open (path_saved_model, 'rb') as fh:
            model_clf_score_route, clf_score_predictors, clf_score_responsecls = pickle.load(  fh )
            # model_clf_score_route = pickle.load(fh)

    else:
        print("-I-: creating model")
        if( runmodels['map_generate_human_readable_dectree'] ):
            model_clf_score_route, clf_score_predictors, clf_score_responsecls = generate_human_readable_dectree(data, data_dummies, featdef)
        if( runmodels['map_manual_analyse_strongest_predictors'] ):
            model_clf_score_route, clf_score_predictors, clf_score_responsecls = manual_analyse_strongest_predictors(data, data_dummies, featdef)
    # dump
    print("-I-: storing model to path %s" % path_saved_model)
    with open (path_saved_model, 'wb') as fh:
        # wrong place, used to create the model # pickle.dump( (data, data_dummies, featdef) , fh )
        pickle.dump(
                (model_clf_score_route, clf_score_predictors, clf_score_responsecls)
                , fh )

    # TODO-PICKLE  - validate here: ValueError: Unable to coerce to Series, length must be 65: given 14
    # </PICKLE>
    # return the model
    return (model_clf_score_route, clf_score_predictors, clf_score_responsecls)
################################################################################
# /MODEL CACHING - generate or retrieve model+features
################################################################################

# self-run
if(__name__ != '__main__'):
    # TMP
    quit()

################################################################################
# PREPROCESS
################################################################################
# load data, featdef, etc
(data, data_dummies, df_int_nonan, featdef) = model_prepare()
################################################################################
# /PREPROCESS
################################################################################

################################################################################
# MODEL+EVALUATION - identify strong features
################################################################################
if( options['verbose'] >= 0):
    print("################################################################################")
    print("-I-: " + "Determination of Strongest Features")
if( runmodels['manual_analyse_strongest_predictors'] ):
    if( options['verbose'] >= 0):
        print("-I-: " + "running ...")
        model_clf_manual, clf_manual_predictors, clf_manual_responsecls = manual_analyse_strongest_predictors(data, data_dummies, df_int_nonan, featdef)
else:
    if( options['verbose'] >= 0):
        print("-I-: " + "skipping ...")
if( options['verbose'] >= 0):
    print("################################################################################")

################################################################################
# MODEL+EVALUATION - human readable
################################################################################
if( options['verbose'] >= 0):
    print("################################################################################")
    print("-I-: " + "Human Readable Decision Tree")
if( runmodels['generate_human_readable_dectree'] ):
    if( options['verbose'] >= 0):
        print("-I-: " + "running ...")
    model = generate_human_readable_dectree(data, data_dummies, featdef)
else:
    if( options['verbose'] >= 0):
        print("-I-: " + "skipping ...")
if( options['verbose'] >= 0):
    print("################################################################################")
################################################################################
# /MODEL+EVALUATION - human readable
################################################################################

################################################################################
# MODEL+EVALUATION - score manual-created user route
################################################################################
if( options['verbose'] >= 0):
    print("################################################################################")
    print("-I-: " + "Score Manually-Created User Route")
if( runmodels['score_manual_predef_route'] ):
    if( options['verbose'] >= 0):
        print("-I-: " + "running ...")
    # no return value, just an enablement stub
    score_manual_predef_route(data, data_dummies, featdef)
else:
    if( options['verbose'] >= 0):
        print("-I-: " + "skipping ...")
if( options['verbose'] >= 0):
    print("################################################################################")
################################################################################
# /MODEL+EVALUATION - score manual-created user route
################################################################################

################################################################################
# clear env as best we can until this all gets refactored
# delete by name
if(0):
    del(user_route_data, model_clf_simple, clf_simple_predictors, clf_simple_responsecls, X_test, y_pred, y_pred_predict)
# then loop and verify
for var in ['user_route_data' , 'model_clf_simple' , 'clf_simple_predictors' , 'clf_simple_responsecls' , 'X_test', 'y_pred' , 'y_pred_predict' ]:
    print(var)
    if var not in locals():
      print("varcheck pass - succesfully removed " + var)
    else:
      print("varcheck fail - stil defined        " + var)

print("################################################################################")
print("-I-: " + "WORK_IN_PROGRESS - <score_manual_generic_route> ")
print("################################################################################")
print("#                                      NEXT:                                     ")
print("#                                      NEXT:                                     ")
print("#                                      NEXT:                                     ")
print("refer to the outline for more details")
print('''
Today's menu:
Get geo-json route:
[x] * geo-json route planning: [JSFiddle+F12] - get sample of third-party routing json response
[x] * client-server mock-interface: [ model.py + tbd ] - mock function returns hard-coded geo-json
[ ]   * this approximates a future http connection
[ ]   * "import mock_webthing as webthing" ....
[ ] Score route:
[x] * input to model: convert geo-json to format usable by model.predict
[x]   * hack: re-use existing data for now, just overwrite the GPS
[x] * score : scores = model.predict(<newdata>)
15:00
[x] * client-server mock-interface: [ model.py + tbd ] - generate json with gps coords, score
[x]   * save coords to file
[ ]   * consumer: maybe modified mapgen to display the scores, not sure yet.

Deliverable: 
see literal pen-and-ink notebook (not ipynb)
''')
verbose_score_manual_generic_route = 0
########################################
# prepare model
########################################
# current SIMPLE user input:
#+ use the generate_human_readable_dectree model, which has only three features
# get a copy of the complete model which was run on the entire dataset

# get relevant model
model_clf_score_route, clf_score_predictors, clf_score_responsecls = retrieve_model(data, data_dummies, featdef)
########################################
# MOCK user environmental input
########################################
print("################################################################################")
print("-I-: " + "featdef:")
print("################################################################################")
for feat in featdef.columns:
    print(feat);
    pp.pprint( featdef[feat].unique() )
print("################################################################################")
print("-I-: " + "/featdef")
print("################################################################################")

########################################
# USER ENVIRONMENT DATA
# auto_route_data is the automatically obtained data from the client
# Note: working on mock data, but with the human_read_dectree that's not a problem. Essentially this is hard-coded to best-possible conditions right now, but also a few "fake conditions" such as whether the cyclist would be at-fault for the accident (i.e. whether they are cycling defensively)
# 
# TODO: convert to handle input data from client
# - steps for model:
# [x] step1: hard-code a client request with input data, populate auto_route_data
# [ ] step1a: create function to read client env-data from file
# - steps for client:
# [ ] step2: update client to send hard-coded data
# [ ] step3: update client to send user-selected data
# [ ] step3b: find out if requested data can be auto-procured

# step1:
# if using model from map_manual_analyse_strongest_predictors
if( runmodels['map_manual_analyse_strongest_predictors'] ):
    print("-E-: user_environment not configured for map_manual_analyse_strongest_predictors")
    quit()
user_environment = {
#    'latitude' ,
#    'longitude' ,
#    'intersecting_street_name' ,  # required
#    'street_name' ,               # required
    #----
    'crash_time' : '16:18' ,                # meant as 'time of travel' 
    'light_condition' : 'daylight' ,           # user should know this, but ultimately it can be looked up based on time?
    'weather_condition' : 'cloudy' ,         # good to know
    'surface_condition' : 'dry' ,         # not always known, but maybe?
    'day_of_week' : 'wednesday' ,               # less relevant for immediate route planning
    #----
#    'intersection_related' ,      # user won't know, but maybe model can find out from DB
#    'manner_of_collision' ,       # leaving here to indicate that model may be able to predict what to watch out for 
    'crash_datetime' : '2017-01-01 16:18:00' ,            # enter exact time ... I think preproc can handle this conversion, actually
    'crash_time_30m' : 1630 ,             # probably irrelevant? maybe easier for planning purposes
    'crash_year' : '2017' ,                # irrelevant, may be needed later for granularity
    }

# if using model from map_generate_human_readable_dectree
user_environment = {
#gmap#        'bin_intersection_related' : '0', # 1 : intersection where cyclist should expect cars , 0 : all others
        'bin_light_condition' : '1',      # 1 : daylight , 0 : all others
        'bin_manner_of_collision' : '0',  # our distinction: 0 : motorist likely at fault , 1 : fault unclear
        }
#
########################################

# TODO : pass in the filename
geodata = mock_receive_request_json()
if(0): # not using using overview_path, too many datapoints
    print("route data - overview_path")
    pp.pprint( geodata['routes'][0]['overview_path'])

print("required features:")
pp.pprint(  clf_score_predictors )
print("convert to model-consumable format")
print(" need to merge GPS data with user input:")
pp.pprint( user_environment.keys())
if(verbose_score_manual_generic_route == 1):
    print('''

    until timer 20min: munge together dataset - insert generic gps as overwrite into manual-user-route data
    => need to match the lengths
    ''')
print("munge - length-adjusted dataset [ auto_route_data ] ")

#<def_get_gmap_direction_coords>
# process google directions data, i.e. extract only  the gps coords
# [x] TODOne: use 'steps' instead of 'overview_path' - overview_path has too much data, need something else
# [x] TODOne: refactor to reference 'geodata' instead of mock_receive_request_json
def get_gmap_direction_coords(geodata):
    '''
    A DirectionsLeg defines a single leg of a journey from the origin to the destination in the calculated route.
    For routes that contain no waypoints, the route will consist of a single "leg,"
     but for routes that define one or more waypoints, the route will consist of one or more legs,
     corresponding to the specific legs of the journey.
    '''
    '''
    doc source: https://developers.google.com/maps/documentation/javascript/directions src "Directions Steps"
    start_location contains the LatLng of the origin of this leg.
        Because the Directions Web Service calculates directions between locations by using the nearest transportation option (usually a road) at the start and end points, start_location may be different than the provided origin of this leg if, for example, a road is not near the origin.
    end_location contains the LatLng of the destination of this leg.
        Because the DirectionsService calculates directions between locations by using the nearest transportation option (usually a road) at the start and end points, end_location may be different than the provided destination of this leg if, for example, a road is not near the destination.
    'maneuver' not readily documented 
    '''
    routes_steps={}
    # routes: 1 , legs : 1 (no waypoints) , steps : n
# refactor_multi_route_score_r1
    # [x] TODOne: anticipate several routes
    # [ ]TODO: anticipate several legs (waypoints) etc
    # geodata['routes'][0]['legs'][0]['steps'][0]['start_location']
    for ri, route in enumerate( geodata['routes'] ):
        routes_steps[ri] = []
        #for step in ( geodata['routes'][0]['legs'][0]['steps'] ):
        for step in ( route['legs'][0]['steps'] ):
            # start_location is segment start, end_location is segment stop
            #+ may be useful in future to look up crash-location data for a segment
            if(0): # miniscule difference, not visible on map at all
                print( step['start_location'] )
                routes_steps[ri].append( step['start_location'] )
#            print( step['end_location'] )
            routes_steps[ri].append( step['end_location'] )
            # maneuver : left,right, etc . shove it in with the lat/lng
            if('maneuver' in step):
                # for determining possible vals of maneuver
                print("maneuver: " + step['maneuver'])
                # NOTE: this is a hard-coded mapping between gmaps api and txdot-model feature, ideally would be abstracted
                # txdot-model has data on 'intersection_related'
                # routes_steps[ri][-1]['maneuver'] = step['maneuver']
                # encode in txdot-model-friendly format
                # values for maneuver:
                # maneuver: ''
                # maneuver: turn-left
                # maneuver: turn-right
                # values for bin_intersection_related :
                # bin_true = ['intersection_related', 'intersection',]
                # bin_false = ['non_intersection', 'driveway_access',]
                routes_steps[ri][-1]['bin_intersection_related'] = 1
                if(step['maneuver'] == ''):
                    routes_steps[ri][-1]['bin_intersection_related'] = 0
                
    # only a few coords
    return routes_steps
    # too many coords
    return geodata['routes'][0]['overview_path']
#</def_get_gmap_direction_coords>

#<def_score_single_route>
def score_single_route(route_feat_list):
    print("creating dataframe")
    print("munge - insert gps coords")
    # create dataframe for X_test
    # get gps coordinates, maneuver and the user data
    auto_route_data = pd.DataFrame.from_dict(
# refactor_multi_route_score_r1
#            get_gmap_direction_coords(geodata)
            route_feat_list
            )

    print("munge - insert user-env data")
    # need to impute the user-env values to each row
    # first, add user-data to the df:
    auto_route_data[list(user_environment.keys())] = pd.DataFrame.from_dict(user_environment, orient='index').transpose()
    # then set the user-data cols to the df
    #+ # avoid the following error:
    #+ A value is trying to be set on a copy of a slice from a DataFrame.
    #+ Try using .loc[row_indexer,col_indexer] = value instead
    #+ 
    #+ => actually - use temporary assignments instead
    #+ either:
    #+ auto_route_data[list(user_environment.keys())] = 
    #+     auto_route_data[list(user_environment.keys())].fillna(method='ffill')
    #+ or:
    auto_route_data.update(
            auto_route_data[list(user_environment.keys())].fillna(method='ffill'))

    print(auto_route_data.head())

    print("auto_route_data total amount:" + str(auto_route_data.shape) )

    print("--------------------------------------------------------------------------------")
    print(" DATA VERIFICATION " )
    print("auto_route_data['lat','lng'] shape:" + str(auto_route_data[['lat','lng']].shape) )
    print("--------------------------------------------------------------------------------")

    #-# print("munge - quickly - verify that two np arrays have same values:")
    #-# #+ src: https://stackoverflow.com/questions/10580676/comparing-two-numpy-arrays-for-equality-element-wise
    #-# if( (auto_route_data[['latitude','longitude']].values == auto_route_dict.values).all() != True ):

    print("running the model")
    print("# vvv copypasta vvv")
    # TODO: copy-pasted the minimal setup, still need to setup model etc
    ########################################
    # scoring the route, assume sanitised input
    ########################################
    # extract relevant data for scoring
    X_test = auto_route_data[clf_score_predictors]

    # 2 cols, corresponding to range of response-class. I.e. binary 0,1 therefore 2 cols
    y_pred = model_clf_score_route.predict_proba(X_test)
    y_pred_predict = model_clf_score_route.predict(X_test)
    print("-I-: WARNING - author could be misinterpretting the function calls! /WARNING As per current understanding of sklearn, for the given route the chance of severe injury given an accident is:")
    print(np.average(y_pred[:,1])) # second column is chance of '1', i.e. severe injury
    print("# ^^^ copypasta ^^^")

    print(" combine score with gps coordinates")
    auto_route_data['score'] = y_pred[:,1]
    #---------------------------------------- 
    # TODO: do it right
    '''
    In [12]: auto_route_data['score'] = y_pred[:,1]
    /home/yoinkbird/devtools/miniconda3/lib/python3.6/site-packages/ipykernel_launcher.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead

    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      """Entry point for launching an IPython kernel.
    '''
    #---------------------------------------- 
    return auto_route_data
#</def_score_single_route>

# refactor_multi_route_score_r1
# refactor_multi_route_score_r2 - limited response for first conversion
geodata_routes = get_gmap_direction_coords(geodata)
# refactor_multi_route_score_r3 - score all routes , return limited json
auto_route_data = {}
for ri,route in enumerate(geodata_routes):
    # refactor_multi_route_score_r4 - score all routes , return full json
    # returns pandas dataframe, only need score and gps coords
    auto_route_data[ri] = score_single_route( geodata_routes[ri])[['score','lat','lng']]

print('''

15:00 - until timer 20min: 
''')
print(" generate json with gps coords, score ")
# weird
if(verbose_score_manual_generic_route == 2):
    print("original:")
    # [x] TODOne: refactor to use 'geodata'
    pp.pprint(
            geodata['routes'][0]['overview_path']
            )
if(verbose_score_manual_generic_route == 3):
    print("new:")
    pp.pprint(
            auto_route_data[['score','latitude','longitude']].to_json()
            )

# closer, need to get rid of index and rename col
'''
original:
[{'lat': 30.288230000000002, 'lng': -97.73692000000001},
 {'lat': 30.289080000000002, 'lng': -97.73684000000002},
 {'lat': 30.28947, 'lng': -97.73679000000001},

In [27]: json.loads(auto_route_data[['score','latitude','longitude']].transpose().to_json())
Out[27]: 
{'0': {'latitude': 30.28823, 'longitude': -97.73692, 'score': 0.8702290076},
 '1': {'latitude': 30.28908, 'longitude': -97.73684, 'score': 0.8885793872},

# has the index:
In [28]: auto_route_data[['score','latitude','longitude']].transpose()
Out[28]: 
                  0          1          2          3          4          5   \
score       0.870229   0.888579   0.870229   0.909357   0.888579   0.909357   
latitude   30.288230  30.289080  30.289470  30.289550  30.289620  30.289630   
longitude -97.736920 -97.736840 -97.736790 -97.737980 -97.738950 -97.739130

# classic X-Y - 'transpose' was only in the way:
# src: https://stackoverflow.com/questions/28590663/pandas-dataframe-to-json-without-index
In [38]: json.loads(auto_route_data[['score','latitude','longitude']].to_json(orient='records'))
Out[38]: 
[{'latitude': 30.28823, 'longitude': -97.73692, 'score': 0.8702290076},
 {'latitude': 30.28908, 'longitude': -97.73684, 'score': 0.8885793872},
'''


print("--------------------------------------------------------------------------------")
print(" final response: ")
import json
# refactor_multi_route_score_r3 - score all routes , return limited json
# refactor_multi_route_score_r4 - score all routes , return full json
response_dict = {'routes' : [] , 'totalScores' : [] }
for ri,route in enumerate(auto_route_data):
    # add total score
    # total route score:
    # 1 - prob of nothing happening, i.e. 1 - product of "non-event" 
    print(1 - auto_route_data[ri]['score'].apply(lambda x: 1 - x).prod())
    # average route score:
    print(auto_route_data[ri]['score'].sum() / auto_route_data[ri]['score'].size)
    response_dict['totalScores'].append( auto_route_data[ri]['score'].sum() / auto_route_data[ri]['score'].size ) 
    #---
    # add segment scores
    # TODO: can omit the slice [['score','lat','lng']] as this is done further above already
    response_dict['routes'].append( auto_route_data[ri][['score','lat','lng']].to_dict(orient='records') )
response_json = json.dumps(response_dict)

print("response json")
pp.pprint( json.loads(
    response_json
))

# reply
print("save json to file. is mock equivalent of submitting json as a response")
if( mock_return_response_json( response_json ) ):
    print("json mock-response sent")
print("internal data structure") #, with only response variables")
# refactor_multi_route_score_r3 - score all routes , return limited json
# refactor_multi_route_score_r4 - score all routes , return full json
pp.pprint(
        auto_route_data # [0][['score','lat','lng']]
        )
if(verbose_score_manual_generic_route == 2):
    print("response data:")
    pp.pprint( json.loads(
        response_json
    ))
    print("request data:")
    pp.pprint(
            geodata['routes'][0]['overview_path']
            )


print("################################################################################")
print("-I-: " + "END - WORK_IN_PROGRESS - </score_manual_generic_route> ")
print("################################################################################")

if( options['verbose'] >= 0):
    print("################################################################################")
    print("-I-: End of File")
    print("################################################################################")
################################################################################
#________________________________________________________________________________
# FIN OVER FINITO STOP LOOKING FOR CODE
#--------------------------------------------------------------------------------

# miscellaneous
# TECH TIPS
## PANDAS
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

## Jupyter QTConsole
'''
%run code/model.py
shift+arr-up - as close to i-search-back as it gets
esc - clear entire input area, equivalent to ctrl+u in bash,csh
ctrl+l - issue on my machine, keeps causing cursor to get out of sync: 
    $ jupyter qtconsole
    QTextCursor::setPosition: Position '-1' out of range
    QTextCursor::setPosition: Position '11' out of range
    QTextCursor::setPosition: Position '-1' out of range                                                 
    QTextCursor::setPosition: Position '12' out of range 
    # sometimes even
    Segmentation fault (core dumped)

ctrl+. - restart kernel, clears out all of the vars that are set in the namespace. 
    due to the ctrl+l issue, it's not always clear whether kernel has restarted, or whether restart has hung.
    solution: just press 'esc'
'''
## MISC- ubuntu
'''
shift+prtscr - screenshot section
'''

'''
# look into dictvectorizer dv.get_feature_names http://stackoverflow.com/a/34194521
'''

# DM INFO

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
# pandas rename column
'''
src: https://stackoverflow.com/questions/11346283/renaming-columns-in-pandas?rq=1
In [50]: auto_route_data.rename(columns={'latitude':'lat','longitude':'lng'}, inplace=True)
/home/yoinkbird/devtools/miniconda3/lib/python3.6/site-packages/pandas/core/frame.py:2834: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
  **kwargs)
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


# Using the predictions?
#  vvv but then what? vvv
#  http://blog.yhat.com/posts/predicting-customer-churn-with-sklearn.html
#  # Make prediction
#  churn_prob = clf.predict_proba(X)
#  response['churn_prob'] = churn_prob[:,1]
#  # Return response DataFrame
#  return response

# worklog
# 20:2x - enabling other model
# 20:50 - fixed the bug from yesterday
# 20:51 - verified pickle working still

# IDEA - ensemble model of one with more data and one with less, e.g. one model has intersection info and one doesn't, can they be stacked?
# 20:58 - done refactoring
#jmisc stuff
# 22:53 - done with adding hard-coded mock client data


# 20171031
# 18:18 - start extract route geojson
# 18:21 - found steps[n][manuever]
# strategy: extract 'manuever' in get_gmap_direction_coords (need to refactor to extract_gmap_feats )
# map/convert to intersection_related
# arch: add extract_gmap_feats , let get_gmap_direction_coords 'filter out' gps coords from extract_gmap_feats 
# for now: just return a tuple with the updated info
# 18:41 - added maneuver to geodata
#         next: enable expanded model
# 18:48 - /break
# 19:12 - in the map! 
