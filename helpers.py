
# don't need most of these imports
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

def print_test():
    print("hi")
    return("hi")

# time conversions
# convert integer crashtime to datetime with year
# input: dataframe with year and time (int)
# todo: add month
def create_datetime_series(df):
    if('crash_month' in df):
        print("-E-: function can't handle months yet")
        return False
    return pd.to_datetime(df.apply(lambda x: "%s.%04d" % (x.crash_year,x.crash_time), axis=1),format="%Y.%H%M")
# convert to 24h time
# data.crash_time = data.crash_time.apply(lambda x: str(x).zfill(4)) # leading zeros
# could convert to datetime, but this forces a year,month,day to be present
# pd.to_datetime(data.crash_time.apply(lambda x: "2015%s"%x),format="%Y%H%M") # http://strftime.org/
# data.apply(lambda x: "%s%s" % (x.crash_year,x.crash_time), axis=1) # flexible year
# data['datetime'] = pd.to_datetime(data.crash_time.apply(lambda x: "2015%s"%x),format="%Y%H%M")
# src: http://stackoverflow.com/a/32375581
# pd.to_datetime(data.crash_time.apply(lambda x: "2015%s"%x),format="%Y%H%M").dt.time
# final:
# convert to decimal time
# src: https://en.wikipedia.org/wiki/Decimal_time#Scientific_decimal_time
# convert hours to fraction of day (HH/24) and minutes to fraction of day (mm/24*60), then add together
def time_base10(time):
    import pandas as pd
    time = pd.tslib.Timestamp(time)
    dech = time.hour/24; decm = time.minute/(24*60)
    #print("%s %f %f %f" % (time.time(), dech, decm, dech+decm))
    base10 = dech+decm
    return base10
def time_base10_to_60(time):
    verbose = 0
    # only round on final digit
    hours10 = time * 24  # 0.9 * 24  == 21.6
    hours10 = round(hours10, 5) # round out floating point issues
    hours24 = int(hours10)  # int(21.6) == 21
    min60 = round((hours10 * 60) % 60)     # 21.6*60 == 1296; 1296%60 == 36
    if(verbose):
        print("time: %f | hours24 %s | hours10 %s | min60 %s" % (time,hours24,hours10,min60))
    return hours24 * 100 + min60
# round to half hour
def time_round30min(pd_ts_time):
    import datetime
    pd_ts_time = pd.tslib.Timestamp(pd_ts_time)
    newtime = datetime.time()
    retmin = 61
    if(pd_ts_time.minute < 16):
        newtime = datetime.time(pd_ts_time.hour,0)
        retmin = 00
    elif((pd_ts_time.minute > 15) & (pd_ts_time.minute < 46)):
        newtime = datetime.time(pd_ts_time.hour,30)
        retmin = "30"
    elif(pd_ts_time.minute > 45):
        pd_ts_time += datetime.timedelta(hours=1)
        newtime = datetime.time(pd_ts_time.hour,00)
        retmin = 00
    #print("%s %s %f %f" % (pd_ts_time.pd_ts_time(), newtime, newtime.hour, newtime.minute))
    time_str = "%s.%02d%02d" % (pd_ts_time.year, newtime.hour, newtime.minute)
    # omit - would have to specify the year
    # time2 = pd.tslib.Timestamp("%02d:%02d" % (newtime.hour, newtime.minute))
    if(0):
        time2 = pd.to_datetime(time_str, format="%Y.%H%M")
    else:
        time_str = "%02d%02d" % (newtime.hour, newtime.minute)
        time2 = int(time_str)
    return time2

def get_ax_time(**kwargs):
    interval = '24h'
    title = ''
    xlabel = ''
    ylabel = ''
    if('title' in kwargs):
        title = kwargs['title']
    if('xlabel' in kwargs):
        xlabel = kwargs['xlabel']
    if('ylabel' in kwargs):
        ylabel = kwargs['ylabel']
    if('interval' in kwargs):
        interval = kwargs['interval']
    #######################################
    ax_time = plt.subplot(111)
    timelbl = []
    if(interval == '24h'):
        time_hrs = range(0,2400,200)
        for i in time_hrs:
            timelbl.append("%02d:%02d" % (i//100,i%100))
    ax_time.set_xticks(time_hrs)
    ax_time.set_xticklabels(timelbl, rotation=45, rotation_mode="anchor",ha="right")
    ax_time.set_title(title)
    ax_time.set_xlabel(xlabel)
    ax_time.set_ylabel(ylabel)
    return ax_time


# DOC: bar plot https://pythonspot.com/en/matplotlib-bar-chart/
# special bar charts:
# horiz 0 (default) : bar  chart with 45-degreee angled labels
# horiz 1           : hbar chart with largest value on top
# usage:
# pie chart - not very useful
# clf_imp_feats = print_model_feats_important(<model>, predictors)
# clf_imp_feats.value_counts().plot(kind='pie');plt.show()
# ax = get_ax_bar(clf_imp_feats, title="DecisionTree Important Features")
# plt.show()
# ax = get_ax_barh(clf_imp_feats, title="DecisionTree Important Features")
# plt.show()
def get_ax_bar(pdser, **kwargs):
    interval = '24h'
    horiz = 0
    title = ''
    xlabel = ''
    ylabel = ''
    if('title' in kwargs):
        title = kwargs['title']
    if('horiz' in kwargs):
        horiz = kwargs['horiz']
    if('xlabel' in kwargs):
        xlabel = kwargs['xlabel']
    if('ylabel' in kwargs):
        ylabel = kwargs['ylabel']
    if('interval' in kwargs):
        interval = kwargs['interval']
    #######################################
    ax_rot = plt.subplot(111)
    label = []
    for i,index_name in enumerate(pdser.index):
        label.append("%s : %0.6f" % (index_name, pdser[i]))
    if(horiz == 0):
      ax_rot.set_xticks(range(0,len(pdser.index)))
      ax_rot.set_xticklabels(pdser.index, rotation=45, rotation_mode="anchor", ha="right") ;
      ax_rot.bar(np.arange(len(pdser.index)), pdser.values)
      ax_rot.set_title(title)
    elif(horiz == 1):
      # reverse the labels, values to sort descending instead  of ascending
      # sort can be np.flip(<list>, axis=0) or <list>[::-1] DOC: reverse any list/array http://stackoverflow.com/questions/15748001/reversed-array-in-numpy
      ax_rot.set_yticks(range(0,len(pdser.index)));
      ax_rot.set_yticklabels(pdser.index[::-1])
      ax_rot.set_yticklabels(label[::-1])
      ax_rot.barh(np.flip(np.arange(len(pdser.index)), axis=0), pdser.values)
      ax_rot.set_title(title)

    ax_rot.set_xlabel(xlabel)
    ax_rot.set_ylabel(ylabel)
    return ax_rot

# horizontal barchart
def get_ax_barh(pdser, **kwargs):
    kwargs['horiz'] = 1
    return(get_ax_bar(pdser, **kwargs))
 

# src: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
#  confusion_matrix(y_test.values.ravel(),y_pred[:,1])
'''
# usage
from sklearn.metrics import confusion_matrix
#  print(confusion_matrix(y_full, pred_full))
cm = confusion_matrix(y_full, pred_full)
#  print(cm)
class_names=['down','up']
plot_confusion_matrix(cm, classes=class_names)
plt.show()
'''

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    import itertools
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, rotation_mode="anchor",ha="right")
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# DOC: How to interpret decision trees' graph results and find most informative features?
# src: http://stackoverflow.com/a/34872454
def print_model_feats_important(model, predictors, printout=1):
    ser = pd.Series()
    for i in np.argsort(model.feature_importances_)[::-1]:
      if model.feature_importances_[i] == 0:
        continue
      ser = ser.append(pd.Series([model.feature_importances_[i]], index=[predictors[i]]))
      if(printout):
        #print("%f : %s" % (model.feature_importances_[i],predictors[i]))
        print("%f : %s" % (ser.ix[predictors[i]],predictors[i]))
    return ser

def print_imp_feats_piecharts(data,featdef, model,predictors):
    # plot important features
    alreadyseen = {}
    for i in np.argsort(model.feature_importances_)[::-1]:
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

def ipynb_header(ipynb_file='Final.ipynb'):
    verbose = 0
    # print('<!-- http://sebastianraschka.com/Articles/2014_ipython_internal_links.html -->')
    import json
    with open(ipynb_file) as dataf:
      jsond = json.load(dataf)
    output = []
    if(verbose):
      print("# TOC - Table of Contents\n")
    output.append("# TOC - Table of Contents\n")
    output.append('<!-- generate with helpers.py -->')
    for cell in jsond['cells']:
        if(cell['cell_type'] == "markdown"):
            if('source' in cell):
                #print(cell['source'])
                import re
                cellysrc = cell['source']
                if(re.match('.*TOC - Table of Contents', cellysrc[0] )):
                  continue
                if(re.match('^#',cell['source'][0])):
                    #print("ermahgerd: %s" % cell['source'][0])
                    anchor = cellysrc[0].replace("\n",'')
                    lst = re.findall('^#+',anchor)[0].replace('#','* ')
                    anchor = re.sub('^\s*#+\s*','',anchor)
                    #print("became: %s" % anchor)
                    # vvv output: 
                    #print("<a id='%s'></a>" % (anchor.replace("\n",'').replace(" ",'-')))
                    # link:
                    #print("<a href='#%s'>%s</a><br/>" % ((anchor.replace(" ",'-'),anchor)))
                    if(verbose):
                      print("%s [%s](#%s)\n" % ((lst, anchor, anchor.replace(" ",'-'))))
                    output.append("%s [%s](#%s)\n" % ((lst, anchor, anchor.replace(" ",'-'))))
            else:
                print(cell.keys())
    print("\n".join(output))
    markdown_dict = {
        "cell_type": "markdown",
        "metadata": {},
        "source": "\n".join(output),
        },
    if(verbose):
      print(json.dumps(markdown_dict))

if(__name__ == '__main__'):
    test_timeconversion = 1
    ipynb_generate_toc = 1
    # testing - visual inspection
    if(test_timeconversion):
        print("verify correct operation of time_base10")
        # not testing 24:00 -> 1.0 because "hour must be in 0..23" for dateutil
        testtimes1 = ["0:00", "4:48"  , "7:12"  , "21:36" , "23:59"     , "0:59"      , "23:00"    ] # "24:00"
        testtimes2 = [0.0   , 0.2     , 0.3     , 0.9     , 0.999305556 , 0.040972222 , 0.958333333] # 1.0
        for i, testtime in enumerate(testtimes1):
            rettime = time_base10(testtime)
            status = "FAIL"
            # round for comparisson because floating point gets messy
            if(round(testtimes2[i],4) == round(rettime,4)):
                status = "PASS"
            print("%s: %6s: %s == %s ?" % (status, testtime , testtimes1[i] , rettime))
        print("verify correct operation of time_base10_to_60")
        for i, testtime in enumerate(testtimes2):
            status = "FAIL"
            rettime = time_base10_to_60(testtime)
            if(int(testtimes1[i].replace(':','')) == rettime):
                status = "PASS"
            print("%s: %6f: %s == %s ?" % (status, testtime , testtimes1[i] , rettime,))
    if(test_timeconversion):
        print("verify correct operation of time_round30min")
        testtimes1 = ["0:00" , "0:14" , "0:15" , "0:16", "0:29","0:30","0:31","0:44","0:45","0:46", "4:48"  , "7:12"  , "21:36" , "23:59"]
        testtimes2 = ["0:00" , "0:00" , "0:00" , "0:30", "0:30","0:30","0:30","0:30","0:30","1:00", "5:00"  , "7:00"  , "21:30" , "00:00"]
        for i, testtime in enumerate(testtimes1):
            #rettime = time_round30min(pd.tslib.Timestamp(testtime))
            rettime = time_round30min(testtime)
            status = "FAIL"
            if(int(testtimes2[i].replace(':','')) == rettime):
                status = "PASS"
            print("%s: %6s: %s == %s ?" % (status, testtime , testtimes2[i] , rettime))
    if(ipynb_generate_toc):
      print("######## START TOC markdown ############")
      ipynb_header()
      print("######## END   TOC markdown ############")
    if(1):
        print("-W-: NOT testing get_ax_time")
        print("-W-: NOT testing print_model_feats_important")
