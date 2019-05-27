#!/usr/bin/env python

import sys
import os
from subprocess import *
from svmutil import *

'''
if len(sys.argv) <= 1:
    print('Usage: {0} training_file [testing_file]'.format(sys.argv[0]))
    raise SystemExit

'''

is_win32 = (sys.platform == 'win32')
if not is_win32:
    svmscale_exe = "../svm-scale"
    svmtrain_exe = "../svm-train"
    svmpredict_exe = "../svm-predict"
    #grid_py = "./grid.py"
    #gnuplot_exe = "/usr/bin/gnuplot"
else:
        # example for windows
    svmscale_exe = r"..\windows\svm-scale.exe"
    svmtrain_exe = r"..\windows\svm-train.exe"
    svmpredict_exe = r"..\windows\svm-predict.exe"
    #gnuplot_exe = r"C:\Program Files\gnuplot\bin\gnuplot.exe"
    #grid_py = r"..\tools\grid.py"

assert os.path.exists(svmscale_exe),"svm-scale executable not found"
assert os.path.exists(svmtrain_exe),"svm-train executable not found"
assert os.path.exists(svmpredict_exe),"svm-predict executable not found"
#assert os.path.exists(gnuplot_exe),"gnuplot executable not found"
#assert os.path.exists(grid_py),"grid.py not found"

def predict(model_name,feature_file):
    #pred_pathname = './feature_10.txt'
    assert os.path.exists(feature_file),"predict file not found"
    scaled_file = feature_file + ".scale"
    #model_file = file_name + ".model"
    range_file = model_name + ".range"
    '''
    if len(sys.argv) > 2:
        test_pathname = sys.argv[2]
        file_name = os.path.split(test_pathname)[1]
        assert os.path.exists(test_pathname),"testing file not found"
        scaled_test_file = file_name + ".scale"
        predict_test_file = file_name + ".predict"

    '''

    #cmd = '{0} -s "{1}" "{2}" > "{3}"'.format(svmscale_exe, range_file, pred_pathname, scaled_file)
    cmd = '{0} -r "{1}" "{2}" > "{3}"'.format(svmscale_exe, range_file, feature_file, scaled_file)

    #print('Scaling predict data...')
    Popen(cmd, shell = True).communicate()#stdout = PIPE

    #behav = 'null'
    y,x = svm_read_problem(scaled_file)
    m = svm_load_model(str(model_name)+'.model')
    p_label, p_acc, p_val = svm_predict(y, x, m)
    return p_label
