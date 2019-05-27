from ti_mmwave.devices import IWR1642
from ti_mmwave.data_sources import SerialSource
import time
import datetime
#import json
from collections import OrderedDict
import pandas as pd
import numpy as np
import math
from svm_predict import predict
import sys

##
##options:
    

FEATURE = ("amean", "amax", "amin", "astd", "v_mean", "v_max", "v_min", "v_std")
config_source = SerialSource('/dev/ttyACM0', 115200)
data_source = SerialSource('/dev/ttyACM1', 921600)

device = IWR1642(config_source, data_source)
device.setup()
g = device.main()

argv0 = 2.5
argv1 = sys.argv[1]
start_time = datetime.datetime.now()
tid_set = set()
#locals()[]
while True:
    frame = next(g)
    #print(frame)
    cur_time = datetime.datetime.now()
    twindow = cur_time - start_time
    #print('twindow',twindow.seconds)
    if  ((twindow.seconds+twindow.microseconds/1000000) >= argv0 and len(frame.target_list) > 0):
        for td in tid_set:
            #print('write data tid and predict',td)
            #print(locals()['cluster_tid_'+str(td)])
            if len(locals()['cluster_tid_'+str(td)]) > 0:
                with open ('./feature_'+str(td)+'.txt','w') as f:
                    row = str("0") + " " + \
                      str(FEATURE.index("amean")+1) + ":" + str(np.mean(locals()['cluster_tid_'+str(td)]['gravity'])) + " " + \
                      str(FEATURE.index("amax")+1) + ":" + str(np.amax(locals()['cluster_tid_'+str(td)]['gravity'])) + " " + \
                      str(FEATURE.index("amin")+1) + ":" + str(np.amin(locals()['cluster_tid_'+str(td)]['gravity'])) + " " + \
                      str(FEATURE.index("astd")+1) + ":" + str(np.std(locals()['cluster_tid_'+str(td)]['gravity'])) + " " + \
                      str(FEATURE.index("v_mean")+1) + ":" + str(np.mean(locals()['cluster_tid_'+str(td)]['velocity'])) + " " + \
                      str(FEATURE.index("v_max")+1) + ":" + str(np.amax(locals()['cluster_tid_'+str(td)]['velocity'])) + " " + \
                      str(FEATURE.index("v_min")+1) + ":" + str(np.amin(locals()['cluster_tid_'+str(td)]['velocity'])) + " " + \
                      str(FEATURE.index("v_std")+1) + ":" + str(np.std(locals()['cluster_tid_'+str(td)]['velocity'])) + "\n"
                    #print('tid_row',td, row)
                    f.write(row)
                    f.close()
                #predict the behavior
                p = predict(model_name='../examples/'+argv1+'/'+argv1,feature_file='./feature_'+str(td)+'.txt') #model=argv[1]
                print(p)
                if p[0] == 1:
                    behavior = 'walking'
                elif p[0] == 2:
                    behavior = 'falling down'
                elif p[0] == 3:
                    behavior = 'standing'
                print(str(cur_time.strftime('%Y_%m_%d.%H_%M_%S.%f')),'Target '+str(td)+' is '+ behavior)
            locals()['cluster_tid_'+str(td)] = []
        tid_set.clear()
        start_time = cur_time
        #print
    elif ((twindow.seconds+twindow.microseconds/1000000) < argv0 and len(frame.target_list) > 0):
        for i in range(0, len(frame.target_list)):
            if frame.target_list[i].acc_x ==0 or frame.target_list[i].acc_y ==0 or frame.target_list[i].vel_x ==0 or frame.target_list[i].vel_y ==0:
                continue
            tid = frame.target_list[i].tid
            #print('tid',tid)
            if tid not in tid_set:
                grvt = math.sqrt(math.pow(frame.target_list[i].acc_x, 2)+math.pow(frame.target_list[i].acc_y, 2))
                vlct = math.sqrt(math.pow(frame.target_list[i].vel_x, 2)+math.pow(frame.target_list[i].vel_y, 2))
                locals()['cluster_tid_'+str(frame.target_list[i].tid)] = pd.DataFrame({
                    "label": ['0'],
                    "time": [cur_time.strftime('%Y_%m_%d.%H_%M_%S.%f')],
                    "accx": [frame.target_list[i].acc_x],
                    "accy": [frame.target_list[i].acc_y],
                    "velx": [frame.target_list[i].vel_x],
                    "vely": [frame.target_list[i].vel_y],
                    "posx": [frame.target_list[i].pos_x],
                    "posy": [frame.target_list[i].pos_y],
                    "tid": [frame.target_list[i].tid],
                    "g": [frame.target_list[i].g],
                    "ec": [frame.target_list[i].ec],
                    "gravity": [grvt],
                    "velocity": [vlct]
                })
                '''
                locals()['cluster_tid_'+str(frame.target_list[i].tid)] = pd.DataFrame({
                    "label": [],
                    "time": [],
                    "accx": [],
                    "accy": [],
                    "velx": [],
                    "vely": [],
                    "posx": [],
                    "posy": [],
                    "tid": [],
                    "g": [],
                    "ec": [],
                    "gravity": [],
                    "velocity": []
                })
                '''
                tid_set.add(tid)
                #print('tid_set',tid_set)
            else:
                grvt = math.sqrt(math.pow(frame.target_list[i].acc_x, 2)+math.pow(frame.target_list[i].acc_y, 2))
                vlct = math.sqrt(math.pow(frame.target_list[i].vel_x, 2)+math.pow(frame.target_list[i].vel_y, 2))
                data = pd.DataFrame({
                    "label": ['0'],
                    "time": [cur_time.strftime('%Y_%m_%d.%H_%M_%S.%f')],
                    "accx": [frame.target_list[i].acc_x],
                    "accy": [frame.target_list[i].acc_y],
                    "velx": [frame.target_list[i].vel_x],
                    "vely": [frame.target_list[i].vel_y],
                    "posx": [frame.target_list[i].pos_x],
                    "posy": [frame.target_list[i].pos_y],
                    "tid": [frame.target_list[i].tid],
                    "g": [frame.target_list[i].g],
                    "ec": [frame.target_list[i].ec],
                    "gravity": [grvt],
                    "velocity": [vlct]
                })
                locals()['cluster_tid_'+str(tid)]=locals()['cluster_tid_'+str(tid)].append(data)
                #print(locals()['cluster_tid_'+str(tid)])
