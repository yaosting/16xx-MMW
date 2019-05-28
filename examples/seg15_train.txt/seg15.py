import numpy as np
import pandas as pd
import math
from datetime import datetime
def string_toDatetime(string):
    string = str(string)
    return datetime.strptime(string, '%H%M%S.%f')
    #return datetime.strptime(string, '%Y%m%d%H%M%S.%f')


FEATURE = ("mean", "max", "min", "std", "v_mean", "v_max", "v_min", "v_std")
STATUS  = ("42", "57", "56")
#status.index:0: 42--walking, 1:57--Fall, 2:56--Standing

f = pd.read_excel('./training_testing_data.xlsx',frame_name='All').sort_values('label')
print(f.head(5))
Seg_granularity = 15
gravity_data = []
#with open(file_dir) as f:
index = 0
for i in range(0,len(f)):#len(f)
    index = index + 1
    label  = f['label'][i]
    t = string_toDatetime(f['time'][i])
    #act_label = f['act_label'][i]
    #g = f['g'][i]
    accX = float(f['accX'][i])
    accY = float(f['accY'][i])
    print (index)
    velX = float(f['velX'][i])
    vely = float(f['vely'][i])
    
    if accX == 0 or accY == 0 or velX == 0 or vely == 0:
        continue
    gravity = math.sqrt(math.pow(accX, 2)+math.pow(accY, 2))
    velocity = math.sqrt(math.pow(velX, 2)+math.pow(vely, 2))
    gravity_tuple = {"gravity": gravity, "velocity":velocity, "label": label, "time":t}
    gravity_data.append(gravity_tuple)
    
splited_data = []
cur_cluster  = []
v_cluster = []
counter      = 0
last_status  = gravity_data[0]["label"]
last_time = gravity_data[0]["time"]
cur_time = gravity_data[0]["time"]
print(len(gravity_data))
for gravity_tuple in gravity_data:
    #print('gravity_tuple',gravity_tuple)
    #print('cur_time',cur_time)
    #print('last_time',last_time)
    #twindow = cur_time-last_time
    #print('twindow',twindow)
    if not (counter < Seg_granularity and gravity_tuple["label"] == last_status):
        #print(counter,last_status)
        seg_data = {"label": last_status, "values": cur_cluster, "v_values": v_cluster}
        #print('seg_data', seg_data)
        splited_data.append(seg_data)
        #print('splited_data',splited_data)
        cur_cluster = []
        v_cluster = []
        last_time = cur_time
        counter = 0
    cur_cluster.append(gravity_tuple["gravity"])
    v_cluster.append(gravity_tuple["velocity"])
    last_status = gravity_tuple["label"]
    cur_time = gravity_tuple['time']
    counter += 1
# compute statistics of gravity data
statistics_data = []
for seg_data in splited_data:
    np_values = np.array(seg_data.pop("values"))
    #print('np_values',np_values)
    seg_data["max"]  = np.amax(np_values)
    seg_data["min"]  = np.amin(np_values)
    seg_data["std"]  = np.std(np_values)
    seg_data["mean"] = np.mean(np_values)
    np_v_values = np.array(seg_data.pop("v_values"))
    seg_data["v_max"]  = np.amax(np_v_values)
    seg_data["v_min"]  = np.amin(np_v_values)
    seg_data["v_std"]  = np.std(np_v_values)
    seg_data["v_mean"] = np.mean(np_v_values)
    statistics_data.append(seg_data)
print('statsdata', len(statistics_data))
# write statistics result into a file in format of LibSVM
with open("./training_testing_data_svm_acc_vel_groupseg15.txt", "w") as the_file:
#with open("./WISDM_ar_v1.1/WISDM_ar_v1.1_raw_svm_2.txt", "a") as the_file:
    for segdata in statistics_data:
        print('seg_data_label',str(segdata["label"]))
        #row = str(STATUS.index(str(seg_data["label"])[3:5])) + " " + \

        row = str(STATUS.index(str(segdata["label"])[(len(str(segdata["label"]))-2):len(str(segdata["label"]))])+1) + " " + \
              str(FEATURE.index("mean")+1) + ":" + str(segdata["mean"]) + " " + \
              str(FEATURE.index("max")+1) + ":" + str(segdata["max"]) + " " + \
              str(FEATURE.index("min")+1) + ":" + str(segdata["min"]) + " " + \
              str(FEATURE.index("std")+1) + ":" + str(segdata["std"]) + " " + \
              str(FEATURE.index("v_mean")+1) + ":" + str(segdata["v_mean"]) + " " + \
              str(FEATURE.index("v_max")+1) + ":" + str(segdata["v_max"]) + " " + \
              str(FEATURE.index("v_min")+1) + ":" + str(segdata["v_min"]) + " " + \
              str(FEATURE.index("v_std")+1) + ":" + str(segdata["v_std"]) + "\n"
        #print(row)
        the_file.write(row)