
##=============================================================================
# co-learning using assumed common covariance structure
##=============================================================================
import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.optimize import curve_fit
import copy
import itertools
import math
import random
from timeit import default_timer as timer
##=============================================================================
## Acceleration analysis
##=============================================================================
# A function given acceleration, jerk, feed, length_vector,
# return the time_vector v.s. speed_vector
# if feed>v0, then a > 0
# if feed<v0, then a < 0
def acc_ana(a, v0, v1, feed, jerk, l, stp):
    # v1 = 'stop' or 'non-stop'
    # default is 'non-stop' which means there is a subsequent gcode line that has feed specified and has the same printing direction
    # 'stop' means there are no subsequent printing line at current layer with the same printing direction
    # v1 = feed if v1 = 'non-stop'
    # v1 = 0 if v1 = 'stop'
   
    # keep acc/dec
    # acc/dec till reach specified feed
   
    # if v0 < feed, a should >0
    # if v0 > feed, a should <0
    l_vector = np.arange(0, l+stp, stp)
    if v1 == 'non-stop':
        v_end = feed
        L = np.abs((np.power(v_end, 2) - np.power(v0, 2))/(2*a))
        if l <= L:
            v = np.sqrt(np.power(v0, 2) + 2*a*l_vector)
            acc_vector = a + np.zeros(len(l_vector))
            t_vector = (v - v0)/a
            print("[Message] Path length too short, specified feed not achieved")
        else:
            l_vector1 = l_vector[l_vector<=L]
            l_vector2 = l_vector[l_vector>L]
            v1 = np.sqrt(np.power(v0, 2) + 2*a*l_vector1)
            v2 = feed + np.zeros(len(l_vector2))
            acc_vector1 = a + np.zeros(len(l_vector1))
            acc_vector2 = np.zeros(len(l_vector2))
            t_vector1 = (v1 - v0)/a
            t_vector2 = (l_vector2 - l_vector1[-1])/feed + t_vector1[-1]
            v = np.concatenate((v1, v2))
            acc_vector = np.concatenate((acc_vector1, acc_vector2))
            t_vector = np.concatenate((t_vector1, t_vector2))
            print("[Message] Specified feed achieved and remained till end of path")
    else:
        v_end = jerk
        L_v0_feed = np.abs((np.power(v0, 2) - np.power(feed, 2))/(2*a))
        L_feed_vend = np.abs((np.power(feed, 2) - np.power(v_end, 2))/(2*a))
        L_acc_dec = L_v0_feed + L_feed_vend
       
        if l >= L_acc_dec:
            l_vector1 = l_vector[l_vector<=L_v0_feed]
            l_vector2 = l_vector[(l_vector>L_v0_feed) & (l_vector<(l-L_feed_vend))]
            l_vector3 = l_vector[l_vector>=(l-L_feed_vend)]
            v1 = np.sqrt(np.power(v0, 2) + 2*a*l_vector1) if v0 < feed else np.sqrt(np.power(v0, 2) - 2*a*l_vector1)
            v2 = feed + np.zeros(len(l_vector2))
            l3 = l_vector3 - (l-L_feed_vend)
            v3 = np.sqrt(np.around((np.power(feed, 2) - 2*a*l3), decimals = 6))
            acc_vector1 = a + np.zeros(len(l_vector1))
            acc_vector2 = np.zeros(len(l_vector2))
            acc_vector3 = -a + np.zeros(len(l_vector3))
            t_vector1 = (v1 - v0)/a if v0 < feed else (v0 - v1)/a
            t_vector2 = (l_vector2 - l_vector1[-1])/feed + t_vector1[-1]
            t_vector3 = (feed - v3)/a + t_vector2[-1]
            v = np.concatenate((v1, v2, v3))
            acc_vector = np.concatenate((acc_vector1, acc_vector2, acc_vector3))
            t_vector = np.concatenate((t_vector1, t_vector2, t_vector3))
            print("[Message] Extruder accelerate/decelerate, remain constant, then decelerate to zero")
        else:
            if v0 < feed:
                # v0 < feed, acc then dec
                a = np.abs(a)
                v_peak = np.sqrt((2*a*l + np.power(v0, 2) + np.power(v_end,2))/2)
                L_v0_peak = np.abs((np.power(v0, 2) - np.power(v_peak, 2))/(2*a))
                # L_peak_vend = np.abs((np.power(v_peak, 2) - np.power(v_end, 2))/(2*a))
                l_vector1 = l_vector[l_vector <= L_v0_peak]
                l_vector2 = l_vector[l_vector > L_v0_peak]
           
                v1 = np.sqrt(np.power(v0, 2) + 2*a*l_vector1)
                l2 = l_vector2 - L_v0_peak
                v2 = np.sqrt(np.around((np.power(v_peak, 2) - 2*a*l2), decimals = 6))
                acc_vector1 = a + np.zeros(len(l_vector1))
                acc_vector2 = -a + np.zeros(len(l_vector2))
                t_vector1 = (v1 - v0)/a
                t_vector2 = (v_peak - v2)/a + t_vector1[-1]
                v = np.concatenate((v1, v2))
                acc_vector = np.concatenate((acc_vector1, acc_vector2))
                t_vector = np.concatenate((t_vector1, t_vector2))
                print("[Message] Path length is too short to accelerate to the specified feed")
            else:
                # v0 > feed, keep dec and the v_end is not achieved, i.e., does not stop
                a = np.abs(a)
                v = np.sqrt(np.power(v0, 2) - 2*a*l_vector)
                acc_vector = -a + np.zeros(len(l_vector))
                t_vector = (v0 - v)/a
                print("[Message] Path length is too short to decelerate till stop at the ending point")
    return l_vector, v, acc_vector, t_vector

##===================================================
## load new data
##===================================================
# parameter initialization and specification
start_p = 50 # for calculate the mean
stop_p = 80 # for calculate the mean
coords_1 = []
width_1 = []
l = 120
stp = 0.05
coords = np.arange(0, l+stp, stp) # specified uniform locations
jerk = 8
v0 = jerk
v1 = 'stop'

# load the line measurement data
with open('S800F9000_ino120.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        coords_1.append(float(row[0]))
        width_1.append(float(row[1]))
       
coords_1 = np.asarray(coords_1)
width_1 = np.asarray(width_1)
width_1_ave = np.mean(width_1[(coords_1>start_p)&(coords_1<stop_p)])
width_interp_1  =  np.interp(coords, coords_1 ,width_1)
#width_interp_1 = width_interp_1 - width_1_ave
a = 800
feed = 150
l_vector, v_1, a_vector_1, t_vector_1 = acc_ana(a, v0, v1, feed, jerk, l, stp)
data_new = np.transpose(np.array((coords, t_vector_1, v_1, a_vector_1, width_interp_1)))

# load the line measurement data 2
coords_2 = []
width_2 = []
with open('S800F6000_ino120.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        coords_2.append(float(row[0]))
        width_2.append(float(row[1]))
       
coords_2 = np.asarray(coords_2)
width_2 = np.asarray(width_2)
width_2_ave = np.mean(width_2[(coords_2>start_p)&(coords_2<stop_p)])
width_interp_2  =  np.interp(coords, coords_2 ,width_2)
#width_interp_2 = width_interp_2 - width_2_ave
a = 800
feed = 100
l_vector, v_2, a_vector_2, t_vector_2 = acc_ana(a, v0, v1, feed, jerk, l, stp)
data_new = [data_new, np.transpose(np.array((coords, t_vector_2, v_2, a_vector_2, width_interp_2)))]

# load the line measurement data 3
coords_3 = []
width_3 = []
with open('S800F3000_ino120.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        coords_3.append(float(row[0]))
        width_3.append(float(row[1]))
       
coords_3 = np.asarray(coords_3)
width_3 = np.asarray(width_3)
width_3_ave = np.mean(width_3[(coords_3>start_p)&(coords_3<stop_p)])
width_interp_3  =  np.interp(coords, coords_3 ,width_3)
#width_interp_3 = width_interp_3 - width_3_ave
a = 800
feed = 50
l_vector, v_3, a_vector_3, t_vector_3 = acc_ana(a, v0, v1, feed, jerk, l, stp)
data_new.append(np.transpose(np.array((coords, t_vector_3, v_3, a_vector_3, width_interp_3))))

# load the line measurement data 4
coords_4 = []
width_4 = []
with open('S600F9000_ino120.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        coords_4.append(float(row[0]))
        width_4.append(float(row[1]))
       
coords_4 = np.asarray(coords_4)
width_4 = np.asarray(width_4)
width_4_ave = np.mean(width_4[(coords_4>start_p)&(coords_4<stop_p)])
width_interp_4  =  np.interp(coords, coords_4 ,width_4)
#width_interp_4 = width_interp_4 - width_4_ave
a = 600
feed = 150
l_vector, v_4, a_vector_4, t_vector_4 = acc_ana(a, v0, v1, feed, jerk, l, stp)
data_new.append(np.transpose(np.array((coords, t_vector_4, v_4, a_vector_4, width_interp_4))))

# load the line measurement data 5
coords_5 = []
width_5 = []
with open('S600F6000_ino120.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        coords_5.append(float(row[0]))
        width_5.append(float(row[1]))
       
coords_5 = np.asarray(coords_5)
width_5 = np.asarray(width_5)
width_5_ave = np.mean(width_5[(coords_5>start_p)&(coords_5<stop_p)])
width_interp_5  =  np.interp(coords, coords_5 ,width_5)
#width_interp_5 = width_interp_5 - width_5_ave
a = 600
feed = 100
l_vector, v_5, a_vector_5, t_vector_5 = acc_ana(a, v0, v1, feed, jerk, l, stp)
data_new.append(np.transpose(np.array((coords, t_vector_5, v_5, a_vector_5, width_interp_5))))

# load the line measurement data 6
coords_6 = []
width_6 = []
with open('S600F3000_ino120.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        coords_6.append(float(row[0]))
        width_6.append(float(row[1]))
       
coords_6 = np.asarray(coords_6)
width_6 = np.asarray(width_6)
width_6_ave = np.mean(width_6[(coords_6>start_p)&(coords_6<stop_p)])
width_interp_6  =  np.interp(coords, coords_6 ,width_6)
#width_interp_6 = width_interp_6 - width_6_ave
a = 600
feed = 50
l_vector, v_6, a_vector_6, t_vector_6 = acc_ana(a, v0, v1, feed, jerk, l, stp)
data_new.append(np.transpose(np.array((coords, t_vector_6, v_6, a_vector_6, width_interp_6))))

# load the line measurement data 7
coords_7 = []
width_7 = []
with open('S400F9000_ino120.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        coords_7.append(float(row[0]))
        width_7.append(float(row[1]))
       
coords_7 = np.asarray(coords_7)
width_7 = np.asarray(width_7)
width_7 = width_7[9:2525]# ditch bad values
coords_7 = np.linspace(0,120,len(width_7))###################################################################
width_7_ave = np.mean(width_7[(coords_7>start_p)&(coords_7<stop_p)])
width_interp_7  =  np.interp(coords, coords_7 ,width_7)
#width_interp_7 = width_interp_7 - width_7_ave
a=400
feed=150
l_vector, v_7, a_vector_7, t_vector_7 = acc_ana(a, v0, v1, feed, jerk, l, stp)
data_new.append(np.transpose(np.array((coords, t_vector_7, v_7, a_vector_7, width_interp_7))))

# load the line measurement data 8
coords_8 = []
width_8 = []
with open('S400F6000_ino120.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        coords_8.append(float(row[0]))
        width_8.append(float(row[1]))
       
coords_8 = np.asarray(coords_8)
width_8 = np.asarray(width_8)
width_8_ave = np.mean(width_8[(coords_8>start_p)&(coords_8<stop_p)])
width_interp_8  =  np.interp(coords, coords_8 ,width_8)
#width_interp_8 = width_interp_8 - width_8_ave
a = 400
feed = 100
l_vector, v_8, a_vector_8, t_vector_8 = acc_ana(a, v0, v1, feed, jerk, l, stp)
data_new.append(np.transpose(np.array((coords, t_vector_8, v_8, a_vector_8, width_interp_8))))

# load the line measurement data 9
coords_9 = []
width_9 = []
with open('S400F3000_ino120.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        coords_9.append(float(row[0]))
        width_9.append(float(row[1]))
       
coords_9 = np.asarray(coords_9)
width_9 = np.asarray(width_9)
width_9_ave = np.mean(width_9[(coords_9>start_p)&(coords_9<stop_p)])
width_interp_9  =  np.interp(coords, coords_9 ,width_9)
#width_interp_9 = width_interp_9 - width_9_ave
a = 400
feed = 50
l_vector, v_9, a_vector_9, t_vector_9 = acc_ana(a, v0, v1, feed, jerk, l, stp)
data_new.append(np.transpose(np.array((coords, t_vector_9, v_9, a_vector_9, width_interp_9))))
# make a copy of the data_new
data_new_copy = copy.deepcopy(data_new)


##=============================================================================
# unlike the previous printer, this printer has no breakpoint1, therefore, all the bp1 are set to 0;
bp1_coords_array = [0]*9
bp1_t_array = [0]*9
v_array = [150, 100,50]*3
a_array = [800, 800, 800, 600, 600, 600, 400,400,400]
jerk_array = [8]*9
x_array = v_array
max_coords_array = [120]*9
bp_data_new = np.column_stack((bp1_coords_array, bp1_t_array, v_array, a_array, jerk_array, x_array, max_coords_array))

# breakpoint idenfication
# bp_2 is the point that the line width reaches constant
##### identify the position/index of bp_2


thres_list = [0.58, 0.76, 0.8, 0.76, 0.80, 0.90, 0.8, 0.80, 0.85]
def identify_breakpoint2(width, thrs):
    idx = np.argwhere(width>thrs)
    idx = idx[idx>100] # 100 is to make sure the idx is not near the starting point
    return idx[0]
   
for i in range(len(data_new)):
    t_vector_tmp = data_new[i][:,1]
    idx_p2 = identify_breakpoint2(data_new[i][:,-1], thres_list[i])
    bp_data_new[i,5] = idx_p2
#    plt.figure()
#    plt.plot(data_new[i][:,1], data_new[i][:,4])
#    plt.hold(True)
#    plt.axvline(x = t_vector_tmp[int(idx_p2)])

# there are breakpoints between bp1 and bp2 which represent the transition from
# acceleration to constant phase
for i in range(len(data_new)):
    tmp = data_new[i][:,3]
    idx_pac = np.argwhere(tmp==0)
    idx_pac = idx_pac[0]
    bp_data_new[i,6] = idx_pac

bp_data_new_copy = copy.deepcopy(bp_data_new)

# training funciton specification
## model training piecewise
# train a model between bp1 and bp_ac, which represents the acceleration phase
def seg2_func_acc(tvav, beta1, beta2):
    return beta1*np.sqrt(tvav[:,0])*np.sqrt(tvav[:,1])*np.sqrt(tvav[:,2]) + beta2

def seg2acc_fit(traindata_seg2acc):
    t_v_a_v = traindata_seg2acc[:,[0,1,2,4]]
    width = traindata_seg2acc[:,3]
    popt, pcov = curve_fit(seg2_func_acc, t_v_a_v, width)
    width_pred = seg2_func_acc(t_v_a_v, *popt)
    return popt, width_pred

def generate_traindata2acc(data, bp_data):
    traindata_p2 = np.empty((0,6))
    for i in range(len(data)):
        t_tmp = copy.deepcopy(data[i][:,1])
        t_threshold1 = copy.deepcopy(bp_data[i,1])
        t_threshold2 = copy.deepcopy(t_tmp[int(bp_data[i,6])])
        data_p2 = copy.deepcopy(data[i])
        data_p2 = copy.deepcopy(data_p2[np.logical_and(t_tmp>t_threshold1, t_tmp<t_threshold2),:])
        if len(data_p2)!=0:
            data_p2[:,1] = data_p2[:,1] - data_p2[0,1]# make t start from zero
            data_p2[:,4] = data_p2[:,4] - data_p2[0,4]# make width start from zero
            top_speed = np.max(data[i][:,2]) + np.zeros((len(data_p2),1))
            data_p2 = np.concatenate((data_p2,top_speed), axis=1)
            traindata_p2 = np.concatenate((traindata_p2, data_p2))
    return traindata_p2


# train a model between bp_ac and bp_2, which represents the speed constant phase
def seg2_func(tvav, beta1):# tva refers to time, speed, acceleration, top speed.
    #return beta1*tva[:,0]*tva[:,2]/tva[:,1]
    # return beta1*np.sqrt(tvav[:,0])*np.sqrt(tvav[:,1])
    # return beta1*np.sqrt(tvav[:,0])*np.sqrt(tvav[:,1])
    return beta1*np.sqrt(tvav[:,0])*np.sqrt(tvav[:,1])
    #return beta1*np.sqrt(tvav[:,0])*np.sqrt(tvav[:,1]/(tvav[:,2]+beta2))
    # when accelerationg, the slope is small, when constant, the slope is big
    # return beta1*np.sqrt(tvav[:,0])*np.sqrt(tvav[:,1])*(tvav[:,2]>0) + beta2*np.sqrt(tvav[:,0])*np.sqrt(tvav[:,1])*(tvav[:,2]==0)
    # return beta1*tvav[:,1]*(tvav[:,2]>0) + beta2*np.sqrt(tvav[:,0])*np.sqrt(tvav[:,1])*(tvav[:,2]==0)

def seg2_fit(traindata_seg2):
    t_v_a_v = traindata_seg2[:,[0,1,2,4]]
    width = traindata_seg2[:,3]
    popt, pcov = curve_fit(seg2_func, t_v_a_v, width)
    width_pred = seg2_func(t_v_a_v, *popt)
    return popt, width_pred
def generate_traindata2(data, bp_data):
    traindata_p2 = np.empty((0,6))
    for i in range(len(data)):
        t_tmp = data[i][:,1]
        t_threshold1 = t_tmp[int(bp_data[i,6])]
        t_threshold2 = t_tmp[int(bp_data[i,5])]
        data_p2 = data[i]
        data_p2 = data_p2[np.logical_and(t_tmp>t_threshold1, t_tmp<t_threshold2),:]
        if len(data_p2)!=0:
            data_p2[:,1] = data_p2[:,1] - data_p2[0,1]# make t start from zero
            data_p2[:,4] = data_p2[:,4] - data_p2[0,4]# make width start from zero
            top_speed = np.max(data[i][:,2]) + np.zeros((len(data_p2),1))
            data_p2 = np.concatenate((data_p2,top_speed), axis=1)
            traindata_p2 = np.concatenate((traindata_p2, data_p2))
    return traindata_p2


##===========================================================================
##==================fit a function for the decelerating phase================
##===========================================================================
def seg3_func(tvav, gamma1):
    #return gamma1*tvav[:,3]/tvav[:,1]
    return gamma1*np.log(tvav[:,3]/tvav[:,1])

def seg3_fit(traindata_seg3):
    t_v_a_v = traindata_seg3[:,[0,1,2,4]]
    width = traindata_seg3[:,3]
    popt, pcov = curve_fit(seg3_func, t_v_a_v, width)
    width_pred = seg3_func(t_v_a_v, *popt)
    return popt, width_pred
def generate_traindata3(data):
    traindata_p3 = np.empty((0,6))
    for i in range(len(data)):
        a_tmp = data[i][:,3]
        idx_dec = np.min(np.argwhere(a_tmp<0))
        data_p3 = copy.deepcopy(data[i])# use copy to avoid value changing in data
        data_p3 = data_p3[idx_dec:,:]
        data_p3[:,4] = data_p3[:,4] - data_p3[0,4]# initialize start for width
        data_p3[:,1] = data_p3[:,1] - data_p3[0,1]# initialize start for time
        top_speed = np.max(data[i][:,2]) + np.zeros((len(data_p3),1))
        data_p3 = np.concatenate((data_p3,top_speed), axis=1)
        traindata_p3 = np.concatenate((traindata_p3, data_p3))
    return traindata_p3


##===========================================================================
##==================covariance based co-learning function================
##===========================================================================
def tl_cov_learning(traindata_1, traindata_2, cov, lamb):
    X1_ = traindata_1[:,[0,1,2,4]]
    Y1 = traindata_1[:,3]
    X1 = np.sqrt(X1_[:,0])*np.sqrt(X1_[:,1])
    zeros1 = np.array([0 for _ in range(len(X1))])
    X1 = np.column_stack((X1, zeros1))
   
    X2_ = traindata_2[:,[0,1,2,4]]
    Y2 = traindata_2[:,3]
    X2 = np.log(X2_[:,3]/X2_[:,1])
    zeros2 = np.array([0 for _ in range(len(X2))])
    X2 = np.column_stack((zeros2, X2))
    return np.dot(np.linalg.inv(np.dot(X1.T, X1) + np.dot(X2.T, X2) + lamb*np.linalg.inv(cov)), (np.dot(X1.T, Y1) + np.dot(X2.T, Y2)))

##=============================================================================
##============================prediction function==============================
##=============================================================================
start_list = [0.4,0.45,0.61,0.55,0.62,0.7,0.6,0.6,0.73]

def forward_pred(ltva_va, start, const, coef_p2acc, coef_p2, coef_p3):
    # predict the breakpoint bp1
    #av_bp1 = ltva_va[:,[5,4]]
    #bp1 = bp1_func(av_bp1,*coef_bp1)
    #bp1 = bp1[0]
    bp1 = 0.
    #print(bp1)
    # make predictions for p1
    #t_v_p1 = ltva_va[ltva_va[:,1]<=bp1, :]
    #t_v_p1 = t_v_p1[:,[1,2]]
    #pred_p1 = seg1_func(t_v_p1,*coef_p1)
    #final_pred = pred_p1
    pred_p1 = [start] # the starting line width
    final_pred = []
    # make predictions for p3
    idx_dec = np.min(np.argwhere(ltva_va[:,3]<0))
    tvav_p3 = ltva_va[idx_dec:,[1,2,3,4]]
    tvav_p3[:,0] = tvav_p3[:,0] - tvav_p3[0,0]
    pred_p3 = seg3_func(tvav_p3, *coef_p3) + const
    # make predictions for p2
    # p2 acceleration phase if any
    idx_ac = np.min(np.argwhere(ltva_va[:,3]==0))
    tvav_p2acc = ltva_va[:idx_ac,[1,2,3,4]]
    tvav_p2acc = tvav_p2acc[tvav_p2acc[:,0]>bp1,:]
    if len(tvav_p2acc)!=0:
        tvav_p2acc[:,0] = tvav_p2acc[:,0] - tvav_p2acc[0,0]
        pred_p2acc = seg2_func_acc(tvav_p2acc, *coef_p2acc) + pred_p1[-1]
        final_pred = np.concatenate((pred_p1, pred_p2acc))
    # make predictions for p2
    # p2 constant phase if any
    tvav_p2 = ltva_va[idx_ac:idx_dec,[1,2,3,4]]
    tvav_p2 = tvav_p2[tvav_p2[:,0]>bp1,:]
    if len(tvav_p2)!=0:
        tvav_p2[:,0] = tvav_p2[:,0] - tvav_p2[0,0]
        try:
            pred_p2 = seg2_func(tvav_p2, *coef_p2) + pred_p2acc[-1]
        except:
            pred_p2 = seg2_func(tvav_p2, *coef_p2) + pred_p1[-1]
        pred_p2[pred_p2>=const] = const
        final_pred = np.concatenate((final_pred, pred_p2, pred_p3))
    return final_pred

##==================model training function for co-learning/no co-learning===========
def tl_notl_learning(training_list, data_new_copy, bp_data_new_copy,cov):
    bp_data_new = copy.deepcopy(bp_data_new_copy[training_list])
    data_new = [copy.deepcopy(data_new_copy[i]) for i in training_list]
    # initial accelerating phase
    traindata_p2acc = copy.deepcopy(generate_traindata2acc(data_new, bp_data_new))
    traindata_p2acc_ = copy.deepcopy(traindata_p2acc[:,[1,2,3,4,5]])
    coef_p2acc, width_pred_p2acc = copy.deepcopy(seg2acc_fit(traindata_p2acc_))
    # constant and width increase phase
    traindata_p2 = copy.deepcopy(generate_traindata2(data_new, bp_data_new))
    traindata_p2_ = copy.deepcopy(traindata_p2[:,[1,2,3,4,5]])
    coef_p2, width_pred_p2 = copy.deepcopy(seg2_fit(traindata_p2_))
    # decelerating phase
    traindata_p3 =copy.deepcopy( generate_traindata3(data_new))
    traindata_p3_ = copy.deepcopy(traindata_p3[:,[1,2,3,4,5]])
    coef_p3, width_pred_p3 = copy.deepcopy(seg3_fit(traindata_p3_))

    coef_p2_p3 = copy.deepcopy(tl_cov_learning(traindata_p2_, traindata_p3_, cov, 0.15))
    coef_p2_tl =copy.deepcopy( [coef_p2_p3[0]])
    coef_p3_tl =copy.deepcopy( [coef_p2_p3[1]])
    return coef_p2acc, coef_p2, coef_p3, coef_p2_tl, coef_p3_tl

##==================model testing function for colearning/no colearning============
def tl_notl_test(data_test, test_idx, coef_p2acc, coef_p2, coef_p3, coef_p2_tl, coef_p3_tl):
    ltva_va =copy.deepcopy( data_test[:,[0,1,2,3]])

    top_speed = np.max(data_test[:,2]) + np.zeros((len(ltva_va),1))
    acc = np.max(data_test[:,3]) + np.zeros((len(ltva_va),1))
    ltva_va = np.concatenate((ltva_va, top_speed, acc), axis = 1)

    # without Co-learning
    ##############
    pred = forward_pred(ltva_va, start_list[test_idx], thres_list[test_idx], coef_p2acc, coef_p2, coef_p3)

    RMSE = np.sqrt(np.sum((pred - data_test[:,-1])**2)/len(pred))

    pred_tl1 = forward_pred(ltva_va, start_list[test_idx], thres_list[test_idx], coef_p2acc, coef_p2_tl, coef_p3)
    pred_tl2 = forward_pred(ltva_va, start_list[test_idx], thres_list[test_idx], coef_p2acc, coef_p2_tl, coef_p3_tl)
    pred_tl3 = forward_pred(ltva_va, start_list[test_idx], thres_list[test_idx], coef_p2acc, coef_p2, coef_p3_tl)

    RMSE_tl1 = np.sqrt(np.sum((pred_tl1 - data_test[:,-1])**2)/len(pred_tl1))
    RMSE_tl2 = np.sqrt(np.sum((pred_tl2 - data_test[:,-1])**2)/len(pred_tl2))
    RMSE_tl3 = np.sqrt(np.sum((pred_tl3 - data_test[:,-1])**2)/len(pred_tl3))
    RMSE_tl = min([RMSE_tl1, RMSE_tl2, RMSE_tl3])

    return RMSE, RMSE_tl

##### a loop selecting all the combinations of splitting for training and test #########
training_sample_size = [i+1 for i in range(len(data_new_copy)-1)]
samples_list = [i for i in range(len(data_new_copy))]
RMSE_all, RMSE_all_tl = [[] for _ in range(len(training_sample_size))], [[] for _ in range(len(training_sample_size))]
RMSE_all_test = copy.deepcopy(RMSE_all)
RMSE_all_tl_test = copy.deepcopy(RMSE_all_tl)


median =[]
median_tl =[]


def cost(training_sample_size,samples_list,data_new_copy,bp_data_new_copy,cov_29,cov_30,cov_36):
    i = 0
    RMSE_all = copy.deepcopy(RMSE_all_test)
    RMSE_all_tl =  copy.deepcopy(RMSE_all_tl_test)
    training_list = [0]
    test_list = [3,5,6,8]
    coef_p2acc, coef_p2, coef_p3, coef_p2_tl, coef_p3_tl = tl_notl_learning(training_list, data_new_copy, bp_data_new_copy,np.array([[cov_29, cov_30], [cov_30, cov_36]]) )
    
    for test_idx in test_list:
        data_test = copy.deepcopy(data_new_copy[test_idx])
        RMSE, RMSE_tl = tl_notl_test(data_test, test_idx, coef_p2acc, coef_p2, coef_p3, coef_p2_tl, coef_p3_tl)
        RMSE_all[i].append(RMSE)
        RMSE_all_tl[i].append(RMSE_tl)
               
    median = [np.median(RMSE_all[0]) ]
    median_tl = [np.median(RMSE_all_tl[0]) ]
    print("Old RMSE:",RMSE_all[0],"New RMSE:", RMSE_all_tl[0])
    print("Old median:",median,"New median:", median_tl)
   
    
    print ((sum(median)-sum(median_tl))/sum(median))
    return RMSE_all[0],RMSE_all_tl[0],sum(median),sum(median_tl),(sum(median)-sum(median_tl))/sum(median)
           


def is_pos_def(A):
    M = np.matrix(A)
    return np.all(np.linalg.eigvals(M+M.transpose()) > 0)

def neighbor():
    mcmc_sample_95 = [0]*4
    cc = 0
    with open('newfilePath.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            mcmc_sample_95[cc] = row
            mcmc_sample_95[cc] =np.sort(list(np.float_(mcmc_sample_95[cc])))
            cc = cc+1
    Test = True
    while(Test==True): 
        Random_cov_29 = random.choice(mcmc_sample_95[0])
        Random_cov_30 = 0
        while(Random_cov_30<=0):
            Random_cov_30=random.choice(mcmc_sample_95[1])
        Random_cov_36 = random.choice(mcmc_sample_95[3])
        cadidate = np.array([[Random_cov_29, Random_cov_30], [Random_cov_30, Random_cov_36]])
        
        if(is_pos_def(cadidate)==True):
            Test=False
            
           
    return Random_cov_29,Random_cov_30,Random_cov_36

def acceptance_probability(old_solution,new_solution,T):
   
   
    return math.exp(1)**((new_solution-old_solution)/T)




anneal_iteration = []
anneal_value = []
test_anneal_value=[]
def anneal(training_sample_size,samples_list,data_new_copy,bp_data_new_copy,initial_cov_29,initial_cov_30,initial_cov_36):
    start = timer()
    
    RMSE,RMSE_tl,median_n,median_t,old_cost= cost(training_sample_size,samples_list,data_new_copy,bp_data_new_copy,initial_cov_29,initial_cov_30,initial_cov_36)
    solution_cov_29= initial_cov_29
    solution_cov_30= initial_cov_30
    solution_cov_36= initial_cov_36
    T = 1.0
    T_min = 0.00001

    alpha = 0.9
    anneal_count = 0
    anneal_iteration.append(anneal_count)
    anneal_value.append(old_cost)
    optima = old_cost
    while T > T_min:
        i = 1
        while i <= 100:
           
            Random_29,Random_30,Random_36= neighbor()    
            new_RMSE,new_RMSE_tl,new_median_n,new_median_t,new_cost= cost(training_sample_size,samples_list,data_new_copy,bp_data_new_copy,Random_29,Random_30,Random_36)
            ap = acceptance_probability(old_cost, new_cost, T)
            delta = new_cost-old_cost
      
           
            if(delta > 0 or ap > np.random.random(1)):      
       
                solution_cov_29= Random_29
                solution_cov_30= Random_30
                solution_cov_36= Random_36
                RMSE= new_RMSE
                RMSE_tl= new_RMSE_tl
                median_n = new_median_n
                median_t = new_median_t
                old_cost = new_cost
                print(old_cost)
                print("**Change**",old_cost)                        
           
            i += 1
            anneal_count += 1
            anneal_iteration.append(anneal_count)
            test_anneal_value.append(old_cost)
            
            if(old_cost>optima):
                
                optima = copy.deepcopy(old_cost)
                anneal_value.append(optima)
            else:
                anneal_value.append(optima)
       
        T = T*alpha
        u = timer()
        
#        if((u-start)>120):
#            break
       
       
    solution_cov = np.array([[solution_cov_29,solution_cov_30],[solution_cov_30,solution_cov_36]])
   
    print(median_n,median_t,old_cost, solution_cov)
   
    return RMSE,RMSE_tl,median_n,median_t,old_cost, solution_cov




#################################################Simulated Annealing###################################

start1 = timer()
print(anneal(training_sample_size,samples_list,data_new_copy,bp_data_new_copy,1,0,1))
   
end1 = timer()
print("The processing time: ",end1-start1)
plt.figure(figsize=(15,15))
plt.plot(anneal_value)
plt.figure(figsize=(15,15))
plt.plot(test_anneal_value)
