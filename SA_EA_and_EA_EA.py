
##=============================================================================
# Co-learning learning using assumed common covariance structure
##=============================================================================
import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.optimize import curve_fit
import matplotlib.patches as mpatches
import copy
import itertools
import math
import random
from timeit import default_timer as timer
from scipy.stats import invwishart
import HPD


start = timer()
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
## load new data. The file ended with ino120 is the Taz 6. xiao120 is Taz 5. 
##Based on the previous study, that we know Taz 6 does not have p1. However, for the coding purpose on the covariance estimation, we add the virtual p1 for Taz 6 ended with ino120_1
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
#with open('S800F9000_xiao120.csv') as csvfile:
#    readCSV = csv.reader(csvfile, delimiter=',')
#    for row in readCSV:
#        coords_1.append(float(row[0]))
#        width_1.append(float(row[1]))
#       
#coords_1 = np.asarray(coords_1)
#width_1 = np.asarray(width_1)
##noise1 = np.random.normal(0,0.1,len(width_1))
##width_1 = width_1+noise1
#width_1_ave = np.mean(width_1[(coords_1>start_p)&(coords_1<stop_p)])
#width_interp_1  =  np.interp(coords, coords_1 ,width_1)
##width_interp_1 = width_interp_1 - width_1_ave
#a = 800
#feed = 150
#l_vector, v_1, a_vector_1, t_vector_1 = acc_ana(a, v0, v1, feed, jerk, l, stp)
#data = np.transpose(np.array((coords, t_vector_1, v_1, a_vector_1, width_interp_1)))

# load the line measurement data 2
coords_2 = []
width_2 = []
with open('S800F6000_ino120_1.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        coords_2.append(float(row[0]))
        width_2.append(float(row[1]))
       
coords_2 = np.asarray(coords_2)
width_2 = np.asarray(width_2)
print(width_2)
noise2 = np.random.normal(0,0.001)
width_2 = width_2+noise2
print(width_2)
width_2_ave = np.mean(width_2[(coords_2>start_p)&(coords_2<stop_p)])
width_interp_2  =  np.interp(coords, coords_2 ,width_2)
#width_interp_2 = width_interp_2 - width_2_ave
a = 800
feed = 100
l_vector, v_2, a_vector_2, t_vector_2 = acc_ana(a, v0, v1, feed, jerk, l, stp)
data = np.transpose(np.array((coords, t_vector_2, v_2, a_vector_2, width_interp_2)))

# load the line measurement data 3
coords_3 = []
width_3 = []
with open('S800F3000_ino120_1.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        coords_3.append(float(row[0]))
        width_3.append(float(row[1]))
       
coords_3 = np.asarray(coords_3)
width_3 = np.asarray(width_3)
noise3 = np.random.normal(0,0.001)
width_3 = width_3+noise3
width_3_ave = np.mean(width_3[(coords_3>start_p)&(coords_3<stop_p)])
width_interp_3  =  np.interp(coords, coords_3 ,width_3)
#width_interp_3 = width_interp_3 - width_3_ave
a = 800
feed = 50
l_vector, v_3, a_vector_3, t_vector_3 = acc_ana(a, v0, v1, feed, jerk, l, stp)
data = [data, np.transpose(np.array((coords, t_vector_3, v_3, a_vector_3, width_interp_3)))]

# load the line measurement data 4
#coords_4 = []
#width_4 = []
#with open('S600F9000_xiao120.csv') as csvfile:
#    readCSV = csv.reader(csvfile, delimiter=',')
#    for row in readCSV:
#        coords_4.append(float(row[0]))
#        width_4.append(float(row[1]))
#       
#coords_4 = np.asarray(coords_4)
#width_4 = np.asarray(width_4)
##noise4 = np.random.uniform(0,0.000001,len(width_4))
##width_4 = width_4+noise4
#width_4_ave = np.mean(width_4[(coords_4>start_p)&(coords_4<stop_p)])
#width_interp_4  =  np.interp(coords, coords_4 ,width_4)
##width_interp_4 = width_interp_4 - width_4_ave
#a = 600
#feed = 150
#l_vector, v_4, a_vector_4, t_vector_4 = acc_ana(a, v0, v1, feed, jerk, l, stp)
#data.append(np.transpose(np.array((coords, t_vector_4, v_4, a_vector_4, width_interp_4))))

# load the line measurement data 5
coords_5 = []
width_5 = []
with open('S600F6000_ino120_1.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        coords_5.append(float(row[0]))
        width_5.append(float(row[1]))
       
coords_5 = np.asarray(coords_5)
width_5 = np.asarray(width_5)
noise5 = np.random.normal(0,0.001)
width_5 = width_5+noise5
width_5_ave = np.mean(width_5[(coords_5>start_p)&(coords_5<stop_p)])
width_interp_5  =  np.interp(coords, coords_5 ,width_5)
#width_interp_5 = width_interp_5 - width_5_ave
a = 600
feed = 100
l_vector, v_5, a_vector_5, t_vector_5 = acc_ana(a, v0, v1, feed, jerk, l, stp)
data.append(np.transpose(np.array((coords, t_vector_5, v_5, a_vector_5, width_interp_5))))

# load the line measurement data 6
#coords_6 = []
#width_6 = []
#with open('S600F3000_xiao120.csv') as csvfile:
#    readCSV = csv.reader(csvfile, delimiter=',')
#    for row in readCSV:
#        coords_6.append(float(row[0]))
#        width_6.append(float(row[1]))
#       
#coords_6 = np.asarray(coords_6)
#width_6 = np.asarray(width_6)
##noise6 = np.random.uniform(0,0.000001,len(width_6))
##width_6 = width_6+noise6
#width_6_ave = np.mean(width_6[(coords_6>start_p)&(coords_6<stop_p)])
#width_interp_6  =  np.interp(coords, coords_6 ,width_6)
##width_interp_6 = width_interp_6 - width_6_ave
#a = 600
#feed = 50
#l_vector, v_6, a_vector_6, t_vector_6 = acc_ana(a, v0, v1, feed, jerk, l, stp)
#data.append(np.transpose(np.array((coords, t_vector_6, v_6, a_vector_6, width_interp_6))))

# load the line measurement data 6
#coords_7 = []
#width_7 = []
#with open('S400F9000_xiao120.csv') as csvfile:
#    readCSV = csv.reader(csvfile, delimiter=',')
#    for row in readCSV:
#        coords_7.append(float(row[0]))
#        width_7.append(float(row[1]))
#       
#coords_7 = np.asarray(coords_7)
#width_7 = np.asarray(width_7)
#width_7 = width_7[9:2525]# ditch bad values
#coords_7 = np.linspace(0,120,len(width_7))###################################################################
#width_7_ave = np.mean(width_7[(coords_7>start_p)&(coords_7<stop_p)])
#width_interp_7  =  np.interp(coords, coords_7 ,width_7)
##width_interp_7 = width_interp_7 - width_7_ave
##a = a_nomi2real(400)
##feed = feed_nomi2real(150)
#l_vector, v_7, a_vector_7, t_vector_7 = acc_ana(a, v0, v1, feed, jerk, l, stp)
#data.append(np.transpose(np.array((coords, t_vector_7, v_7, a_vector_7, width_interp_7))))

# load the line measurement data 8
coords_8 = []
width_8 = []
with open('S400F6000_ino120_1.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        coords_8.append(float(row[0]))
        width_8.append(float(row[1]))
       
coords_8 = np.asarray(coords_8)
width_8 = np.asarray(width_8)
noise8 = np.random.normal(0,0.001)
width_8 = width_8+noise8
width_8_ave = np.mean(width_8[(coords_8>start_p)&(coords_8<stop_p)])
width_interp_8  =  np.interp(coords, coords_8 ,width_8)
#width_interp_8 = width_interp_8 - width_8_ave
a = 400
feed = 100
l_vector, v_8, a_vector_8, t_vector_8 = acc_ana(a, v0, v1, feed, jerk, l, stp)
data.append(np.transpose(np.array((coords, t_vector_8, v_8, a_vector_8, width_interp_8))))

# load the line measurement data 9
coords_9 = []
width_9 = []
with open('S800F9000_ino120_1.csv') as csvfile:
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
data.append(np.transpose(np.array((coords, t_vector_9, v_9, a_vector_9, width_interp_9))))

# load the line measurement data 10
coords_10 = []
width_10 = []
with open('S800F9000_xiao120.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        coords_10.append(float(row[0]))
        width_10.append(float(row[1]))
       
coords_10 = np.asarray(coords_10)
width_10 = np.asarray(width_10)

noise10 = np.random.normal(0,0.001)
width_10 = width_10+noise10
width_10_ave = np.mean(width_10[(coords_10>start_p)&(coords_10<stop_p)])
width_interp_10  =  np.interp(coords, coords_10 ,width_10)
#width_interp_10 = width_interp_10 - width_10_ave
a = 800
feed = 150
l_vector, v_10, a_vector_10, t_vector_10 = acc_ana(a, v0, v1, feed, jerk, l, stp)
data.append(np.transpose(np.array((coords, t_vector_10, v_10, a_vector_10, width_interp_10))))

# load the line measurement data 11
#coords_11 = []
#width_11 = []
#with open('S800F6000_ino120_1.csv') as csvfile:
#    readCSV = csv.reader(csvfile, delimiter=',')
#    for row in readCSV:
#        coords_11.append(float(row[0]))
#        width_11.append(float(row[1]))
#       
#coords_11 = np.asarray(coords_11)
#width_11 = np.asarray(width_11)
#noise11 = np.random.uniform(0,0.000001,len(width_11))
#width_11 = width_11+noise11
#width_11_ave = np.mean(width_11[(coords_11>start_p)&(coords_11<stop_p)])
#width_interp_11  =  np.interp(coords, coords_11 ,width_11)
######comment

#width_interp_11 = width_interp_11 - width_11_ave

######comment
#a = 800
#feed = 100
#l_vector, v_11, a_vector_11, t_vector_11 = acc_ana(a, v0, v1, feed, jerk, l, stp)
#data.append(np.transpose(np.array((coords, t_vector_11, v_11, a_vector_11, width_interp_11))))


# load the line measurement data 12
coords_12 = []
width_12 = []
with open('S600F9000_xiao120.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        coords_12.append(float(row[0]))
        width_12.append(float(row[1]))
       
coords_12 = np.asarray(coords_12)
width_12 = np.asarray(width_12)
noise12 = np.random.normal(0,0.001)
width_12 = width_12+noise12
width_12_ave = np.mean(width_12[(coords_12>start_p)&(coords_12<stop_p)])
width_interp_12  =  np.interp(coords, coords_12 ,width_12)
#width_interp_12 = width_interp_12 - width_12_ave
a = 600
feed = 150
l_vector, v_12, a_vector_12, t_vector_12 = acc_ana(a, v0, v1, feed, jerk, l, stp)
data.append(np.transpose(np.array((coords, t_vector_12, v_12, a_vector_12, width_interp_12))))

# load the line measurement data 13
#coords_13 = []
#width_13 = []
#with open('S600F6000_ino120_1.csv') as csvfile:
#    readCSV = csv.reader(csvfile, delimiter=',')
#    for row in readCSV:
#        coords_13.append(float(row[0]))
#        width_13.append(float(row[1]))
#       
#coords_13 = np.asarray(coords_13)
#width_13 = np.asarray(width_13)
##noise13 = np.random.uniform(0,0.000001,len(width_13))
##width_13 = width_13+noise13
#width_13_ave = np.mean(width_13[(coords_13>start_p)&(coords_13<stop_p)])
######comment
#width_interp_13  =  np.interp(coords, coords_13 ,width_13)
######comment
#width_interp_13 = width_interp_13 - width_13_ave
#a = 600
#feed = 100
#l_vector, v_13, a_vector_13, t_vector_13 = acc_ana(a, v0, v1, feed, jerk, l, stp)
#data.append(np.transpose(np.array((coords, t_vector_13, v_13, a_vector_13, width_interp_13))))

# load the line measurement data 14
coords_14 = []
width_14 = []
with open('S600F3000_xiao120.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        coords_14.append(float(row[0]))
        width_14.append(float(row[1]))
       
coords_14 = np.asarray(coords_14)
width_14 = np.asarray(width_14)
noise14 = np.random.normal(0,0.001)
width_14 = width_14+noise14
width_14_ave = np.mean(width_14[(coords_14>start_p)&(coords_14<stop_p)])
width_interp_14  =  np.interp(coords, coords_14 ,width_14)
#width_interp_13 = width_interp_13 - width_13_ave
a = 600
feed = 50
l_vector, v_14, a_vector_14, t_vector_14 = acc_ana(a, v0, v1, feed, jerk, l, stp)
data.append(np.transpose(np.array((coords, t_vector_14, v_14, a_vector_14, width_interp_14))))

# load the line measurement data 15
coords_15 = []
width_15 = []
with open('S400F9000_xiao120.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        coords_15.append(float(row[0]))
        width_15.append(float(row[1]))
       
coords_15 = np.asarray(coords_15)
width_15 = np.asarray(width_15)

width_15 = width_15[9:2525]# ditch bad values
coords_15 = np.linspace(0,120,len(width_15))
noise15 = np.random.normal(0,0.001)
width_15 = width_15+noise15
width_15_ave = np.mean(width_15[(coords_15>start_p)&(coords_15<stop_p)])
width_interp_15  =  np.interp(coords, coords_15 ,width_15)
#width_interp_13 = width_interp_13 - width_13_ave
a = 400
feed = 150
l_vector, v_15, a_vector_15, t_vector_15 = acc_ana(a, v0, v1, feed, jerk, l, stp)
data.append(np.transpose(np.array((coords, t_vector_15, v_15, a_vector_15, width_interp_15))))

# load the line measurement data 16
coords_16 = []
width_16 = []
with open('S400F3000_xiao120.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        coords_16.append(float(row[0]))
        width_16.append(float(row[1]))
       
coords_16 = np.asarray(coords_16)
width_16 = np.asarray(width_16)
noise16 = np.random.normal(0,0.001)
width_16 = width_16+noise16
width_16_ave = np.mean(width_16[(coords_16>start_p)&(coords_16<stop_p)])
width_interp_16  =  np.interp(coords, coords_16 ,width_16)
#width_interp_13 = width_interp_13 - width_13_ave
a = 400
feed = 50
l_vector, v_16, a_vector_16, t_vector_16 = acc_ana(a, v0, v1, feed, jerk, l, stp)
data.append(np.transpose(np.array((coords, t_vector_16, v_16, a_vector_16, width_interp_16))))


# load the line measurement data 17
coords_17 = []
width_17 = []
with open('HHH___101ts.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        coords_17.append(float(row[0]))
        width_17.append(float(row[1]))
       
coords_17 = np.asarray(coords_17)
width_17 = np.asarray(width_17)
noise17 = np.random.normal(0,0.001)
width_17 = width_17+noise17
width_17_ave = np.mean(width_17[(coords_17>start_p)&(coords_17<stop_p)])
width_interp_17  =  np.interp(coords, coords_17 ,width_17)
#width_interp_10 = width_interp_10 - width_10_ave
a = 800
feed = 150
l_vector, v_17, a_vector_17, t_vector_17 = acc_ana(a, v0, v1, feed, jerk, l, stp)
data.append(np.transpose(np.array((coords, t_vector_17, v_17, a_vector_17, width_interp_17))))

# load the line measurement data 18
coords_18 = []
width_18 = []
with open('MHH___101ts.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        coords_18.append(float(row[0]))
        width_18.append(float(row[1]))
       
coords_18 = np.asarray(coords_18)
width_18 = np.asarray(width_18)
noise18 = np.random.normal(0,0.001)
width_18 = width_18+noise18
width_18_ave = np.mean(width_18[(coords_18>start_p)&(coords_18<stop_p)])
width_interp_18  =  np.interp(coords, coords_18 ,width_18)
#width_interp_11 = width_interp_11 - width_11_ave
a = 600
feed = 150
l_vector, v_18, a_vector_18, t_vector_18 = acc_ana(a, v0, v1, feed, jerk, l, stp)
data.append(np.transpose(np.array((coords, t_vector_18, v_18, a_vector_18, width_interp_18))))


# load the line measurement data 19
coords_19 = []
width_19 = []
with open('MLH___101ts.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        coords_19.append(float(row[0]))
        width_19.append(float(row[1]))
       
coords_19 = np.asarray(coords_19)
width_19 = np.asarray(width_19)
noise19 = np.random.normal(0,0.001)
width_19 = width_19+noise19
width_19_ave = np.mean(width_19[(coords_19>start_p)&(coords_19<stop_p)])
width_interp_19  =  np.interp(coords, coords_19 ,width_19)
#width_interp_12 = width_interp_12 - width_12_ave
a = 600
feed = 50
l_vector, v_19, a_vector_19, t_vector_19 = acc_ana(a, v0, v1, feed, jerk, l, stp)
data.append(np.transpose(np.array((coords, t_vector_19, v_19, a_vector_19, width_interp_19))))

# load the line measurement data 20
coords_20 = []
width_20 = []
with open('LHH___101ts.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        coords_20.append(float(row[0]))
        width_20.append(float(row[1]))
       
coords_20 = np.asarray(coords_20)
width_20 = np.asarray(width_20)
noise20 = np.random.normal(0,0.001)
width_20 = width_20+noise20
width_20_ave = np.mean(width_20[(coords_20>start_p)&(coords_20<stop_p)])
width_interp_20  =  np.interp(coords, coords_20 ,width_20)
#width_interp_13 = width_interp_13 - width_13_ave
a = 400
feed = 150
l_vector, v_20, a_vector_20, t_vector_20 = acc_ana(a, v0, v1, feed, jerk, l, stp)
data.append(np.transpose(np.array((coords, t_vector_20, v_20, a_vector_20, width_interp_20))))

# load the line measurement data 21
coords_21 = []
width_21 = []
with open('LLH___101ts.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        coords_21.append(float(row[0]))
        width_21.append(float(row[1]))
       
coords_21 = np.asarray(coords_21)
width_21 = np.asarray(width_21)
noise21 = np.random.normal(0,0.001)
width_21 = width_21+noise21
width_21_ave = np.mean(width_21[(coords_21>start_p)&(coords_21<stop_p)])
width_interp_21  =  np.interp(coords, coords_21 ,width_21)
#width_interp_13 = width_interp_13 - width_13_ave
a = 400
feed = 50
l_vector, v_21, a_vector_21, t_vector_21 = acc_ana(a, v0, v1, feed, jerk, l, stp)
data.append(np.transpose(np.array((coords, t_vector_21, v_21, a_vector_21, width_interp_21))))

# load the line measurement data 22
coords_22 = []
width_22 = []
with open('HHH___101s.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        coords_22.append(float(row[0]))
        width_22.append(float(row[1]))
       
coords_22 = np.asarray(coords_22)
width_22 = np.asarray(width_22)
noise22 = np.random.normal(0,0.001)
width_22 = width_22+noise22
width_22_ave = np.mean(width_22[(coords_22>start_p)&(coords_22<stop_p)])
width_interp_22  =  np.interp(coords, coords_22 ,width_22)
#width_interp_10 = width_interp_10 - width_10_ave
a = 800
feed = 150
l_vector, v_22, a_vector_22, t_vector_22 = acc_ana(a, v0, v1, feed, jerk, l, stp)
data.append(np.transpose(np.array((coords, t_vector_22, v_22, a_vector_22, width_interp_22))))

# load the line measurement data 23
coords_23 = []
width_23 = []
with open('MHH2___101s.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        coords_23.append(float(row[0]))
        width_23.append(float(row[1]))
       
coords_23 = np.asarray(coords_23)
width_23 = np.asarray(width_23)
noise23 = np.random.normal(0,0.001)
width_23 = width_23+noise23
width_23_ave = np.mean(width_23[(coords_23>start_p)&(coords_23<stop_p)])
width_interp_23  =  np.interp(coords, coords_23 ,width_23)
#width_interp_11 = width_interp_11 - width_11_ave
a = 600
feed = 150
l_vector, v_23, a_vector_23, t_vector_23 = acc_ana(a, v0, v1, feed, jerk, l, stp)
data.append(np.transpose(np.array((coords, t_vector_23, v_23, a_vector_23, width_interp_23))))


# load the line measurement data 24
coords_24 = []
width_24 = []
with open('MLH___101s.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        coords_24.append(float(row[0]))
        width_24.append(float(row[1]))
       
coords_24 = np.asarray(coords_24)
width_24 = np.asarray(width_24)
noise24 = np.random.normal(0,0.001)
width_24 = width_24+noise24
width_24_ave = np.mean(width_24[(coords_24>start_p)&(coords_24<stop_p)])
width_interp_24  =  np.interp(coords, coords_24 ,width_24)
#width_interp_12 = width_interp_12 - width_12_ave
a = 600
feed = 50
l_vector, v_24, a_vector_24, t_vector_24 = acc_ana(a, v0, v1, feed, jerk, l, stp)
data.append(np.transpose(np.array((coords, t_vector_24, v_24, a_vector_24, width_interp_24))))

# load the line measurement data 25
coords_25 = []
width_25 = []
with open('LHH___101s.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        coords_25.append(float(row[0]))
        width_25.append(float(row[1]))
       
coords_25 = np.asarray(coords_25)
width_25 = np.asarray(width_25)
noise25 = np.random.normal(0,0.001)
width_25 = width_25+noise25
width_25_ave = np.mean(width_25[(coords_25>start_p)&(coords_25<stop_p)])
width_interp_25  =  np.interp(coords, coords_25 ,width_25)
#width_interp_13 = width_interp_13 - width_13_ave
a = 400
feed = 150
l_vector, v_25, a_vector_25, t_vector_25 = acc_ana(a, v0, v1, feed, jerk, l, stp)
data.append(np.transpose(np.array((coords, t_vector_25, v_25, a_vector_25, width_interp_25))))

# load the line measurement data 26
coords_26 = []
width_26 = []
with open('LLH___101s.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        coords_26.append(float(row[0]))
        width_26.append(float(row[1]))
       
coords_26 = np.asarray(coords_26)
width_26 = np.asarray(width_26)
noise26 = np.random.normal(0,0.001)
width_26 = width_26+noise26
width_26_ave = np.mean(width_26[(coords_26>start_p)&(coords_26<stop_p)])
width_interp_26  =  np.interp(coords, coords_26 ,width_26)
#width_interp_13 = width_interp_13 - width_13_ave
a = 400
feed = 50
l_vector, v_26, a_vector_26, t_vector_26 = acc_ana(a, v0, v1, feed, jerk, l, stp)
data.append(np.transpose(np.array((coords, t_vector_26, v_26, a_vector_26, width_interp_26))))


# deepcopy data
data_copy = copy.deepcopy(data)
##=============================================================================

# Make initial break points
def identify_breakpoints(line_coords, line_width):
    #p1
    # after breakpoint 1, p1, the width start increase
    width_difference = line_width[1:] - line_width[0:-1]
    idx1 = np.argwhere(width_difference > 0.)
    #p1_idx = line_coords[idx1[0]]
    p1_idx = idx1[0]
    # print(p1_idx)
    #p2
    # after breakpoint 2, p2, the width remains constant, the increasing trend ends
    # the breakpoint p2 is determined by selecting the point when the line width first
    # reaches the nominal line width (zero, if predict deviation)
    # width_average = np.mean(line_width[(line_coords>80) & (line_coords<90)])
    # width_average = 0.8
    # idx2 = np.argwhere(line_width>width_average)
    # idx2 = idx2[idx2>idx1[0]]
#    p2 = line_coords[idx2[0]]
    # p2_idx = idx2[0]
   
    #p3
    # breakpoint 3 is the point where the line width start increasing beyond the mean width
    # initial breakpoint 3 can be obtained by calculate the down/decreasing trend from the right end, the first point
    # that starts increase will be the initial breakpoint 3
    # width_max = np.max(line_width[(line_coords>40) & (line_coords<80)])
    # idx3 = np.argwhere(line_width>width_max)
    # idx3 = idx3[idx3>idx1[0]]
    # p3 = line_coords[idx3[0]]
    # line_width_reverse = line_width[::-1]
    # width_difference_reverse = line_width_reverse[1:] - line_width_reverse[0:-1]
    # idx3_reverse = np.argwhere(width_difference_reverse > 0.)
    # p3 = line_coords[-idx3_reverse[0]]
    # p3 = 100
   
    return p1_idx
   
# the training data set bp_data contains [p1, p2, p3, v, a, jerk, (v^2-jerk^2)/a, l]
#p1_idx = identify_breakpoints(coords, width_interp_1)
#v = 150.00
#a = 800.00
#jerk = 8.00
##x = (np.square(v) - np.square(jerk))/a
#x = v
#bp_data = np.array([coords[p1_idx][0], t_vector_1[p1_idx][0], v, a, jerk, x, np.max(coords)],ndmin=2)


p1_idx = identify_breakpoints(coords, width_interp_2)
v = 100.00
a = 800.00
jerk = 8.00
#x = (np.square(v) - np.square(jerk))/a
x = v
bp_data = np.array([coords[p1_idx][0], t_vector_2[p1_idx][0], v, a, jerk, x, np.max(coords)],ndmin=2)


p1_idx = identify_breakpoints(coords, width_interp_3)
v = 50.00
a = 800.00
jerk = 8.00
#x = (np.square(v) - np.square(jerk))/a
x = v
bp_data = np.concatenate((bp_data, np.array([coords[p1_idx][0], t_vector_3[p1_idx][0], v, a, jerk, x, np.max(coords)],ndmin=2)))


#p1_idx = identify_breakpoints(coords, width_interp_4)
#v = 150.00
#a = 600.00
#jerk = 8.00
##x = (np.square(v) - np.square(jerk))/a
#x = v
#bp_data = np.concatenate((bp_data, np.array([coords[p1_idx][0], t_vector_4[p1_idx][0], v, a, jerk, x, np.max(coords)],ndmin=2)))

p1_idx = identify_breakpoints(coords, width_interp_5)
v = 100.00
a = 600.00
jerk = 8.00
#x = (np.square(v) - np.square(jerk))/a
x = v
bp_data = np.concatenate((bp_data, np.array([coords[p1_idx][0], t_vector_5[p1_idx][0], v, a, jerk, x, np.max(coords)],ndmin=2)))

#p1_idx = identify_breakpoints(coords, width_interp_6)
#v = 50.00
#a = 600.00
#jerk = 8.00
##x = (np.square(v) - np.square(jerk))/a
#x = v
#bp_data = np.concatenate((bp_data, np.array([coords[p1_idx][0], t_vector_6[p1_idx][0], v, a, jerk, x, np.max(coords)],ndmin=2)))

#width_6 = width_6[9:2525]
#coords_6 = np.linspace(0,120,len(width_6))
#p1_idx = identify_breakpoints(coords, width_interp_7)
#v = 150.00
#a = 400.00
#jerk = 8.00
##x = (np.square(v) - np.square(jerk))/a
#x = v
#bp_data = np.concatenate((bp_data, np.array([coords[p1_idx][0], t_vector_7[p1_idx][0], v, a, jerk, x, np.max(coords)],ndmin=2)))

p1_idx = identify_breakpoints(coords, width_interp_8)
v = 100.00
a = 400.00
jerk = 8.00
#x = (np.square(v) - np.square(jerk))/a
x = v
bp_data = np.concatenate((bp_data, np.array([coords[p1_idx][0], t_vector_8[p1_idx][0], v, a, jerk, x, np.max(coords)],ndmin=2)))

p1_idx = identify_breakpoints(coords, width_interp_9)
v = 150.00
a = 800.00
jerk = 8.00
#x = (np.square(v) - np.square(jerk))/a
x = v
bp_data = np.concatenate((bp_data, np.array([coords[p1_idx][0], t_vector_9[p1_idx][0], v, a, jerk, x, np.max(coords)],ndmin=2)))

p1_idx = identify_breakpoints(coords, width_interp_10)
v = 150.00
a = 800.00
jerk = 8.00
#x = (np.square(v) - np.square(jerk))/a
x = v
bp_data = np.concatenate((bp_data, np.array([coords[p1_idx][0], t_vector_10[p1_idx][0], v, a, jerk, x, np.max(coords)],ndmin=2)))

#p1_idx = identify_breakpoints(coords, width_interp_11)
#v = 100.00
#a = 800.00
#jerk = 8.00
##x = (np.square(v) - np.square(jerk))/a
#x = v
#bp_data = np.concatenate((bp_data, np.array([coords[p1_idx][0], t_vector_11[p1_idx][0], v, a, jerk, x, np.max(coords)],ndmin=2)))

p1_idx = identify_breakpoints(coords, width_interp_12)
v = 150.00
a = 600.00
jerk = 8.00
#x = (np.square(v) - np.square(jerk))/a
x = v
bp_data = np.concatenate((bp_data, np.array([coords[p1_idx][0], t_vector_12[p1_idx][0], v, a, jerk, x, np.max(coords)],ndmin=2)))

#p1_idx = identify_breakpoints(coords, width_interp_13)
#v = 100.00
#a = 600.00
#jerk = 8.00
##x = (np.square(v) - np.square(jerk))/a
#x = v
#bp_data = np.concatenate((bp_data, np.array([coords[p1_idx][0], t_vector_13[p1_idx][0], v, a, jerk, x, np.max(coords)],ndmin=2)))

p1_idx = identify_breakpoints(coords, width_interp_14)
v = 50.00
a = 600.00
jerk = 8.00
#x = (np.square(v) - np.square(jerk))/a
x = v
bp_data = np.concatenate((bp_data, np.array([coords[p1_idx][0], t_vector_14[p1_idx][0], v, a, jerk, x, np.max(coords)],ndmin=2)))

p1_idx = identify_breakpoints(coords, width_interp_15)
v = 150.00
a = 400.00
jerk = 8.00
#x = (np.square(v) - np.square(jerk))/a
x = v
bp_data = np.concatenate((bp_data, np.array([coords[p1_idx][0], t_vector_15[p1_idx][0], v, a, jerk, x, np.max(coords)],ndmin=2)))

p1_idx = identify_breakpoints(coords, width_interp_16)
v = 50.00
a = 400.00
jerk = 8.00
#x = (np.square(v) - np.square(jerk))/a
x = v
bp_data = np.concatenate((bp_data, np.array([coords[p1_idx][0], t_vector_16[p1_idx][0], v, a, jerk, x, np.max(coords)],ndmin=2)))

p1_idx = identify_breakpoints(coords, width_interp_17)
v = 150.00
a = 800.00
jerk = 8.00
#x = (np.square(v) - np.square(jerk))/a
x = v
bp_data = np.concatenate((bp_data, np.array([coords[p1_idx][0], t_vector_17[p1_idx][0], v, a, jerk, x, np.max(coords)],ndmin=2)))


p1_idx = identify_breakpoints(coords, width_interp_18)
v = 150.00
a = 600.00
jerk = 8.00
#x = (np.square(v) - np.square(jerk))/a
x = v
bp_data = np.concatenate((bp_data, np.array([coords[p1_idx][0], t_vector_18[p1_idx][0], v, a, jerk, x, np.max(coords)],ndmin=2)))


p1_idx = identify_breakpoints(coords, width_interp_19)
v = 50.00
a = 600.00
jerk = 8.00
#x = (np.square(v) - np.square(jerk))/a
x = v
bp_data = np.concatenate((bp_data, np.array([coords[p1_idx][0], t_vector_19[p1_idx][0], v, a, jerk, x, np.max(coords)],ndmin=2)))


p1_idx = identify_breakpoints(coords, width_interp_20)
v = 150.00
a = 400.00
jerk = 8.00
#x = (np.square(v) - np.square(jerk))/a
x = v
bp_data = np.concatenate((bp_data, np.array([coords[p1_idx][0], t_vector_20[p1_idx][0], v, a, jerk, x, np.max(coords)],ndmin=2)))

p1_idx = identify_breakpoints(coords, width_interp_21)
v = 50.00
a = 400.00
jerk = 8.00
#x = (np.square(v) - np.square(jerk))/a
x = v
bp_data = np.concatenate((bp_data, np.array([coords[p1_idx][0], t_vector_21[p1_idx][0], v, a, jerk, x, np.max(coords)],ndmin=2)))

p1_idx = identify_breakpoints(coords, width_interp_22)
v = 150.00
a = 800.00
jerk = 8.00
#x = (np.square(v) - np.square(jerk))/a
x = v
bp_data = np.concatenate((bp_data, np.array([coords[p1_idx][0], t_vector_22[p1_idx][0], v, a, jerk, x, np.max(coords)],ndmin=2)))


p1_idx = identify_breakpoints(coords, width_interp_23)
v = 150.00
a = 600.00
jerk = 8.00
#x = (np.square(v) - np.square(jerk))/a
x = v
bp_data = np.concatenate((bp_data, np.array([coords[p1_idx][0], t_vector_23[p1_idx][0], v, a, jerk, x, np.max(coords)],ndmin=2)))


p1_idx = identify_breakpoints(coords, width_interp_24)
v = 50.00
a = 600.00
jerk = 8.00
#x = (np.square(v) - np.square(jerk))/a
x = v
bp_data = np.concatenate((bp_data, np.array([coords[p1_idx][0], t_vector_24[p1_idx][0], v, a, jerk, x, np.max(coords)],ndmin=2)))


p1_idx = identify_breakpoints(coords, width_interp_25)
v = 150.00
a = 400.00
jerk = 8.00
#x = (np.square(v) - np.square(jerk))/a
x = v
bp_data = np.concatenate((bp_data, np.array([coords[p1_idx][0], t_vector_25[p1_idx][0], v, a, jerk, x, np.max(coords)],ndmin=2)))

p1_idx = identify_breakpoints(coords, width_interp_26)
v = 50.00
a = 400.00
jerk = 8.00
#x = (np.square(v) - np.square(jerk))/a
x = v
bp_data = np.concatenate((bp_data, np.array([coords[p1_idx][0], t_vector_26[p1_idx][0], v, a, jerk, x, np.max(coords)],ndmin=2)))

#training_list = [0,1,2,3]
#bp_data = bp_data[training_list]

## figure plot
#plt.figure()
#plt.scatter(bp_data[:,3], bp_data[:,0])
#plt.title('bp1 a v.s. l')

plt.figure()
plt.scatter(bp_data[:,3], bp_data[:,1])
#plt.hold(True)
plt.plot(bp_data[::3,3],bp_data[::3,1])
plt.plot(bp_data[1::3,3],bp_data[1::3,1])
plt.plot(bp_data[2::3,3],bp_data[2::3,1])
plt.title('bp1 a v.s. t')


# bp_2 is the point that the line width reaches constant
##### identify the position/index of bp_2. The thres_list is the constant linewidth for p4.

##Target data 1: 0.58, data4: 0.76, data6: 0.90,data7: 0.8,data9: 0.85
#Each each run only one target data will be chosen without adding gaussian noise.
##When the printer condition selected as target printer, the treshold for the thres_list[4] will be changed.
thres_list =  [0.76, 0.8, 0.80, 0.80,0.58,0.8, 0.75, 0.82, 0.8, 0.82, 0.6083,0.5954,0.7564,0.7435,0.8026,0.5674,0.5547,0.5463,0.5421,0.5453]


def identify_breakpoint2(width, thrs):
    idx = np.argwhere(width>thrs)
    idx = idx[idx>100] # 100 is to make sure the idx is not near the starting point
    return idx[0]
   
for i in range(len(data)):
    t_vector_tmp = data[i][:,1]
    idx_p2 = identify_breakpoint2(data[i][:,-1], thres_list[i])
    bp_data[i,5] = idx_p2
#    plt.figure()
#    plt.plot(data[i][:,1], data[i][:,4])
#    plt.hold(True)
#    plt.axvline(x = t_vector_tmp[int(idx_p2)])

# there might be breakpoints between bp1 and bp2 which represent the transition from
# acceleration to constant phase
for i in range(len(data)):
    tmp = data[i][:,3]
    idx_pac = np.argwhere(tmp==0)
    idx_pac = idx_pac[0]
    bp_data[i,6] = idx_pac

bp_data_copy = copy.deepcopy(bp_data)

##===========================================================================
##==================fit a function for bp1 prediction========================
def bp1_func(av, par1, par2):
    return par1*np.power(av[:,0], par2)

def bp1_fit(traindata_bp1):
    a_v = traindata_bp1[:,0:2]
    bp1 = traindata_bp1[:,2]
    popt, pcov = curve_fit(bp1_func, a_v, bp1)
    bp1_pred = bp1_func(a_v, *popt)
    return popt, bp1_pred
traindata_bp1 = bp_data[:,[3,2,1]]
coef_bp1, bp1_pred = bp1_fit(traindata_bp1)
plt.figure()
plt.scatter(bp_data[:,3], bp_data[:,1])
#plt.hold(True)
plt.plot(traindata_bp1[:,0], bp1_pred)

test_bp1 = np.arange(200,810,10).reshape(61,1)
bp1_pred2 = bp1_func(test_bp1, *coef_bp1)
plt.figure()
plt.scatter(bp_data[:,3], bp_data[:,1])
#plt.hold(True)
plt.plot(test_bp1, bp1_pred2, 'k')


##===========================================================================
##==================fit a function before bp1================================
# tvav
def seg1_func(t_v_a, alpha1, alpha2, alpha3):
    #return alpha1/(alpha2*t_v_a[:,0]*t_v_a[:,1] + alpha3)
    return alpha1/(alpha2*t_v_a[:,1] + alpha3)

def seg1_fit(traindata_seg1):
    t_v = traindata_seg1[:,0:2]
    width = traindata_seg1[:,-1]
    popt, pcov = curve_fit(seg1_func, t_v, width)
    width_pred = seg1_func(t_v, *popt)
    return popt, width_pred
def generate_traindata(data, bp_data):
    traindata_p1 = np.empty((0,5))
    for i in range(len(data)):
        t_tmp = data[i][:,1]
        t_threshold = bp_data[i,1]
        data_p1 = data[i]
        data_p1 = data_p1[t_tmp<=t_threshold,:]
        traindata_p1 = np.concatenate((traindata_p1, data_p1))
    return traindata_p1



hpd_mcmc_cov = []

def S_for_sigma(x, y):
    return np.matmul((x-y),(x-y))

Credible_95 = [0]*4
MCMC_autocorrelation = [0]*4

## Checking the matrix is positive definite or not.
from scipy.stats import multivariate_normal
def is_pos_def(A):
    M = np.matrix(A)
    return np.all(np.linalg.eigvals(M+M.transpose()) > 0)
    
##=============================================================================
## MCMC for posterior commmon covariance.
##=============================================================================

def MCMC_Covariance_Estimation(parameters,parameter_x1, parameter_x2, Time, Burn_in):
    
    idt = 0
    count = 0
    count1=0
    row1=0
    row = 0
    hpd_count = 0


    x = np.linspace(0.01, 1, 100)
    mu_0 = np.matrix([0,0])
    sigma_0 = np.eye(2,2)*0.1
    idt =idt + 1
    k_0 = 1
    df_0 = 6

    n = len(parameters[:,parameter_x1])
    k = k_0 + n
    df =  df_0 + n

    parameters_combined= np.matrix([parameters[:, parameter_x1],parameters[:, parameter_x2]]).T
    x_bar=parameters_combined.mean(axis=0)

   
    S_sigma = np.zeros((2,2))
   
    for i in parameters_combined:

        S_sigma= S_sigma + np.dot((i-x_bar),(i-x_bar).T)


    S1 = ((k_0*n)/(k_0+n))*np.dot((x_bar-mu_0),(x_bar-mu_0).T)
    
    sigma_n = np.linalg.inv(sigma_0) + S_sigma + S1
    
    print(sigma_n)
    

    np.random.seed(1024)
    
    for j in range(0,2):
        for k in range(0,2):
         
            theta = [0.0] * (Time+1)
            theta_t = [[[0.0,0.0],[0.0,0.0]]] * (Time+1)
            test = False
            while (test==False):
                theta_t[0] = invwishart.rvs(df = df, scale=np.linalg.inv(sigma_n))
                theta[0] = theta_t[0][j][k]
                if (is_pos_def(theta_t[0])==True):
                   test=True 
    
            t = 0
            while t < Time:
                t = t + 1
                test = False
                theta_stat_temp=[]
                while (test==False):

                     temp = multivariate_normal.rvs(mean=[theta_t[t-1][0][0],theta_t[t-1][0][1],theta_t[t-1][1][1]], cov=np.eye(3,3)*1)

                     theta_stat_temp=[[temp[0],temp[1]],[temp[1],temp[2]]]

                     if (is_pos_def(theta_stat_temp)==True):
                         test=True 
                
                q1 = multivariate_normal.pdf(x=[theta_t[t-1][0][0],theta_t[t-1][0][1],theta_t[t-1][1][1]],mean=[theta_stat_temp[0][0],theta_stat_temp[0][1],theta_stat_temp[1][1]], cov=np.eye(3,3)*1)
                q2 = multivariate_normal.pdf(x=[theta_stat_temp[0][0],theta_stat_temp[0][1],theta_stat_temp[1][1]],mean=[theta_t[t-1][0][0],theta_t[t-1][0][1],theta_t[t-1][1][1]], cov=np.eye(3,3)*1)         
                alpha = min(1, ((invwishart.pdf(theta_stat_temp, df = df, scale=np.linalg.inv(sigma_n))*q1) / (invwishart.pdf(theta_t[t-1],df = df, scale=np.linalg.inv(sigma_n))*q2)) )
               
                u = np.random.uniform(0, 1)
                
                if u < alpha:
                    theta[t] = theta_stat_temp[j][k]

                    theta_t[t] = theta_stat_temp
                else:
                    theta[t] = theta[t - 1]
                    theta_t[t] = theta_t[t-1] 
        
            theta_burn_in = [0]*(Time- Burn_in+1)

            for i in range(len(theta)):
                if i >=Burn_in:
                    theta_burn_in[i-Burn_in] = copy.deepcopy(theta[i])
                   

            
            hpd_mcmc_cov.append([HPD.hpd_grid(theta_burn_in, Time- Burn_in+1 , alpha=0.05, roundto=6)])

            theta_burn_in_95 =[]
           
            MCMC_autocorrelation[hpd_count] = theta_burn_in
         
            for times in theta_burn_in:
                if((times>=hpd_mcmc_cov[hpd_count][0][0][0][0]) and (times<=hpd_mcmc_cov[hpd_count][0][0][0][1]) ):
                    theta_burn_in_95.append(times)
                     
            hpd_count = hpd_count+1                
           
            Credible_95[count1] =copy.deepcopy(theta_burn_in_95)
            count1 = count1+1

            if (count == 2):
                count = -2
                row = row +1
           
            count = count + 2
       
            print(k,df,sigma_0,sigma_n,x_bar, end = ',')
            
    
    with open("MCMC_diagonstics66.csv", "a", newline='') as f:
        writer = csv.writer(f)
        for row in MCMC_autocorrelation:
            writer.writerow(row)
                   
    return hpd_mcmc_cov, theta, theta_burn_in

###@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
##===================================================
## load the oringal target (Taz 6) data without p1 for the RMSE of the prediction vs. real printing.
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
##==================covariance based CO-learning learning function================
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

## The start point for the p2 from ino120 (Taz 6).
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
    ##co-learning with covariance for coef_p2 and coef_p3

    coef_p2_p3 = copy.deepcopy(tl_cov_learning(traindata_p2_, traindata_p3_, cov, 0.15))
    coef_p2_tl =copy.deepcopy( [coef_p2_p3[0]])
    coef_p3_tl =copy.deepcopy( [coef_p2_p3[1]])
    return coef_p2acc, coef_p2, coef_p3, coef_p2_tl, coef_p3_tl

##==================model testing function for co/no co-learning============
def tl_notl_test(data_test, test_idx, coef_p2acc, coef_p2, coef_p3, coef_p2_tl, coef_p3_tl):
    ltva_va =copy.deepcopy( data_test[:,[0,1,2,3]])

    top_speed = np.max(data_test[:,2]) + np.zeros((len(ltva_va),1))
    acc = np.max(data_test[:,3]) + np.zeros((len(ltva_va),1))
    ltva_va = np.concatenate((ltva_va, top_speed, acc), axis = 1)

    # without co-learning
    ##############
    pred = forward_pred(ltva_va, start_list[test_idx], thres_list[test_idx], coef_p2acc, coef_p2, coef_p3)
    RMSE = np.sqrt(np.sum((pred - data_test[:,-1])**2)/len(pred))

    # with co-learning
    ##############

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

##=============================================================================
##============================Inner optimization:SA: Outer optimizaiton: EA==============================
##=============================================================================
########################################The cost for SA#######################################
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
###############################Simulated Annealing###################################

def anneal(training_sample_size,samples_list,data_new_copy,bp_data_new_copy,initial_cov_29,initial_cov_30,initial_cov_36):
    start = timer()
    
    RMSE,RMSE_tl,median_n,median_t,old_cost= cost(training_sample_size,samples_list,data_new_copy,bp_data_new_copy,initial_cov_29,initial_cov_30,initial_cov_36)
    solution_cov_29= initial_cov_29
    solution_cov_30= initial_cov_30
    solution_cov_36= initial_cov_36
    T = 1.0

    T_min = 0.00001

    alpha = 0.5
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
           
            i += 1
            anneal_count += 1
            anneal_iteration.append(anneal_count)
            
            if(old_cost>optima):
                
                optima = copy.deepcopy(old_cost)
                anneal_value.append(optima)
            else:
                anneal_value.append(optima)
       
        T = T*alpha
        u = timer()
        
        if((u-start)>120):
            break
       
       
    solution_cov = np.array([[solution_cov_29,solution_cov_30],[solution_cov_30,solution_cov_36]])

    
    
    print(median_n,median_t,old_cost, solution_cov)
    u = timer()
    print(u-start)
   
    return RMSE,RMSE_tl,median_n,median_t,old_cost, solution_cov


###############################Evolutionary Algorithm: Genetic Algorithm###################################

from geneticalgorithm import geneticalgorithm as ga
from geneticalgorithm_inner import geneticalgorithm_inner as ga_inner

plt.figure(figsize=(10,10))

algorithm_param3 = {'max_num_iteration': 12,\
                   'population_size':5,\
                   'mutation_probability':0.01,\
                   'elit_ratio': 0.2,\
                   'crossover_probability': 0.8,\
                   'parents_portion': 0.6,\
                   'crossover_type':'uniform',\
                   'max_iteration_without_improv':None}

with open("sga.csv", "w", newline='') as ff:
    sga_writer = csv.writer(ff)
    sga_writer.writerow([])
    


def cost3(X):
    

    X = X.astype(int)
    print(X,'-------')
    X[4] = 1
    print(X,'-------')
    sample_training_list = np.where(X==1)[0]
    print(sample_training_list)
    sample_training_list = sample_training_list.tolist()
    
    #sample_M = [i for i in range(4)]  # a list of experimental data
    sample_M = [i for i in sample_training_list ]  # a list of experimental data
    print(sample_M,'-------')
    sample_C = [i+1 for i in range(1)] # a list of the number of selected samples for parameter estimations
    print(sample_C,'-------')
    parameters = [[]]
    
    time = 0;
    for c in sample_C:
        all_combos = list(itertools.combinations(sample_M, c))
        for train_tup in all_combos:
            print(train_tup)
            training_list = list(train_tup)
            bp_data = bp_data_copy[training_list]
            data = [data_copy[i] for i in training_list]
            traindata_p1 = generate_traindata(data, bp_data)
            if traindata_p1.size:
                traindata_p1_ = traindata_p1[:,[1,2,4]]
                coef_p1, width_pred_p1 = seg1_fit(traindata_p1_)
            else:
                coef_p1 = np.array([0., 0., 0.])
            traindata_p2acc = generate_traindata2acc(data, bp_data)

            if traindata_p2acc.size:
                traindata_p2acc_ = traindata_p2acc[:,[1,2,3,4,5]]
                coef_p2acc, width_pred_p2acc = seg2acc_fit(traindata_p2acc_)
            else:
                coef_p2acc = np.array([0.])
            traindata_p2 = generate_traindata2(data, bp_data)
            if traindata_p2.size:
                traindata_p2_ = traindata_p2[:,[1,2,3,4,5]]
                coef_p2, width_pred_p2 = seg2_fit(traindata_p2_)
            else:
                coef_p2 = np.array([0.])
            traindata_p3 = generate_traindata3(data)
            if traindata_p3.size:
                traindata_p3_ = traindata_p3[:,[1,2,3,4,5]]
                coef_p3, width_pred_p3 = seg3_fit(traindata_p3_)
            else:
                coef_p3 = np.array([0.])
           
            parameters.append([coef_p1[0], coef_p1[1], coef_p1[2], coef_p2acc[0], coef_p2[0], coef_p3[0]])
            time = time +1;
           
    parameters.pop(0)
    parameters = np.array(parameters)
    

    MCMC_Covariance_Estimation(parameters,4,5,12000,2000)

    
    with open("newfilePath.csv", "w", newline='') as f:
        writer = csv.writer(f)
        for row in Credible_95:
            writer.writerow(row)
            
    RMSE,RMSE_tl,median_n,median_t,old_cost, solution_cov = anneal(training_sample_size,samples_list,data_new_copy,bp_data_new_copy,1,0,1)
    
    with open("sga.csv", "a", newline='') as ff:
        sga_writer = csv.writer(ff)
        sga_writer.writerow([old_cost,solution_cov[0][0],solution_cov[0][1],solution_cov[1][1]])
    
    return -old_cost
    

varbound  = ([[0,1]]*20)
model3=ga(function=cost3,dimension=20,variable_type='int',variable_boundaries=np.array(varbound),algorithm_parameters=algorithm_param3)

start = timer()
obj_var_sga,obj_func_sga = model3.run()

end = timer()
print("The processing time: ",end-start)


##=============================================================================
##============================Inner optimization:EA: Outer optimizaiton: EA==============================
##=============================================================================

def find_index_bound():    
    mcmc_sample_95 = [0]*4
    cc = 0
    with open('newfilePath.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            mcmc_sample_95[cc] = row
            mcmc_sample_95[cc] =np.sort(list(np.float_(mcmc_sample_95[cc])))
            cc = cc+1
            
    Random_cov_29 = [0, (len(mcmc_sample_95[0])-1)]
    Random_cov_30 = [0, (len(mcmc_sample_95[1])-1)]
    Random_cov_36 = [0, (len(mcmc_sample_95[3])-1)]
    
    return [Random_cov_29,Random_cov_30,Random_cov_36]

    
def cost4(X):
       
    mcmc_sample_95 = [0]*4
    cc = 0
    with open('newfilePath.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            mcmc_sample_95[cc] = row
            mcmc_sample_95[cc] =np.sort(list(np.float_(mcmc_sample_95[cc])))
            cc = cc+1
    i = 0
    RMSE_all = copy.deepcopy(RMSE_all_test)
    RMSE_all_tl =  copy.deepcopy(RMSE_all_tl_test)
    training_list = [0]
    test_list = [3,5,6,8]
    coef_p2acc, coef_p2, coef_p3, coef_p2_tl, coef_p3_tl = tl_notl_learning(training_list, data_new_copy, bp_data_new_copy,np.array([[mcmc_sample_95[0][int(X[0])],mcmc_sample_95[1][int(X[1])]],[mcmc_sample_95[1][int(X[1])],mcmc_sample_95[3][int(X[2])]]]) )

    for test_idx in test_list:
        data_test = copy.deepcopy(data_new_copy[test_idx])
        RMSE, RMSE_tl = tl_notl_test(data_test, test_idx, coef_p2acc, coef_p2, coef_p3, coef_p2_tl, coef_p3_tl)
        RMSE_all[i].append(RMSE)
        RMSE_all_tl[i].append(RMSE_tl)
                
    median = [np.median(RMSE_all[0]) ]
    median_tl = [np.median(RMSE_all_tl[0]) ]
    print("Old RMSE:",RMSE_all[0],"New RMSE:", RMSE_all_tl[0])
    print("Old median:",median,"New median:", median_tl)
    print("Current covariance: ",np.array([[mcmc_sample_95[0][int(X[0])],mcmc_sample_95[1][int(X[1])]],[mcmc_sample_95[1][int(X[1])],mcmc_sample_95[3][int(X[2])]]]) )
    print ((sum(median)-sum(median_tl))/sum(median))

    return -(sum(median)-sum(median_tl))/sum(median)




algorithm_param4 = {'max_num_iteration': 10000,\
                   'population_size':5,\
                   'mutation_probability':0.01,\
                   'elit_ratio': 0.2,\
                   'crossover_probability': 0.8,\
                   'parents_portion': 0.6,\
                   'crossover_type':'uniform',\
                   'max_iteration_without_improv':None}

algorithm_param5 = {'max_num_iteration': 12,\
                   'population_size':5,\
                   'mutation_probability':0.01,\
                   'elit_ratio': 0.2,\
                   'crossover_probability': 0.8,\
                   'parents_portion': 0.6,\
                   'crossover_type':'uniform',\
                   'max_iteration_without_improv':None}

def cost5(X):
    
    X = X.astype(int)
    print(X,'-------')
    X[4] = 1
    print(X,'-------')
    sample_training_list = np.where(X==1)[0]
    print(sample_training_list)
    sample_training_list = sample_training_list.tolist()
    #sample_M = [i for i in range(4)]  # a list of experimental data
    sample_M = [i for i in sample_training_list ]  # a list of experimental data
    print(sample_M,'-------')
    sample_C = [i+1 for i in range(1)] # a list of the number of selected samples for parameter estimations
    print(sample_C,'-------')
    parameters = [[]]
    
    time = 0;
    for c in sample_C:
        all_combos = list(itertools.combinations(sample_M, c))
        for train_tup in all_combos:
            print(train_tup)
            training_list = list(train_tup)
            bp_data = bp_data_copy[training_list]
            data = [data_copy[i] for i in training_list]
            traindata_p1 = generate_traindata(data, bp_data)
            if traindata_p1.size:
                traindata_p1_ = traindata_p1[:,[1,2,4]]
                coef_p1, width_pred_p1 = seg1_fit(traindata_p1_)
            else:
                coef_p1 = np.array([0., 0., 0.])
            traindata_p2acc = generate_traindata2acc(data, bp_data)

            if traindata_p2acc.size:
                traindata_p2acc_ = traindata_p2acc[:,[1,2,3,4,5]]
                coef_p2acc, width_pred_p2acc = seg2acc_fit(traindata_p2acc_)
            else:
                coef_p2acc = np.array([0.])
            traindata_p2 = generate_traindata2(data, bp_data)
            if traindata_p2.size:
                traindata_p2_ = traindata_p2[:,[1,2,3,4,5]]
                coef_p2, width_pred_p2 = seg2_fit(traindata_p2_)
            else:
                coef_p2 = np.array([0.])
            traindata_p3 = generate_traindata3(data)
            if traindata_p3.size:
                traindata_p3_ = traindata_p3[:,[1,2,3,4,5]]
                coef_p3, width_pred_p3 = seg3_fit(traindata_p3_)
            else:
                coef_p3 = np.array([0.])
           
            parameters.append([coef_p1[0], coef_p1[1], coef_p1[2], coef_p2acc[0], coef_p2[0], coef_p3[0]])
            time = time +1;
           
    parameters.pop(0)
    parameters = np.array(parameters)
    

    MCMC_Covariance_Estimation(parameters,4,5,12000,2000)
  
    
    with open("newfilePath.csv", "w", newline='') as f:
        writer = csv.writer(f)
        for row in Credible_95:
            writer.writerow(row)            
    
    model4=ga_inner(function=cost4,dimension=3,variable_type='int',variable_boundaries=np.array(find_index_bound()),algorithm_parameters=algorithm_param4)
    obj_var_ga,obj_func_ga = model4.run()
        
    return obj_func_ga

model5=ga(function=cost5,dimension=20,variable_type='int',variable_boundaries=np.array(varbound),algorithm_parameters=algorithm_param5)


start1 = timer()
#obj_var_gga,obj_func_gga=model5.run()
plt.show()
end1 = timer()
print("The processing time: ",end1-start1)

###################################################End#########################################################


##=============================================================================
##============================The comparison between SA + EA:GA and EA:GA + EA:GA==============================
##=============================================================================



print("-----------------------SA+EA----------------------------")


RMSE_1=[0.06766900183825711, 0.15950909644314834, 0.0785527041534568, 0.12106314409862311,0.038173705600765176, 0.05037796416979792, 0.05514013388902525, 0.045655641229201044,0.1298606481231939, 0.1432828456242262, 0.1637187358379245, 0.04054072627428842,0.02734631447077954, 0.05225109510152264, 0.040897279035297496, 0.04316812589820332,0.07477662399540912, 0.08833747763997478, 0.07770026787265054, 0.10171205004605288]
new_RMSE_1 = [0.05531125215858173, 0.03914072039833416, 0.02807305379862189, 0.045899927980249396,0.03666356666127844, 0.04575296357333964, 0.05010396783076562, 0.04763732480913945,0.12166783497230513, 0.13556763789675613, 0.15585891151261444, 0.04182296395988157,0.02833970826430928, 0.04524551337227723, 0.03767823589833532, 0.04345111327000191,0.056426694856788584, 0.07287068916943351, 0.05854158550493675, 0.08411610713941436]


RMSE_median_1 = np.median(RMSE_1)

new_RMSE_median_1 = np.median(new_RMSE_1)

print("RMSE_median with Single learning",RMSE_median_1)
print("25th percentile:", np.percentile(RMSE_1,25))
print("50th percentile:", np.percentile(RMSE_1,50))
print("75th percentile:", np.percentile(RMSE_1,75))

print("RMSE_median with Co-learning proposed:",new_RMSE_median_1)
print("25th percentile:", np.percentile(new_RMSE_1,25))
print("50th percentile:", np.percentile(new_RMSE_1,50))
print("75th percentile:", np.percentile(new_RMSE_1,75))

print("RMSE reduction:",(RMSE_median_1-new_RMSE_median_1)/RMSE_median_1)

print("---------------------EA+EA------------------------------")


RMSE_2=[0.06766900183825711, 0.15950909644314834, 0.0785527041534568, 0.12106314409862311,0.038173705600765176, 0.05037796416979792, 0.05514013388902525, 0.045655641229201044,0.1298606481231939, 0.1432828456242262, 0.1637187358379245, 0.04054072627428842,0.02734631447077954, 0.05225109510152264, 0.040897279035297496, 0.04316812589820332,0.07477662399540912, 0.08833747763997478, 0.07770026787265054, 0.10171205004605288]

new_RMSE_2=[0.05449221193753829, 0.12352499270519117, 0.05972011074832581, 0.08799260390277781,0.0369690254544574, 0.047054426097925504, 0.05133099784864131, 0.047101482553811976,0.12977289551683158, 0.1414518004856526, 0.16363913519717735, 0.04055229621615859,0.02796479703926529, 0.04566814072187469, 0.03803346988553545, 0.04338726113026632,0.07439928968236312, 0.08161934845042511, 0.07662744238207891, 0.10140212098840752]

RMSE_median_2 = np.median(RMSE_2)

new_RMSE_median_2 = np.median(new_RMSE_2)

print("RMSE_median with Single learning",RMSE_median_2)
print("25th percentile:", np.percentile(RMSE_2,25))
print("50th percentile:", np.percentile(RMSE_2,50))
print("75th percentile:", np.percentile(RMSE_2,75))

print("RMSE_median with Co-learning proposed:",new_RMSE_median_2)
print("25th percentile:", np.percentile(new_RMSE_2,25))
print("50th percentile:", np.percentile(new_RMSE_2,50))
print("75th percentile:", np.percentile(new_RMSE_2,75))

print("RMSE reduction:",(RMSE_median_2-new_RMSE_median_2)/RMSE_median_2)

print("---------------------Gradient_2------------------------------")

RMSE_3=[0.06363343208569315, 0.05404139295676339, 0.08553179892843973, 0.040337518102042745,0.22908234152686574, 0.1553544571553176, 0.13361172234857893, 0.13284890087695303,0.05699624321393498, 0.05015709852807144, 0.08483624031352113, 0.04318186494949176,0.168811075982358, 0.12386228214732156, 0.11004119777456031, 0.07825755681677202,0.06086110434225256, 0.05537770930825102, 0.07716100457049707, 0.05058976980813691]

new_RMSE_3=[0.05337143940953864, 0.1340845846534414, 0.06522052050826091, 0.09752391662633961,0.03941769295241061, 0.05044350317510827, 0.0579649000838271, 0.04469779131191841,0.1347487807053703, 0.1423934555040816, 0.170378504025994, 0.043209520949566545,0.03444357541574262, 0.04442196427530965, 0.04009567832384506, 0.04477676119299994,0.08094285225961961, 0.08237741276316633, 0.07713300614064073, 0.10891907669633089]

RMSE_median_3 = np.median(RMSE_3)

new_RMSE_median_3 = np.median(new_RMSE_3)

print("RMSE_median with Single learning",RMSE_median_3)
print("25th percentile:", np.percentile(RMSE_3,25))
print("50th percentile:", np.percentile(RMSE_3,50))
print("75th percentile:", np.percentile(RMSE_3,75))

print("RMSE_median with Co-learning proposed:",new_RMSE_median_3)
print("25th percentile:", np.percentile(new_RMSE_3,25))
print("50th percentile:", np.percentile(new_RMSE_3,50))
print("75th percentile:", np.percentile(new_RMSE_3,75))

print("RMSE reduction:",(RMSE_median_3-new_RMSE_median_3)/RMSE_median_3)

print("---------------------Gradient_1------------------------------")

RMSE_4=[0.06363343208569315, 0.05404139295676339, 0.08553179892843973, 0.040337518102042745,0.22908234152686574, 0.1553544571553176, 0.13361172234857893, 0.13284890087695303,0.05699624321393498, 0.05015709852807144, 0.08483624031352113, 0.04318186494949176,0.168811075982358, 0.12386228214732156, 0.11004119777456031, 0.07825755681677202,0.06086110434225256, 0.05537770930825102, 0.07716100457049707, 0.05058976980813691]

new_RMSE_4=[0.047801475366932425, 0.07366553313699165, 0.036394099528169535, 0.0468458680112042,0.03750224247357479, 0.04387221610324751, 0.04156087955914309, 0.0457814521716333,0.12446392465515922, 0.13641556753189524, 0.15865401013877103, 0.04134470945195952,0.028151940182194777, 0.047191664909535955, 0.037908486441159545, 0.04324401296975467,0.05237830474135469, 0.06142349884848853, 0.053365306759189664, 0.0783179306214635]

RMSE_median_4 = np.median(RMSE_4)

new_RMSE_median_4 = np.median(new_RMSE_4)

print("RMSE_median with Single learning",RMSE_median_4)
print("25th percentile:", np.percentile(RMSE_4,25))
print("50th percentile:", np.percentile(RMSE_4,50))
print("75th percentile:", np.percentile(RMSE_4,75))

print("RMSE_median with Co-learning proposed:",new_RMSE_median_4)
print("25th percentile:", np.percentile(new_RMSE_4,25))
print("50th percentile:", np.percentile(new_RMSE_4,50))
print("75th percentile:", np.percentile(new_RMSE_4,75))

print("RMSE reduction:",(RMSE_median_4-new_RMSE_median_4)/RMSE_median_4)


import matplotlib.pyplot as plt

RMSE_all = [RMSE_3,RMSE_2]
RMSE_all_tl = [RMSE_1,new_RMSE_2,new_RMSE_1]

data_a = RMSE_all
data_b = RMSE_all_tl

ticks = ['Single printer\nlearning','EA','Hybrid\n(SA+EA)']

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

plt.figure(figsize=(15,10))

boxprops = dict(linestyle='-', linewidth=5)
medianprops = dict(linestyle='-', linewidth=5)
bpr = plt.boxplot(data_b, positions=np.array(range(len(data_b)))*3.0, sym='', medianprops=medianprops,boxprops=boxprops,widths=1,whis = (25,75))

set_box_color(bpr, '#1f77b4')


plt.ylabel("RMSE (mm)",fontsize=30)
plt.xticks(range(0, len(ticks) * 3, 3), ticks)
plt.xlim(-3, len(ticks)*3)
plt.ylim(0.02, 0.13)
plt.yticks(fontsize = 30)
plt.xticks(fontsize = 30)

plt.tight_layout()

