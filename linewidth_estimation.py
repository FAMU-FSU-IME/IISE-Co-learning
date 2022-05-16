# import packages
import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.optimize import curve_fit
import copy
##=============================================================================
## Acceleration analysis
##=============================================================================
# A function given acceleration, jerk, feed, length_vector,
# return the time_vector v.s. speed_vector
# if feed>v0, then a > 0
# if feed<v0, then a < 0
r1 = np.linspace(0,120,2401)
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
### data loading #####
##===================================================
# parameter initialization and specification
start_p = 50 # for calculate the mean
stop_p = 80 # for calculate the mean
coords_1 = []
width_1 = []
l = 120#########################################################################################################
stp = 0.05
coords = np.arange(0, l+stp, stp) # specified uniform locations
jerk = 8
v0 = jerk
v1 = 'stop'
training_list = [0,1,2,3,4,5,6,7] # index of the training sampels#########################################################

# transform nominal acceleration and feed to real acceleration and feed using some
# tuning parameters
def a_nomi2real(a):
    #return 0.91*a
    return 1.0*a
def feed_nomi2real(feed):
    #return 0.93*feed
    return 1.0*feed

# load the line measurement data
with open('S800F9000_xiao120.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        coords_1.append(float(row[0]))
        width_1.append(float(row[1]))
       
print("--------------------------------------------------------")
       
coords_1 = np.asarray(coords_1)
width_1 = np.asarray(width_1)

width_1_ave = np.mean(width_1[(coords_1>start_p)&(coords_1<stop_p)])
width_interp_1  =  np.interp(coords, coords_1 ,width_1)
#width_interp_1 = width_interp_1 - width_1_ave
a = a_nomi2real(800)
feed = feed_nomi2real(150)
l_vector, v_1, a_vector_1, t_vector_1 = acc_ana(a, v0, v1, feed, jerk, l, stp)
data = np.transpose(np.array((coords, t_vector_1, v_1, a_vector_1, width_interp_1)))

print(data)
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
# load the line measurement data 2
coords_2 = []
width_2 = []
with open('S800F6000_xiao120.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        coords_2.append(float(row[0]))
        width_2.append(float(row[1]))
       
coords_2 = np.asarray(coords_2)
print(coords_2)
width_2 = np.asarray(width_2)
print(width_2)
width_2_ave = np.mean(width_2[(coords_2>start_p)&(coords_2<stop_p)])
width_interp_2  =  np.interp(coords, coords_2 ,width_2)
#width_interp_2 = width_interp_2 - width_2_ave
a = a_nomi2real(800)
feed = feed_nomi2real(100)
l_vector, v_2, a_vector_2, t_vector_2 = acc_ana(a, v0, v1, feed, jerk, l, stp)
data = [data, np.transpose(np.array((coords, t_vector_2, v_2, a_vector_2, width_interp_2)))]

print(data)

print("__________________________________________________________________")

# load the line measurement data 3
coords_3 = []
width_3 = []
with open('S800F3000_xiao120.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        coords_3.append(float(row[0]))
        width_3.append(float(row[1]))
       
coords_3 = np.asarray(coords_3)
width_3 = np.asarray(width_3)
width_3_ave = np.mean(width_3[(coords_3>start_p)&(coords_3<stop_p)])
width_interp_3  =  np.interp(coords, coords_3 ,width_3)
#width_interp_3 = width_interp_3 - width_3_ave
a = a_nomi2real(800)
feed = feed_nomi2real(50)
l_vector, v_3, a_vector_3, t_vector_3 = acc_ana(a, v0, v1, feed, jerk, l, stp)
data.append(np.transpose(np.array((coords, t_vector_3, v_3, a_vector_3, width_interp_3))))

# load the line measurement data 4
coords_4 = []
width_4 = []
with open('S600F9000_xiao120.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        coords_4.append(float(row[0]))
        width_4.append(float(row[1]))
       
coords_4 = np.asarray(coords_4)
width_4 = np.asarray(width_4)
width_4_ave = np.mean(width_4[(coords_4>start_p)&(coords_4<stop_p)])
width_interp_4  =  np.interp(coords, coords_4 ,width_4)
#width_interp_4 = width_interp_4 - width_4_ave
a = a_nomi2real(600)
feed = feed_nomi2real(150)
l_vector, v_4, a_vector_4, t_vector_4 = acc_ana(a, v0, v1, feed, jerk, l, stp)
data.append(np.transpose(np.array((coords, t_vector_4, v_4, a_vector_4, width_interp_4))))

# load the line measurement data 5
coords_5 = []
width_5 = []
with open('S600F6000_xiao120.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        coords_5.append(float(row[0]))
        width_5.append(float(row[1]))
       
coords_5 = np.asarray(coords_5)
width_5 = np.asarray(width_5)
width_5_ave = np.mean(width_5[(coords_5>start_p)&(coords_5<stop_p)])
width_interp_5  =  np.interp(coords, coords_5 ,width_5)
#width_interp_5 = width_interp_5 - width_5_ave
a = a_nomi2real(600)
feed = feed_nomi2real(100)
l_vector, v_5, a_vector_5, t_vector_5 = acc_ana(a, v0, v1, feed, jerk, l, stp)
data.append(np.transpose(np.array((coords, t_vector_5, v_5, a_vector_5, width_interp_5))))

# load the line measurement data 6
coords_6 = []
width_6 = []
with open('S600F3000_xiao120.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        coords_6.append(float(row[0]))
        width_6.append(float(row[1]))
       
coords_6 = np.asarray(coords_6)
width_6 = np.asarray(width_6)
width_6_ave = np.mean(width_6[(coords_6>start_p)&(coords_6<stop_p)])
width_interp_6  =  np.interp(coords, coords_6 ,width_6)
#width_interp_6 = width_interp_6 - width_6_ave
a = a_nomi2real(600)
feed = feed_nomi2real(50)
l_vector, v_6, a_vector_6, t_vector_6 = acc_ana(a, v0, v1, feed, jerk, l, stp)
data.append(np.transpose(np.array((coords, t_vector_6, v_6, a_vector_6, width_interp_6))))

# load the line measurement data 7
coords_7 = []
width_7 = []
with open('S400F9000_xiao120.csv') as csvfile:
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
a = a_nomi2real(400)
feed = feed_nomi2real(150)
l_vector, v_7, a_vector_7, t_vector_7 = acc_ana(a, v0, v1, feed, jerk, l, stp)
data.append(np.transpose(np.array((coords, t_vector_7, v_7, a_vector_7, width_interp_7))))

# load the line measurement data 8
coords_8 = []
width_8 = []
with open('S400F6000_xiao120.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        coords_8.append(float(row[0]))
        width_8.append(float(row[1]))
       
coords_8 = np.asarray(coords_8)
width_8 = np.asarray(width_8)
width_8_ave = np.mean(width_8[(coords_8>start_p)&(coords_8<stop_p)])
width_interp_8  =  np.interp(coords, coords_8 ,width_8)
#width_interp_8 = width_interp_8 - width_8_ave
a = a_nomi2real(400)
feed = feed_nomi2real(100)
l_vector, v_8, a_vector_8, t_vector_8 = acc_ana(a, v0, v1, feed, jerk, l, stp)
data.append(np.transpose(np.array((coords, t_vector_8, v_8, a_vector_8, width_interp_8))))

# load the line measurement data 9
coords_9 = []
width_9 = []
with open('S400F3000_xiao120.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        coords_9.append(float(row[0]))
        width_9.append(float(row[1]))
       
coords_9 = np.asarray(coords_9)
width_9 = np.asarray(width_9)
width_9_ave = np.mean(width_9[(coords_9>start_p)&(coords_9<stop_p)])
width_interp_9  =  np.interp(coords, coords_9 ,width_9)
#width_interp_9 = width_interp_9 - width_9_ave
a = a_nomi2real(400)
feed = feed_nomi2real(50)
l_vector, v_9, a_vector_9, t_vector_9 = acc_ana(a, v0, v1, feed, jerk, l, stp)
data.append(np.transpose(np.array((coords, t_vector_9, v_9, a_vector_9, width_interp_9))))

print(data)
print("*********************************************")

## load the line measurement data 10
#coords_10 = []
#width_10 = []
#with open('S400F9000_ino120.csv') as csvfile:
#    readCSV = csv.reader(csvfile, delimiter=',')
#    for row in readCSV:
#        coords_10.append(float(row[0]))
#        width_10.append(float(row[1]))
#        
#coords_10 = np.asarray(coords_10)
#width_10 = np.asarray(width_10)
#width_10_ave = np.mean(width_10[(coords_10>start_p)&(coords_10<stop_p)])
#width_interp_10  =  np.interp(coords, coords_10 ,width_10)
##width_interp_10 = width_interp_10 - width_10_ave
#a = a_nomi2real(400)
#feed = feed_nomi2real(150)
#l_vector, v_10, a_vector_10, t_vector_10 = acc_ana(a, v0, v1, feed, jerk, l, stp)
#data.append(np.transpose(np.array((coords, t_vector_10, v_10, a_vector_10, width_interp_10))))

## load the line measurement data 11
#coords_11 = []
#width_11 = []
#with open('S200F6000_xiao120.csv') as csvfile:
#    readCSV = csv.reader(csvfile, delimiter=',')
#    for row in readCSV:
#        coords_11.append(float(row[0]))
#        width_11.append(float(row[1]))
#        
#coords_11 = np.asarray(coords_11)
#width_11 = np.asarray(width_11)
#width_11_ave = np.mean(width_11[(coords_11>start_p)&(coords_11<stop_p)])
#width_interp_11  =  np.interp(coords, coords_11 ,width_11)
##width_interp_11 = width_interp_11 - width_11_ave
#a = a_nomi2real(200)
#feed = feed_nomi2real(100)
#l_vector, v_11, a_vector_11, t_vector_11 = acc_ana(a, v0, v1, feed, jerk, l, stp)
#data.append(np.transpose(np.array((coords, t_vector_11, v_11, a_vector_11, width_interp_11))))
#
## load the line measurement data 12
#coords_12 = []
#width_12 = []
#with open('S200F3000_xiao120.csv') as csvfile:
#    readCSV = csv.reader(csvfile, delimiter=',')
#    for row in readCSV:
#        coords_12.append(float(row[0]))
#        width_12.append(float(row[1]))
#        
#coords_12 = np.asarray(coords_12)
#width_12 = np.asarray(width_12)
#width_12_ave = np.mean(width_12[(coords_12>start_p)&(coords_12<stop_p)])
#width_interp_12  =  np.interp(coords, coords_12 ,width_12)
##width_interp_12 = width_interp_12 - width_12_ave
#a = a_nomi2real(200)
#feed = feed_nomi2real(50)
#l_vector, v_12, a_vector_12, t_vector_12 = acc_ana(a, v0, v1, feed, jerk, l, stp)
#data.append(np.transpose(np.array((coords, t_vector_12, v_12, a_vector_12, width_interp_12))))

# deepcopy data
data_copy = copy.deepcopy(data)
##=============================================================================
data = [data[i] for i in training_list]
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
   
# the training dataset bp_data contains [bp1_l, bp1_t, v, a, jerk, x, l]
   
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

print(coords)
print(width_interp_1)


p1_idx = identify_breakpoints(coords, width_interp_1)
v = feed_nomi2real(150.00)
a = a_nomi2real(800.00)
jerk = 8.00
#x = (np.square(v) - np.square(jerk))/a
x = v
bp_data = np.array([coords[p1_idx][0], t_vector_1[p1_idx][0], v, a, jerk, x, np.max(coords)],ndmin=2)


p1_idx = identify_breakpoints(coords, width_interp_2)
v = feed_nomi2real(100.00)
a = a_nomi2real(800.00)
jerk = 8.00
#x = (np.square(v) - np.square(jerk))/a
x = v
bp_data = np.concatenate((bp_data, np.array([coords[p1_idx][0], t_vector_2[p1_idx][0], v, a, jerk, x, np.max(coords)],ndmin=2)))


p1_idx = identify_breakpoints(coords, width_interp_3)
v = feed_nomi2real(50.00)
a = a_nomi2real(800.00)
jerk = 8.00
#x = (np.square(v) - np.square(jerk))/a
x = v
bp_data = np.concatenate((bp_data, np.array([coords[p1_idx][0], t_vector_3[p1_idx][0], v, a, jerk, x, np.max(coords)],ndmin=2)))


p1_idx = identify_breakpoints(coords, width_interp_4)
v = feed_nomi2real(150.00)
a = a_nomi2real(600.00)
jerk = 8.00
#x = (np.square(v) - np.square(jerk))/a
x = v
bp_data = np.concatenate((bp_data, np.array([coords[p1_idx][0], t_vector_4[p1_idx][0], v, a, jerk, x, np.max(coords)],ndmin=2)))

p1_idx = identify_breakpoints(coords, width_interp_5)
v = feed_nomi2real(100.00)
a = a_nomi2real(600.00)
jerk = 8.00
#x = (np.square(v) - np.square(jerk))/a
x = v
bp_data = np.concatenate((bp_data, np.array([coords[p1_idx][0], t_vector_5[p1_idx][0], v, a, jerk, x, np.max(coords)],ndmin=2)))

p1_idx = identify_breakpoints(coords, width_interp_6)
v = feed_nomi2real(50.00)
a = a_nomi2real(600.00)
jerk = 8.00
#x = (np.square(v) - np.square(jerk))/a
x = v
bp_data = np.concatenate((bp_data, np.array([coords[p1_idx][0], t_vector_6[p1_idx][0], v, a, jerk, x, np.max(coords)],ndmin=2)))

#width_7 = width_7[9:2525]
#coords_7 = np.linspace(0,120,len(width_7))
p1_idx = identify_breakpoints(coords, width_interp_7)
v = feed_nomi2real(150.00)
a = a_nomi2real(400.00)
jerk = 8.00
#x = (np.square(v) - np.square(jerk))/a
x = v
bp_data = np.concatenate((bp_data, np.array([coords[p1_idx][0], t_vector_7[p1_idx][0], v, a, jerk, x, np.max(coords)],ndmin=2)))

p1_idx = identify_breakpoints(coords, width_interp_8)
v = feed_nomi2real(100.00)
a = a_nomi2real(400.00)
jerk = 8.00
#x = (np.square(v) - np.square(jerk))/a
x = v
bp_data = np.concatenate((bp_data, np.array([coords[p1_idx][0], t_vector_8[p1_idx][0], v, a, jerk, x, np.max(coords)],ndmin=2)))

p1_idx = identify_breakpoints(coords, width_interp_9)
v = feed_nomi2real(50.00)
a = a_nomi2real(400.00)
jerk = 8.00
#x = (np.square(v) - np.square(jerk))/a
x = v
bp_data = np.concatenate((bp_data, np.array([coords[p1_idx][0], t_vector_9[p1_idx][0], v, a, jerk, x, np.max(coords)],ndmin=2)))

#print("++++++++++++++++++++++++++")
#print(bp_data)
#
#print(bp_data.size)
#print("++++++++++++++++++++++++++")
#p1_idx = identify_breakpoints(coords, width_interp_10)
#v = feed_nomi2real(150.00)
#a = a_nomi2real(400.00)
#jerk = 8.00
##x = (np.square(v) - np.square(jerk))/a
#x = v
#bp_data = np.concatenate((bp_data, np.array([coords[p1_idx][0], t_vector_10[p1_idx][0], v, a, jerk, x, np.max(coords)],ndmin=2)))

#p1_idx = identify_breakpoints(coords, width_interp_11)
#v = feed_nomi2real(100.00)
#a = a_nomi2real(200.00)
#jerk = 8.00
##x = (np.square(v) - np.square(jerk))/a
#x = v
#bp_data = np.concatenate((bp_data, np.array([coords[p1_idx][0], t_vector_11[p1_idx][0], v, a, jerk, x, np.max(coords)],ndmin=2)))
#
#p1_idx = identify_breakpoints(coords, width_interp_12)
#v = feed_nomi2real(50.00)
#a = a_nomi2real(200.00)
#jerk = 8.00
##x = (np.square(v) - np.square(jerk))/a
#x = v
#bp_data = np.concatenate((bp_data, np.array([coords[p1_idx][0], t_vector_12[p1_idx][0], v, a, jerk, x, np.max(coords)],ndmin=2)))

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

# test_bp1 = np.arange(200,810,10).reshape(61,1)
test_bp1 = np.arange(300,900,10).reshape(60,1)
bp1_pred2 = bp1_func(test_bp1, *coef_bp1)
plt.figure()
plt.scatter(bp_data[:,3], bp_data[:,1])
#plt.hold(True)

print("''''''''''''''''''''''''''''''''''''''''")
plt.plot(test_bp1, bp1_pred2, 'k')

print("''''''''''''''''''''''''''''''''''''''''")
# select training bp_data
bp_data = bp_data[training_list]

##===========================================================================
##==================fit a function before bp1================================
# tvav
def seg1_func(t_v_a, alpha1, alpha2, alpha3):
    #return alpha1/(alpha2*t_v_a[:,0]*t_v_a[:,1] + alpha3)
    #return alpha1/(alpha2*t_v_a[:,1] + alpha3)
    return 1/(alpha1*t_v_a[:,1] + alpha2)

def seg1_fit(traindata_seg1):
    t_v = traindata_seg1[:,0:2]
    width = traindata_seg1[:,-1]
    popt, pcov = curve_fit(seg1_func, t_v, width)
    width_pred = seg1_func(t_v, *popt)
    plt.figure()
    plt.scatter(t_v[:,1],1./width)
#    plt.hold(True)
#    plt.plot(np.sqrt(t_v_a_v[:,0])*np.sqrt(t_v_a_v[:,1])*np.sqrt(t_v_a_v[:,2]), width_pred)
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
traindata_p1 = generate_traindata(data, bp_data)
traindata_p1_ = traindata_p1[:,[1,2,4]]
coef_p1, width_pred_p1 = seg1_fit(traindata_p1_)

#plt.figure()
#plt.scatter(1/(0.3*traindata_p1_[:,0]*traindata_p1_[:,1]+1.3), traindata_p1_[:,2])
#plt.figure()
#plt.scatter(traindata_p1_[:,1], traindata_p1_[:,2])
#plt.figure()
#plt.scatter(1/traindata_p1_[:,0], traindata_p1_[:,2])

for i in range(len(data)):
    t_v = data[i][:,[1,2]]
    width = data[i][:,4]
    width = width[t_v[:,0]<=bp_data[i,1]]
    t_v = t_v[t_v[:,0]<=bp_data[i,1],:]
    width_pred = seg1_func(t_v,*coef_p1)
    plt.figure()
    plt.plot(t_v[:,0], width)
    #plt.hold(True)
    plt.xlim(0.0, 0.21)
    plt.ylim(0.2, 1.1)
    plt.plot(t_v[:,0], width_pred)
   

##===========================================================================
##==================fit a function after bp1=================================
# bp_2 is the point that the line width reaches constant
##### identify the position/index of bp_2

# thres_list is the thershold treated as the nominal printing-width
#thres_list = [0.8, 0.8, 0.8, 0.75, 0.75, 0.82, 0.8, 0.75, 0.82, 0.71, 0.8, 0.82]
thres_list =  [0.8, 0.8, 0.8, 0.75, 0.75, 0.82, 0.8, 0.75, 0.82]
#thres_list = [0.58, 0.76, 0.8, 0.76, 0.80, 0.90, 0.8, 0.80, 0.85]
#thres_list = [0.58, 0.8, 0.8, 0.76, 0.75, 0.82, 0.8, 0.75, 0.82, 0.85, 0.8, 0.82]
thres_list = [thres_list[i] for i in training_list]

def identify_breakpoint2(width, thrs):
    idx = np.argwhere(width>thrs)
    idx = idx[idx>100] # 100 is to make sure the idx is not near the starting point
    return idx[0]
   
for i in range(len(data)):
    t_vector_tmp = data[i][:,1]
    idx_p2 = identify_breakpoint2(data[i][:,-1], thres_list[i])
    bp_data[i,5] = idx_p2
    plt.figure()
    plt.plot(data[i][:,1], list(reversed(data[i][:,4])))
    #plt.plot(r1, data[i][:,4])
    plt.xticks(fontsize = 10)
    plt.yticks(fontsize = 10)
    plt.xlabel("L (mm)")
    plt.ylabel("Width \n(mm) ", rotation='horizontal',ha='right')
    #plt.xlim(-0.1, 1.1)
    #plt.hold(True)
    #plt.axvline(x = t_vector_tmp[int(idx_p2)])

# there might be breakpoints between bp1 and bp2 which represent the transition from
# acceleration to constant phase
for i in range(len(data)):
    tmp = data[i][:,3]
    idx_pac = np.argwhere(tmp==0)
    idx_pac = idx_pac[0]
    bp_data[i,6] = idx_pac
   

# train a model between bp1 and bp_ac, which represents the acceleration phase
def seg2_func_acc(tvav, beta1):
    return beta1*np.sqrt(tvav[:,0])*np.sqrt(tvav[:,1])*np.sqrt(tvav[:,2])

def seg2acc_fit(traindata_seg2acc):
    t_v_a_v = traindata_seg2acc[:,[0,1,2,4]]
    width = traindata_seg2acc[:,3]
    popt, pcov = curve_fit(seg2_func_acc, t_v_a_v, width)
    width_pred = seg2_func_acc(t_v_a_v, *popt)
    plt.figure()
    plt.scatter(np.sqrt(t_v_a_v[:,0])*np.sqrt(t_v_a_v[:,1])*np.sqrt(t_v_a_v[:,2]), width)
#    plt.hold(True)
    plt.plot(np.sqrt(t_v_a_v[:,0])*np.sqrt(t_v_a_v[:,1])*np.sqrt(t_v_a_v[:,2]), width_pred)
    return popt, width_pred

def generate_traindata2acc(data, bp_data):
    traindata_p2 = np.empty((0,6))
    for i in range(len(data)):
        t_tmp = data[i][:,1]
        t_threshold1 = bp_data[i,1]
        t_threshold2 = t_tmp[int(bp_data[i,6])]
        data_p2 = data[i]
        data_p2 = data_p2[np.logical_and(t_tmp>t_threshold1, t_tmp<t_threshold2),:]
        if len(data_p2)!=0:
            data_p2[:,1] = data_p2[:,1] - data_p2[0,1]# make t start from zero
            data_p2[:,4] = data_p2[:,4] - data_p2[0,4]# make width start from zero
            top_speed = np.max(data[i][:,2]) + np.zeros((len(data_p2),1))
            data_p2 = np.concatenate((data_p2,top_speed), axis=1)
            traindata_p2 = np.concatenate((traindata_p2, data_p2))
    return traindata_p2
traindata_p2acc = generate_traindata2acc(data, bp_data)
print(traindata_p2acc)
traindata_p2acc_ = traindata_p2acc[:,[1,2,3,4,5]]
coef_p2acc, width_pred_p2acc = seg2acc_fit(traindata_p2acc_)
for i in range(len(data)):
    t_v_a = data[i][:,[1,2,3]]
    top_speed = np.max(data[i][:,2]) + np.zeros((len(t_v_a),1))
    t_v_a_v = np.concatenate((t_v_a, top_speed), axis=1)
    width = data[i][:,4]
    t_threshold1 = bp_data[i,1]
    t_threshold2 = t_v_a_v[int(bp_data[i,6]),0]
    width = width[np.logical_and(t_v_a_v[:,0]>t_threshold1, t_v_a_v[:,0]<t_threshold2)]
    if len(width)!=0:
        width = width -width[0]
        t_v_a_v = t_v_a_v[np.logical_and(t_v_a_v[:,0]>t_threshold1, t_v_a_v[:,0]<t_threshold2),:]
        t_v_a_v[:,0] = t_v_a_v[:,0] - t_v_a_v[0, 0]
        width_pred = seg2_func_acc(t_v_a_v,*coef_p2acc)
        plt.figure()
        plt.plot(t_v_a_v[:,0], width)
        #plt.hold(True)
        plt.xlim(0.0, 0.8)
        plt.ylim(-0.1, 0.5)
#        plt.figure()
#       plt.hold(True)
#       plt.xlim(0.0, 0.8)
#       plt.ylim(-0.1, 0.5)
        plt.plot(t_v_a_v[:,0], width_pred)
        plt.title(str(i))


# train a model between bp1 and bp_ac, which represents the speed constant phase
def seg2_func(tvav, beta1, beta2):# tva refers to time, speed, acceleration, top speed.
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
    plt.figure()
    plt.scatter(np.sqrt(t_v_a_v[:,0])*np.sqrt(t_v_a_v[:,1]), width)
#    plt.hold(True)
    plt.plot(np.sqrt(t_v_a_v[:,0])*np.sqrt(t_v_a_v[:,1]), width_pred)
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
traindata_p2 = generate_traindata2(data, bp_data)
traindata_p2_ = traindata_p2[:,[1,2,3,4,5]]
coef_p2, width_pred_p2 = seg2_fit(traindata_p2_)

for i in range(len(data)):
    t_v_a = data[i][:,[1,2,3]]
    top_speed = np.max(data[i][:,2]) + np.zeros((len(t_v_a),1))
    t_v_a_v = np.concatenate((t_v_a, top_speed), axis=1)
    width = data[i][:,4]
    t_threshold1 = t_v_a_v[int(bp_data[i,6]),0]
    t_threshold2 = t_v_a_v[int(bp_data[i,5]),0]
    width = width[np.logical_and(t_v_a_v[:,0]>t_threshold1, t_v_a_v[:,0]<t_threshold2)]
    if len(width)!=0:
        width = width -width[0]
        t_v_a_v = t_v_a_v[np.logical_and(t_v_a_v[:,0]>t_threshold1, t_v_a_v[:,0]<t_threshold2),:]
        t_v_a_v[:,0] = t_v_a_v[:,0] - t_v_a_v[0, 0]
        width_pred = seg2_func(t_v_a_v,*coef_p2)
        plt.figure()
        plt.plot(t_v_a_v[:,0], width)
        #plt.hold(True)
        plt.xlim(0.0, 0.8)
        plt.ylim(-0.1, 0.5)
#    plt.figure(31)
#    plt.hold(True)
#    plt.xlim(0.0, 0.8)
#    plt.ylim(-0.1, 0.5)
        plt.plot(t_v_a_v[:,0], width_pred)
        plt.title(str(i))


##===========================================================================
##==================fit a function for the decelerating phase================
##===========================================================================
def seg3_func(tvav, gamma1, gamma2, gamma3):
    #return gamma1*tvav[:,3]/np.sqrt(tvav[:,1])
    #return gamma1*tvav[:,3]/tvav[:,1]
    return gamma1*np.log(tvav[:,3]/tvav[:,1])

def seg3_fit(traindata_seg3):
    t_v_a_v = traindata_seg3[:,[0,1,2,4]]
    width = traindata_seg3[:,3]
    popt, pcov = curve_fit(seg3_func, t_v_a_v, width)
    width_pred = seg3_func(t_v_a_v, *popt)
    plt.figure()
    plt.scatter(np.log(t_v_a_v[:,3]/t_v_a_v[:,1]), width)
#    plt.hold(True)
    plt.plot(np.log(t_v_a_v[:,3]/t_v_a_v[:,1]), width_pred)
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
traindata_p3 = generate_traindata3(data)
traindata_p3_ = traindata_p3[:,[1,2,3,4,5]]
coef_p3, width_pred_p3 = seg3_fit(traindata_p3_)
#coef_p3=np.array([0.08128630999708565, 1.        , 1.        ])

for i in range(len(data)):
    t_v_a = data[i][:,[1,2,3]]
    top_speed = np.max(data[i][:,2]) + np.zeros((len(t_v_a),1))
    t_v_a_v = np.concatenate((t_v_a, top_speed), axis=1)
    width = data[i][:,4]
    a_tmp = data[i][:,3]
    idx_dec = np.min(np.argwhere(a_tmp<0))
    width = width[idx_dec:]
    width = width - width[0]
    t_v_a_v = t_v_a_v[idx_dec:,:]
    t_v_a_v[:,0] = t_v_a_v[:,0] - t_v_a_v[0, 0]
    width_pred = seg3_func(t_v_a_v,*coef_p3)
    plt.figure()
    plt.plot(t_v_a_v[:,0], width)
    #plt.hold(True)
    plt.xlim(0.0, 0.8)
    plt.ylim(-0.1, 1.4)
##    plt.figure(31)
##    plt.hold(True)
##    plt.xlim(0.0, 0.8)
##    plt.ylim(-0.1, 0.5)
    plt.plot(t_v_a_v[:,0], width_pred)

##=============================================================================
##============================prediction function==============================
##=============================================================================
#start_list = [0.4,0.45,0.61,0.55,0.62,0.7,0.6,0.6,0.73]

def forward_pred(ltva_va):
    # predict the breakpoint bp1
    av_bp1 = ltva_va[:,[5,4]]
    bp1 = bp1_func(av_bp1,*coef_bp1)
    bp1 = bp1[0]
#    bp1=0
#    pred_p1=[0.5]
#    final_pred=[]
    #print(bp1)
    # make predictions for p1
    t_v_p1 = ltva_va[ltva_va[:,1]<=bp1, :]
    t_v_p1 = t_v_p1[:,[1,2]]
    pred_p1 = seg1_func(t_v_p1,*coef_p1)
    final_pred = pred_p1
    # make predictions for p3
    idx_dec = np.min(np.argwhere(ltva_va[:,3]<0))
    tvav_p3 = ltva_va[idx_dec:,[1,2,3,4]]
    tvav_p3[:,0] = tvav_p3[:,0] - tvav_p3[0,0]
    pred_p3 = seg3_func(tvav_p3, *coef_p3) + 0.82
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
        pred_p2[pred_p2>=0.82] = 0.82
        final_pred = np.concatenate((final_pred, pred_p2, pred_p3))
    return final_pred
plt.figure(figsize=(8,5))
data_idx = 8
ltva_va = data_copy[data_idx][:,[0,1,2,3]]
top_speed = np.max(data_copy[data_idx][:,2]) + np.zeros((len(ltva_va),1))
acc = np.max(data_copy[data_idx][:,3]) + np.zeros((len(ltva_va),1))
ltva_va = np.concatenate((ltva_va, top_speed, acc), axis = 1)
pred = forward_pred(ltva_va)

plt.plot(data_copy[data_idx][:,1], data_copy[data_idx][:,-1],linestyle='dotted',linewidth = 3,label="True width")
#plt.plot(coords, data_copy[data_idx][:,-1])##################################################
#plt.hold(True)
plt.plot(data_copy[data_idx][:,1], pred,label="Predicted width")


plt.rcParams.update({'font.size': 15})
plt.title("F:150 mm/s; A:400 mm/s\u00b2")
plt.ylim(0.2, 1.6)
plt.xlim(-0.05, 2.6)
plt.legend(loc= "upper right",fontsize=15)

plt.ylabel("Width \n(mm) ", rotation='horizontal',ha='right')
plt.xlabel("T(s) ")

RMSE = np.sqrt(np.sum((pred - data_copy[data_idx][:,-1])**2)/len(pred))
Text = 'RMSE: '+str(round(RMSE,3))
plt.text(1.8,0.25, Text, fontsize = 15)
print("---------------------------------------")
print("RMSE = ",RMSE)

plt.plot(data_copy[data_idx][:,1], data_copy[data_idx][:,-1]-pred)
plt.ylim(0.2, 1.6)
plt.xlim(-0.05, 2.6)

print(ltva_va)
   
import csv
with open('outputtest.csv', 'w', newline='') as csvfile:

  writer = csv.writer(csvfile, delimiter=' ')


  for i in pred:
      writer.writerow([str(i)])
     
plt.figure(figsize=(8,5))  

plt.plot(r1,data_copy[data_idx][:,-1])
plt.ylim(0.2, 1.3)


##===========================================================================
############################# END ############################################
##===========================================================================