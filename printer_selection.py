# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 01:56:13 2020

@author: aw18j
"""
#from Xiao_Lacking_Covariance_Estimation import cor
import numpy as np
print("Test 3")
print("----------------------Xiao-----------------------------")


RMSE_1=[0.06766900183825711, 0.15950909644314834, 0.0785527041534568, 0.12106314409862311,0.038173705600765176, 0.05037796416979792, 0.05514013388902525, 0.045655641229201044,0.1298606481231939, 0.1432828456242262, 0.1637187358379245, 0.04054072627428842,0.02734631447077954, 0.05225109510152264, 0.040897279035297496, 0.04316812589820332,0.07477662399540912, 0.08833747763997478, 0.07770026787265054, 0.10171205004605288]
new_RMSE_1 =[0.0537725263167941, 0.08430378847478928, 0.04093232511922538, 0.055561896694671556,0.0367475962668536, 0.050513733424673656, 0.05365231725220014, 0.04523700187839124,0.12968843272939592, 0.14136876436774398, 0.16356244389059968, 0.04056351756260248,0.02807123999667238, 0.044892954899932634, 0.03780458006108233, 0.04351857189247087,0.06804207376586908, 0.08278276039948287, 0.06995600725834865, 0.0959898851072254]


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

print("---------------------Top_Ender------------------------------")


RMSE_2=[0.06766900183825711, 0.15950909644314834, 0.0785527041534568, 0.12106314409862311,0.038173705600765176, 0.05037796416979792, 0.05514013388902525, 0.045655641229201044,0.1298606481231939, 0.1432828456242262, 0.1637187358379245, 0.04054072627428842,0.02734631447077954, 0.05225109510152264, 0.040897279035297496, 0.04316812589820332,0.07477662399540912, 0.08833747763997478, 0.07770026787265054, 0.10171205004605288]

new_RMSE_2=[0.05293943726464142, 0.11025493539027022, 0.05308522168498284, 0.07682507470713658,0.03696373880609722, 0.05048402426954943, 0.05386079454708714, 0.04542377065627267,0.12593660485124902, 0.1396280401026677, 0.16007386980949675, 0.041110589615091016,0.027826114486950975, 0.04474827322583036, 0.03820792400401319, 0.04355149186243422,0.07463050409041243, 0.0816485257736154, 0.07689444148839901, 0.10159215594810848]

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

print("---------------------Ender------------------------------")

RMSE_3= [0.06766900183825711, 0.15950909644314834, 0.0785527041534568, 0.12106314409862311,0.038173705600765176, 0.05037796416979792, 0.05514013388902525, 0.045655641229201044,0.1298606481231939, 0.1432828456242262, 0.1637187358379245, 0.04054072627428842,0.02734631447077954, 0.05225109510152264, 0.040897279035297496, 0.04316812589820332,0.07477662399540912, 0.08833747763997478, 0.07770026787265054, 0.10171205004605288]
new_RMSE_3=[0.06528275536904223, 0.15748281758181007, 0.0775018171232006, 0.11919532207697876,0.036192103388094685, 0.042077519146340135, 0.04718234235760525, 0.04894617573887489,0.1297796525288049, 0.14146850157454047, 0.1636452673128833, 0.04055140319051341,0.027346293788610677, 0.052251093854695715, 0.040897282510495195, 0.04316806620643853,0.0694713181272931, 0.08396482829844301, 0.07156191037599421, 0.09724333593385415]
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


print("---------------------Xiao_Ender 1------------------------------")

RMSE_4= [0.06766900183825711, 0.15950909644314834, 0.0785527041534568, 0.12106314409862311,0.038173705600765176, 0.05037796416979792, 0.05514013388902525, 0.045655641229201044,0.1298606481231939, 0.1432828456242262, 0.1637187358379245, 0.04054072627428842,0.02734631447077954, 0.05225109510152264, 0.040897279035297496, 0.04316812589820332,0.07477662399540912, 0.08833747763997478, 0.07770026787265054, 0.10171205004605288]
new_RMSE_4= [0.05147587227535162, 0.0551449255472996, 0.03008737245978521, 0.0401361085912881,0.03633953317473648, 0.0401890858792029, 0.0436900448549393, 0.05061815942652853,0.10362321604748112, 0.11969887772310497, 0.13587252981226391, 0.04510811166699776,0.028299058944600212, 0.0452881693507984, 0.03768996142529867, 0.043443937857468376,0.05373130061205987, 0.07032623813264673, 0.05574936047096652, 0.08028489983233257]
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


print("---------------------Xiao_Ender 1 2------------------------------")

RMSE_5= [0.06766900183825711, 0.15950909644314834, 0.0785527041534568, 0.12106314409862311,0.038173705600765176, 0.05037796416979792, 0.05514013388902525, 0.045655641229201044,0.1298606481231939, 0.1432828456242262, 0.1637187358379245, 0.04054072627428842,0.02734631447077954, 0.05225109510152264, 0.040897279035297496, 0.04316812589820332,0.07477662399540912, 0.08833747763997478, 0.07770026787265054, 0.10171205004605288]
new_RMSE_5= [0.028063127870573825, 0.04581358001460366, 0.037909573632248034, 0.04336849654313649,0.07462566637597844, 0.08160424679851791, 0.07687790580149816, 0.10158818385743339,0.05349081058375571, 0.040269618173687634, 0.028237848637095036, 0.04337873737986411,0.035964439898427784, 0.050716825451520564, 0.05306726957449788, 0.045071299101077024,0.12972010997953684, 0.14141016697712913, 0.16359121513048527, 0.04055930208269486]
RMSE_median_5 = np.median(RMSE_5)

new_RMSE_median_5 = np.median(new_RMSE_5)

print("RMSE_median with Single learning",RMSE_median_5)
print("25th percentile:", np.percentile(RMSE_5,25))
print("50th percentile:", np.percentile(RMSE_5,50))
print("75th percentile:", np.percentile(RMSE_5,75))

print("RMSE_median with Co-learning proposed:",new_RMSE_median_5)
print("25th percentile:", np.percentile(new_RMSE_5,25))
print("50th percentile:", np.percentile(new_RMSE_5,50))
print("75th percentile:", np.percentile(new_RMSE_5,75))

print("RMSE reduction:",(RMSE_median_5-new_RMSE_median_5)/RMSE_median_5)

print("---------------------Ender 1 2------------------------------")

RMSE_6= [0.06766900183825711, 0.15950909644314834, 0.0785527041534568, 0.12106314409862311,0.038173705600765176, 0.05037796416979792, 0.05514013388902525, 0.045655641229201044,0.1298606481231939, 0.1432828456242262, 0.1637187358379245, 0.04054072627428842,0.02734631447077954, 0.05225109510152264, 0.040897279035297496, 0.04316812589820332,0.07477662399540912, 0.08833747763997478, 0.07770026787265054, 0.10171205004605288]
new_RMSE_6= [0.06482537781240111, 0.15846828465326615, 0.07804506243119287, 0.12016070141498683,0.0375558027045145, 0.05042259355959361, 0.05446879018546839, 0.045543883520373286,0.1287105797171397, 0.1422178093632871, 0.16266906364981126, 0.04069877977862182,0.02726306578139045, 0.05017598952298255, 0.0398829990640228, 0.04315317977714607,0.05555462202055703, 0.07206292780718739, 0.05775109534590223, 0.0829616171028368]
RMSE_median_6 = np.median(RMSE_6)

new_RMSE_median_6 = np.median(new_RMSE_6)

print("RMSE_median with Single learning",RMSE_median_6)
print("25th percentile:", np.percentile(RMSE_6,25))
print("50th percentile:", np.percentile(RMSE_6,50))
print("75th percentile:", np.percentile(RMSE_6,75))

print("RMSE_median with Co-learning proposed:",new_RMSE_median_6)
print("25th percentile:", np.percentile(new_RMSE_6,25))
print("50th percentile:", np.percentile(new_RMSE_6,50))
print("75th percentile:", np.percentile(new_RMSE_6,75))

print("RMSE reduction:",(RMSE_median_6-new_RMSE_median_6)/RMSE_median_6)

print("---------------------Xiao_Ender 2------------------------------")

RMSE_7= [0.06766900183825711, 0.15950909644314834, 0.0785527041534568, 0.12106314409862311,0.038173705600765176, 0.05037796416979792, 0.05514013388902525, 0.045655641229201044,0.1298606481231939, 0.1432828456242262, 0.1637187358379245, 0.04054072627428842,0.02734631447077954, 0.05225109510152264, 0.040897279035297496, 0.04316812589820332,0.07477662399540912, 0.08833747763997478, 0.07770026787265054, 0.10171205004605288]
new_RMSE_7= [0.0619716969824046, 0.15278204072096355, 0.07502568085158534, 0.11480027156279617,0.03615263633798364, 0.04135101234728779, 0.046451768401787226, 0.049262129294158574,0.1253557638192593, 0.13908149852369872, 0.15951768370572061, 0.04120157129022571,0.027645836113494728, 0.045287173984053145, 0.03879572880506265, 0.04344410319641469,0.07444822713911509, 0.08140878206590663, 0.0766687359561847, 0.10144237557740561]
RMSE_median_7 = np.median(RMSE_7)

new_RMSE_median_7 = np.median(new_RMSE_7)

print("RMSE_median with Single learning",RMSE_median_7)
print("25th percentile:", np.percentile(RMSE_7,25))
print("50th percentile:", np.percentile(RMSE_7,50))
print("75th percentile:", np.percentile(RMSE_7,75))

print("RMSE_median with Co-learning proposed:",new_RMSE_median_7)
print("25th percentile:", np.percentile(new_RMSE_7,25))
print("50th percentile:", np.percentile(new_RMSE_7,50))
print("75th percentile:", np.percentile(new_RMSE_7,75))

print("RMSE reduction:",(RMSE_median_7-new_RMSE_median_7)/RMSE_median_7)
import matplotlib.pyplot as plt

RMSE_all = [RMSE_1]
RMSE_all_tl = [new_RMSE_1,new_RMSE_2,new_RMSE_3,new_RMSE_4,new_RMSE_7,new_RMSE_6,new_RMSE_5]

data_a = RMSE_all
data_b = RMSE_all_tl

ticks = ['Taz 6','Taz 6\n+Taz 5', 'Taz 6\n+Ender 1', 'Taz 6\n+Ender 2','Taz 6\n+Taz 5\n+Ender 1','Taz 6\n+Taz 5\n+Ender 2','Taz 6\n+Ender1\n+Ender 2','Taz 6\n+Taz 5\n+Ender 1\n+Ender 2']

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

plt.figure(figsize=(40,20))

boxprops = dict(linestyle='-', linewidth=5)
medianprops = dict(linestyle='-', linewidth=5)
bpl = plt.boxplot(data_a, positions=np.array(range(len(data_a)))*7.0, sym='', medianprops=medianprops,boxprops=boxprops,widths=2,whis = (25,75))
bpr = plt.boxplot(data_b, positions=np.array(range(1,len(data_b)+1))*7.0, sym='',medianprops=medianprops ,boxprops=boxprops,widths=2,whis = (25,75))
#set_box_color(bpl, '#1f77b4')
set_box_color(bpl, '#1f77b4')
#set_box_color(bpr, '#ff7f0e')              # colors are from http://colorbrewer2.org/
set_box_color(bpr, '#1f77b4')

## draw temporary red and blue lines and use them to create a legend
#plt.plot([], c='#1f77b4', label='IQR of single-printer learning')
#plt.plot([], c='#ff7f0e', label='IQR of co-learning proposed with different printer(s)')
         
leg=plt.legend(fontsize = 50)        

leg_lines = leg.get_lines()

plt.setp(leg_lines, linewidth=5)
 


#plt.title("Co-learning with each Source \n(When the Target and the Source have different G-codes)",fontsize =40)
plt.ylabel("RMSE (mm)",fontsize=50)
plt.xticks(range(0, len(ticks) * 7, 7), ticks,fontsize=50)
plt.yticks(fontsize=50)
plt.xlim(-2, len(ticks)*7)
plt.ylim(0.02, 0.12)

plt.tight_layout()

#print("The median of single learning: ",RMSE_median_1)
#print("The average median of Co-learning proposed from the three tests: ", (new_RMSE_median_1+new_RMSE_median_2+new_RMSE_median_3)/3)
#print("RMSE reduction: ", (RMSE_median_1-((new_RMSE_median_1+new_RMSE_median_2+new_RMSE_median_3)/3))/RMSE_median_1)
# plt.savefig('boxcompare.png')