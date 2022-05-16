# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 10:01:02 2020

@author: USER
"""

import numpy as np
import matplotlib.pyplot as plt



ticks = ['1']
ticks1 = ['1','2','3','4','5']

Single_learning1 =  [0.0712228129168331]
Robin_Method_median1 = [0.058876131787390984]
Single_learning = [ 0.0712228129168331, 0.0712228129168331, 0.0712228129168331, 0.0712228129168331]
Jie_Method_median = [0.07483087919046237,0.05455747461803706,0.05009346173677695,0.04607145206210349]
Robin_Method_median = [0.05511740785806938,0.053201482086034084,0.04679664152823218,0.04392745461146458]
x1 = np.arange(1)
x = np.arange(1,5)  # the label locations
width = 0.3  # the width of the bars

plt.figure(figsize=(5,5))


RMSE_GAP = [0.17335289949667312,0.2261270567559584,0.25302750751844827,0.342954320227753,0.38323898183074184]
RMSE_GAP_tl = [-0.050658856704275425,0.23398876871454324,0.29666549683637045,0.3531363031687455]



fig, ax=plt.subplots(figsize=(10,10))
ax.set_ylim([0.04,0.09])

ax.bar(x1,Robin_Method_median1,color="gold",width = 0.3,label = "Median RMSE in co-learning proposed")
ax.bar(x-0.15,Jie_Method_median,color="darkturquoise",width = 0.3,label = "Median RMSE in transfer learning (Ren et al., 2021)")

ax.bar(x+0.15,Robin_Method_median,color="gold",width = 0.3,label = "Median RMSE in co-learning proposed")


def legend_without_duplicate_labels(ax):
    current_handles, current_labels = plt.gca().get_legend_handles_labels()

    # sort or reorder the labels and handles
    reversed_handles = list(reversed(current_handles))
    reversed_labels = list(reversed(current_labels))
    unique = [(h, l) for i, (h, l) in enumerate(zip(reversed_handles, reversed_labels)) if l not in reversed_labels[:i]]
    ax.legend(loc="upper right", bbox_to_anchor=(1, 0., 0., 0.91),*zip(*unique))

legend_without_duplicate_labels(ax)


ax.set_xticks([i for i in range(5)], ticks1)
ax.set_xlabel("Taz 5's sample size",fontsize=20)
ax.set_ylabel("RMSE (mm)",fontsize=20)

ax2 = ax.twinx()
#ax3 = ax.twinx()
ax2.set_ylim([-0.1,0.51])


ax2.scatter(x,RMSE_GAP_tl,color="blue",marker = "o", s=100)
ax2.plot(x,RMSE_GAP_tl,color ="blue",linestyle = "--",linewidth=3,label="Median RMSE Reduction in transfer learning (Ren et al., 2021)")
ax2.scatter(x1,RMSE_GAP[0],color="red",marker = "o", s=100)
ax2.scatter(x,RMSE_GAP[1:5],color="red",marker = "o", s=100)
ax2.plot(RMSE_GAP,color ="red",linestyle= ":",linewidth=3,label="Median RMSE Reduction in co-learning proposed")


def legend_without_duplicate_labels1(ax2):
    current_handles, current_labels = plt.gca().get_legend_handles_labels()

    # sort or reorder the labels and handles
    reversed_handles = list(reversed(current_handles))
    reversed_labels = list(reversed(current_labels))
    unique = [(h, l) for i, (h, l) in enumerate(zip(reversed_handles, reversed_labels)) if l not in reversed_labels[:i]]
    ax2.legend(loc="upper right", bbox_to_anchor=(1, 0., 0., 1),*zip(*unique))

legend_without_duplicate_labels1(ax2)

ax2.set_ylabel("RMSE Redcution (%)",fontsize = 20)


plt.xticks([i for i in range(5)], ticks1)


plt.show()

