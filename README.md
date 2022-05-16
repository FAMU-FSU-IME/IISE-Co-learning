# Two-stage optimization for solving co-learning problem. #

## Section 4: The validation for the kinematic-piecewise model in Eqn.(1). 

For the validation of the kinematics piecewise fucntion proposed by (Ren. et al 2021). The file "linewidth estimation.py" is attached under the folder called "piecewise model (Ren 2021).zip". All the linewidth_data is provided in "Linewidth_Data.zip". The running results will be like...

![image](https://user-images.githubusercontent.com/105607708/168645847-e6ca692e-465c-4e8e-8e44-9c8cf2c755d9.png)

### Note: Change different g-code in the code is allowalbe, but the user must manully change the following parameters in the code as well. Such as...

training_list = [0,1,2,3,4,5,6,7] # index of the training sampels, I selected 8 samples (0-7) for the training data, 8 for the prediction.

thres_list =  [0.8, 0.8, 0.8, 0.75, 0.75, 0.82, 0.8, 0.75, 0.82] # is the the starting linewidth for p4 based on the index.

def forward_pred(ltva_va):

  pred_p3 = seg3_func(tvav_p3, *coef_p3) + 0.82 # change 0.82 to the corresponing threshold in thres_list.

  pred_p2[pred_p2>=0.82] = 0.82 # change 0.82 to the corresponing threshold in thres_list.

  data_idx = 8 # Predict the index of the g-code you want.
  
## Section 5 ##
For the 95%-NCI of posterior common covariance. The file "MCMC_covariance_estimation.py" and "HPD.py" are attached under the folder called "Section 5: Case_study.zip". The running results will be like...

![image](https://user-images.githubusercontent.com/105607708/168651711-422e97e7-2b02-4bb1-b140-4a34766af7c1.png)

Autocorrelation within 1000 lags

![image](https://user-images.githubusercontent.com/105607708/168652316-208be8b1-d52b-4e34-abc9-ed5a04c19ab7.png)


### Note: Change combination of g-codes for the co-learning in the code is allowalbe, but the user must manully change the following parameters in the code as well. Such as...

##### Part 1: MCMC for covariance estimation (Running this first, since the MCMC samples will be written in the newly created file called "newfilePath.csv") #####
In "MCMC_covariance_estimation.py", we have 26 differents g-codes from 4 printers example ,___ino120_1.csv (Taz6), xiao120.csv (Taz5), ___101ts.csv (Ender 1), ___101s.csv (Ender 2).

Since the Taz 6 doesn't have p1, for the coding purpose, we manually created p1 for ___ino120.csv --> ___ino120_1.csv .

However, in the paper each printer we include only 5 g-codes from TABLE 1 (paper) for each pritner. The index for the g-code will become:

IISE paper conditions in Table 1 for each printer.

xiao120 (Taz 5): 0, 3,5, 6, 8;  ino120 (Taz 6): 9, 11, 13, 14, 15; 101ts(Ender 1): 16, 17, 18, 19, 20; 101s(Ender 2): 21, 22, 23, 24, 25

In the code:

sample_training_list = [0,9] # Choose the combination for g-codes from different printers. 

##### Part 2: Simulated Annealing (Run this code after finishing running the code in Part 1) #####

In "colearning_with_SA.py", the resutls for one target g-code (ino120 (Taz 6)) in co-learning and validated by the rest of non-selected four g-codes in TABLE 1 (paper) from ino120 (Taz 6) will be generated as follows,

In the code:
training_list = [0] #One g-code training for (ino120)Taz 6        
test_list = [3,5,6,8] #The rest of non-selected testing g-codes in TABLE 1 for (ino120)Taz 6.  

![image](https://user-images.githubusercontent.com/105607708/168660075-e533aa74-505f-43fc-9286-53fc57aab438.png)

Row 1: The RMSEs validated by the rest of non-selected printer in single printer learning. 

Row 2: The RMSEs validated by the rest of non-selected printer in co-learning proposed. 

3: Total RMSEs in single printer leanring. 4: Total RMSEs in co-leanring proposed. 5: RMSE reduction 6: The covariance we found so far. 

The RMSE reduction (%) through the iteration as follows,

![image](https://user-images.githubusercontent.com/105607708/168660127-590b60fb-6da4-43af-9600-f6105e225517.png)


In between-printer learning comparison, the code must be adjusted manually when the training g-code changes. For exmaple:

training_list = [0]       
test_list = [3,5,6,8] 

training_list = [3]       
test_list = [0,5,6,8] 

training_list = [5]       
test_list = [0,3,6,8] 

Based on code, we randomly picked a number of of g-code-combinations in a certain size compared with the results obtained from prior study (Transfer learning (Ren. et al 2021)). Also, "Ren_vs_proposed.py", are provided for presenting the results I attached in Fig. 7 (paper). The covariance estimation (def cov_cal) from the prior stduy also attached in "MCMC_covariance_estimation.py"

![image](https://user-images.githubusercontent.com/105607708/168661481-34143a3a-3541-4d40-9546-8c632ad5a30c.png)

**For this results, user needs to include the g-code only from xiao120(Taz 5: one-gcode contained only) and ino120 (Taz 6: depends on the varying size)** 


## Section 6.1 ##

## Section 6.2 ##

