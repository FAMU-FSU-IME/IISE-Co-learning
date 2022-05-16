# Two-stage optimization for solving co-learning problem. #

## Section 4: The validation for the kinematic-piecewise model in Eqn.(1). 

For the validation of the kinematics piecewise fucntion proposed by (Ren. et al 2021). The file "linewidth estimation.py" is attached under the folder called "piecewise model (Ren 2021).zip". All the linewidth_data is provided in "Linewidth_Data.zip". The running results will be like...

![image](https://user-images.githubusercontent.com/105607708/168645847-e6ca692e-465c-4e8e-8e44-9c8cf2c755d9.png)

### Note: Change different g-code in the code is allowalbe, but the user must manully change the following parameters as well. Such as...

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


### Note: Change combination of g-codes for the co-learning in the code is allowalbe, but the user must manully change the following parameters as well. Such as...

In "MCMC_covariance_estimation.py", we have 26 differents g-codes from 4 printers example ,___ino120_1.csv (Taz6), xiao120.csv (Taz5), ___101ts.csv (Ender 1), ___101s.csv (Ender 2).

Since the Taz 6 doesn't have p1, for the coding purpose, we manually created p1 for ___ino120 right ___ino120_1

However, in the paper each printer we include only 5 g-codes from TABLE 1 (paper) for each pritner. The index for the g-code will become:




## Section 6.1 ##

## Section 6.2 ##
