# Two-stage optimization for solving co-learning problem. #

## Section 5 ##

For the validation of the kinematics piecewise fucntion proposed by (Ren. et al 2021). The file "linewidth estimation.py" is attached under the folder called "piecewise model (Ren 2021).zip". All the linewidth_data is provided in "Linewidth_Data.zip". The running results will be like...

![image](https://user-images.githubusercontent.com/105607708/168645847-e6ca692e-465c-4e8e-8e44-9c8cf2c755d9.png)

### Note: Change different g-code in the code is allowalbe, but the user must manully change the following parameters as well. Such as...

training_list = [0,1,2,3,4,5,6,7] # index of the training sampels, I selected 8 samples (0-7) for the training data, 8 for the prediction.

thres_list =  [0.8, 0.8, 0.8, 0.75, 0.75, 0.82, 0.8, 0.75, 0.82] # is the the starting linewidth for p4 based on the index.

def forward_pred(ltva_va):

  pred_p3 = seg3_func(tvav_p3, *coef_p3) + 0.82 # change 0.82 to the corresponing threshold in thres_list.

  pred_p2[pred_p2>=0.82] = 0.82 # change 0.82 to the corresponing threshold in thres_list.

  data_idx = 8 # Predict the index of the g-code you want.

## Section 6.1 ##

## Section 6.2 ##
