# 04 pipeline | SAMPLE=1 CALENDAR=1 N_SEEDS=2 origins=[457, 557, 657]
[     0s] loaded 132,026 rows, 120 stocks, train days 0-804, test days 805-1151
calendar check: {'n_third_fridays': 54, 'share_matched': 1.0, 'n_half_days': 11, 'share_half_days_next_to_holiday': 1.0, 'first_date': '2014-06-02', 'last_date': '2018-12-28'}
[     1s] features ready: 157 columns
[    73s] origin 457 done (49,347 train rows, 41,114 val rows, 347 days)
[   147s] origin 557 done (61,129 train rows, 29,332 val rows, 247 days)
[   222s] origin 657 done (72,938 train rows, 17,523 val rows, 147 days)

## Validation, MSE by training end (rows sorted by mean)
fold                                            457     557     657    mean
model                                                                      
stock_blend_300 + true day effect            0.3011  0.2889  0.2836  0.2912
stock_blend_600 + true day effect            0.3027  0.2893  0.2829  0.2916
stock_blend_900 + true day effect            0.3044  0.2905  0.2834  0.2928
stock_blend_600 leaves127 + true day effect  0.3036  0.2918  0.2847  0.2934
stock_l1_600 + true day effect               0.3036  0.2910  0.2862  0.2936
stock_l1_300 + true day effect               0.3028  0.2914  0.2871  0.2938
stock_l1_900 + true day effect               0.3040  0.2910  0.2864  0.2938
stock_l2_300 + true day effect               0.3059  0.2929  0.2845  0.2944
stock_l2_600 + true day effect               0.3101  0.2958  0.2851  0.2970
stock_l2_900 + true day effect               0.3145  0.2994  0.2865  0.3001
FULL cal trend2                              0.3337  0.3185  0.3057  0.3193
FULL cal main                                0.3469  0.3276  0.3036  0.3260
FULL cal trend0                              0.3676  0.3459  0.3104  0.3413
FULL nocal trend2                            0.3819  0.3523  0.3499  0.3614
FULL nocal main                              0.3972  0.3578  0.3480  0.3677
FULL nocal trend0                            0.4204  0.3743  0.3558  0.3835
ref_lin_gbm                                  0.4889  0.4667  0.4164  0.4573
ref_linear                                   0.6362  0.6012  0.5955  0.6110
ref_pid_mean                                 0.6231  0.6255  0.6143  0.6210

## Validation, MAE
fold                                            457     557     657    mean
model                                                                      
stock_l1_300 + true day effect               0.4083  0.3989  0.3904  0.3992
stock_l1_600 + true day effect               0.4093  0.3993  0.3899  0.3995
stock_l1_900 + true day effect               0.4099  0.3995  0.3899  0.3998
stock_blend_300 + true day effect            0.4099  0.4004  0.3900  0.4001
stock_blend_600 + true day effect            0.4120  0.4019  0.3895  0.4012
stock_blend_900 + true day effect            0.4139  0.4033  0.3896  0.4022
stock_blend_600 leaves127 + true day effect  0.4130  0.4036  0.3912  0.4026
stock_l2_300 + true day effect               0.4168  0.4073  0.3935  0.4059
stock_l2_600 + true day effect               0.4211  0.4112  0.3939  0.4087
stock_l2_900 + true day effect               0.4250  0.4146  0.3946  0.4114
FULL cal main                                0.4341  0.4241  0.4089  0.4224
FULL cal trend2                              0.4284  0.4226  0.4164  0.4225
FULL cal trend0                              0.4457  0.4331  0.4087  0.4292
FULL nocal main                              0.4583  0.4423  0.4351  0.4453
FULL nocal trend2                            0.4528  0.4461  0.4445  0.4478
FULL nocal trend0                            0.4699  0.4472  0.4334  0.4502
ref_lin_gbm                                  0.5528  0.5086  0.4927  0.5180
ref_pid_mean                                 0.5902  0.5914  0.5814  0.5876
ref_linear                                   0.6069  0.5860  0.5805  0.5911

## Validation, mean residual (target - prediction)
fold                                           457    557    657   mean
model                                                                  
ref_linear                                  -0.153 -0.048 -0.068 -0.090
ref_lin_gbm                                 -0.218  0.194 -0.078 -0.034
stock_l2_900 + true day effect              -0.057 -0.043  0.012 -0.029
stock_l2_600 + true day effect              -0.052 -0.039  0.007 -0.028
stock_l2_300 + true day effect              -0.046 -0.029  0.005 -0.024
stock_blend_900 + true day effect           -0.033 -0.010  0.030 -0.004
stock_blend_600 + true day effect           -0.029 -0.008  0.026 -0.004
stock_blend_300 + true day effect           -0.026 -0.000  0.025 -0.001
stock_blend_600 leaves127 + true day effect -0.026 -0.002  0.028 -0.000
FULL nocal trend2                            0.094  0.012 -0.044  0.021
stock_l1_900 + true day effect              -0.009  0.023  0.048  0.021
stock_l1_600 + true day effect              -0.007  0.024  0.046  0.021
stock_l1_300 + true day effect              -0.006  0.028  0.044  0.022
FULL cal trend2                              0.079  0.049 -0.042  0.029
FULL nocal main                              0.150  0.079  0.020  0.083
FULL cal main                                0.133  0.111  0.020  0.088
FULL nocal trend0                            0.206  0.147  0.084  0.146
FULL cal trend0                              0.188  0.173  0.081  0.147
ref_pid_mean                                 0.281  0.315  0.303  0.300

## Day model alone: MSE of the predicted day effect
origin                           457     557     657    mean
day_model             level                                 
cal                   main    0.0594  0.0410  0.0248  0.0417
                      trend0  0.0841  0.0608  0.0287  0.0579
                      trend2  0.0420  0.0305  0.0298  0.0341
constant (train mean) -       0.1966  0.2048  0.1948  0.1987
nocal                 main    0.1124  0.0759  0.0746  0.0877
                      trend0  0.1398  0.0942  0.0794  0.1045
                      trend2  0.0928  0.0689  0.0795  0.0804

## Day model alone: mean error of the level (true - predicted)
origin              457    557    657
day_model level                      
cal       main    0.162  0.118 -0.007
          trend0  0.217  0.180  0.055
          trend2  0.108  0.057 -0.068
nocal     main    0.180  0.087 -0.006
          trend0  0.235  0.155  0.058
          trend2  0.124  0.020 -0.070

## Paired tests by day, Newey-West (negative = full model better)
 origin             A            B loss  mean_diff   t_stat  share_days_A_better
    457 FULL cal main  ref_lin_gbm  mse    -0.1425 -10.8531               0.8357
    457 FULL cal main  ref_lin_gbm  mae    -0.1188 -16.1886               0.8790
    457 FULL cal main ref_pid_mean  mse    -0.2775 -10.7164               0.9942
    457 FULL cal main ref_pid_mean  mae    -0.1564 -16.4174               0.9885
    557 FULL cal main  ref_lin_gbm  mse    -0.1406  -7.7382               0.9271
    557 FULL cal main  ref_lin_gbm  mae    -0.0851 -12.4449               0.9312
    557 FULL cal main ref_pid_mean  mse    -0.3000  -9.7383               0.9879
    557 FULL cal main ref_pid_mean  mae    -0.1678 -14.8935               0.9879
    657 FULL cal main  ref_lin_gbm  mse    -0.1145  -5.5413               0.9252
    657 FULL cal main  ref_lin_gbm  mae    -0.0844 -11.8098               0.9388
    657 FULL cal main ref_pid_mean  mse    -0.3145  -6.8916               0.9932
    657 FULL cal main ref_pid_mean  mae    -0.1737 -10.9202               0.9932

## MSE by horizon (days after the end of training)
horizon               (0, 50]  (50, 100]  (100, 150]  (150, 400]
model         origin                                            
FULL cal main 457      0.2924     0.3509      0.3381      0.3618
              557      0.3104     0.3343      0.2897      0.3520
              657      0.2633     0.3089      0.3402         NaN
ref_lin_gbm   457      0.4348     0.5840      0.4821      0.4802
              557      0.4413     0.4970      0.4093      0.4930
              657      0.3676     0.3961      0.4893         NaN
ref_pid_mean  457      0.4604     0.6083      0.5753      0.6793
              557      0.5442     0.6071      0.5771      0.7005
              657      0.5347     0.5835      0.7308         NaN

## Day effects (target units), fitted up to the last origin
witch        1.472
russell      1.664
eom          0.718
eoq          0.205
msci         0.023
msci_semi    0.139
opex_m       0.070
first_fri    0.048
pre_hol      0.035
post_hol     0.114
dow0        -0.001
dow1        -0.016
dow2        -0.051
dow3        -0.056
half        -0.268
eom_half    -0.685
nlv_d20     -0.030
lr_d20      -0.002

## Top 25 features of the stock model (gain share)
lin250          0.1678
lin60           0.1126
lin2000         0.0821
te2000          0.0300
log_r_sum_xd    0.0258
log_r_sum_xr    0.0223
v_com_m60       0.0219
NLV_xd          0.0170
NLV_xr          0.0162
v_com_m20       0.0152
NLV_d120        0.0135
v0_m60          0.0128
LS_m250         0.0107
NLV             0.0106
v_last7_m60     0.0105
NLV_dexp        0.0104
NLV_d60         0.0096
NLV_m120        0.0095
te_ev           0.0091
NLV_d5          0.0088
NLV_m250        0.0084
NLV_d20         0.0078
NLV_m60         0.0075
r_sum_m60       0.0075
LS_m60          0.0075

number of trees: {300: np.float64(0.2911924355922766), 600: np.float64(0.2916367721394148), 900: np.float64(0.2927575035589962)} -> 300 used for the final fit

## Final day model: anchor -1.762 at day 774, slope 0.0671 per 100 days
submission_main.csv: 41,565 rows, mean -1.5385, std 0.6414, level first/last test day -1.742 / -1.510
submission_trend0.csv: 41,565 rows, mean -1.6749, std 0.6374, level first/last test day -1.762 / -1.762
submission_trend2.csv: 41,565 rows, mean -1.4021, std 0.6522, level first/last test day -1.721 / -1.257
training target for comparison: mean -1.9683, std 0.9099, last 60 training days mean -1.6782
[   256s] final fit done
