# test_range = np.append(np.arange(1, 10+0.1, 0.1), np.arange(11, 41))
# (opt_alpha, min_rmse), rmse_list = lib.cv(X, y, test_range=test_range)
# print("OPT --", opt_alpha, min_rmse)
#
# cv_ridge = pd.Series(rmse_list, index=test_range)
#
# #plot the cross validation
# lib.pplot(cv_ridge, min_rmse, opt_alpha)
