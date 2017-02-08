import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.linear_model import Ridge
from sklearn.cross_validation import cross_val_score
from matplotlib import pyplot as plt



# =======================================================================================================
#   Very simple function that take a pandas' dataframe and remove
#   any columns that exceed a certain ratio(default 1/2) of NaN among its entries.
# =======================================================================================================
def remove_nan(dataframe, threshold=1/2):
    size = len(dataframe)
    dataframe.dropna(thresh=size - np.floor(size*threshold), axis=1, inplace=True)


# =======================================================================================================
#   function compute_skewness take a pandas' dataframe and computes skewness for each columns,
#   return a pandas' Series containing the skewness indexed with columns names.
# =======================================================================================================
def compute_skewness(dataframe):
    skewness = []
    for column in dataframe:
        skewness.append(skew(dataframe[column].dropna()))
    s = pd.Series(skewness, index=dataframe.columns)
    return s


# =======================================================================================================
#   function select_skewed_features take a pandas' dataframe and computes skewness for each columns,
#   return a pandas' Series containing the skewness indexed with columns names.
# =======================================================================================================
def select_skewed_features(series, skew_thresh=0.75):
    skewed = []
    for col in series.index:
        if series[col] > skew_thresh:
            skewed.append(col)
    return skewed


# =======================================================================================================
#   This function performs a cross validation to evaluate our prediction model
# =======================================================================================================
def cv(X, y, test_range=np.arange(1, 10+0.1, 0.1)):
    rmse = []
    min_rse_opt_alpha = (np.inf, 0)
    for alpha in test_range:
        current_rmse = rmse_cv(Ridge(alpha=alpha), X, y).mean()
        rmse.append(current_rmse)
        if current_rmse < min_rse_opt_alpha[0]:
            min_rse_opt_alpha = (current_rmse, alpha)
        print(min_rse_opt_alpha, "current alpha:", alpha)
    return min_rse_opt_alpha, rmse


def rmse_cv(model, X, y):
    return np.sqrt(-cross_val_score(model, X, y, cv=50, scoring='mean_squared_error'))


# =======================================================================================================
#   Like its name suggests this utility function convert categorical feature into a numerical version.
#   Requires a target dataframe, the feature that has to be converted and a list of possible values that
#   the feature can take. Return a new dataframe with requested variable converted.
#
#   IT'S NOT ACTUALLY IN USE BECAUSE IT DOES NOT IMPROVE THE KAGGLE SCORE
# =======================================================================================================
def convert_feature_cat_to_num(dataframe, feature, value_list):
    for index, row in dataframe.iterrows():
        if dataframe.loc[index, feature] not in value_list:
            dataframe.loc[index, feature] = 0
        else:
            dataframe.loc[index, feature] = value_list.index(row[feature]) + 1
    return dataframe


def bulk_convert_cat_to_num(dataframe):
    dataframe = convert_feature_cat_to_num(dataframe, "ExterQual", ["Po", "Fa", "TA", "Gd", "Ex"])
    dataframe = convert_feature_cat_to_num(dataframe, "Street", ["Grvl", "Pave"])
    dataframe = convert_feature_cat_to_num(dataframe, "LandContour", ["Low", "HLS", "Bnk", "Lvl"])
    dataframe = convert_feature_cat_to_num(dataframe, "Utilities", ["ELO", "NoSeWa", "NoSewr", "AllPub"])
    dataframe = convert_feature_cat_to_num(dataframe, "ExterCond", ["Po", "Fa", "TA", "Gd", "Ex"])
    dataframe = convert_feature_cat_to_num(dataframe, "BsmtQual", ["Po", "Fa", "TA", "Gd", "Ex"])
    dataframe = convert_feature_cat_to_num(dataframe, "BsmtCond", ["Po", "Fa", "TA", "Gd", "Ex"])
    dataframe = convert_feature_cat_to_num(dataframe, "BsmtExposure", ["Mn", "Av", "Gd"])
    dataframe = convert_feature_cat_to_num(dataframe, "BsmtFinType1", ["Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"])
    dataframe = convert_feature_cat_to_num(dataframe, "BsmtFinType2", ["Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"])
    dataframe = convert_feature_cat_to_num(dataframe, "HeatingQC", ["Po", "Fa", "TA", "Gd", "Ex"])
    dataframe = convert_feature_cat_to_num(dataframe, "CentralAir", ["No", "Yes"])
    dataframe = convert_feature_cat_to_num(dataframe, "KitchenQual", ["Po", "Fa", "TA", "Gd", "Ex"])
    dataframe = convert_feature_cat_to_num(dataframe, "Functional", ["Sal", "Sev", "Maj2", "Maj1",
                                                                     "Mod", "Min2", "Min1", "Typ"])
    dataframe = convert_feature_cat_to_num(dataframe, "GarageFinish", ["Unf", "RFn", "Fin"])
    dataframe = convert_feature_cat_to_num(dataframe, "GarageQual", ["Po", "Fa", "TA", "Gd", "Ex"])
    dataframe = convert_feature_cat_to_num(dataframe, "GarageCond", ["Po", "Fa", "TA", "Gd", "Ex"])
    dataframe = convert_feature_cat_to_num(dataframe, "PavedDrive", ["N", "P", "Y"])
    return dataframe


# =======================================================================================================
#   Function used to plot the RMSE with respect of alpha of the Ridge model
# =======================================================================================================
def pplot(data, dotx, doty):
    fig, ax = plt.subplots()
    ax.plot(data)
    ax.plot(dotx, doty, 'ro')
    ax.annotate("min RMSE="+str(np.round(doty, decimals=4)), xy=(dotx, doty), xytext=(-20, 20),
                textcoords='offset points', ha='left', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.6),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    ax.grid(True)
    plt.title('Cross Validation Ridge Model')
    plt.xlabel("alpha")
    plt.ylabel("RMSE")
    plt.xlim([0, 10])
    plt.ylim([0.1204, 0.1217])

    ticklines = ax.get_xticklines() + ax.get_yticklines()
    gridlines = ax.get_xgridlines() + ax.get_ygridlines()
    ticklabels = ax.get_xticklabels() + ax.get_yticklabels()

    for line in ticklines:
        line.set_linewidth(3)

    for line in gridlines:
        line.set_linestyle('-.')

    for label in ticklabels:
        label.set_color('k')
        label.set_fontsize('medium')

    plt.show()
