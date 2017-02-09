import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.linear_model import Lasso
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
#   Two function to detect and remove dataset entries that are identified as outliers,
#   it can be tuned with a threshold parameter. detect_outliers take in input a dataframe
#   and a threshold and return two list, each containing the outliers of trainset and testset
# =======================================================================================================
def detect_outliers(df, std_thresh=5):
    index_train = set()
    index_test = set()
    for feature in df.columns:
        mean = np.mean(df[feature].values)
        std = np.std(df[feature].values)
        for index, value in df[feature].iteritems():
            if np.abs((value - mean)/std) > std_thresh:
                if index <= 1460:
                    index_train.add(index)
                else:
                    index_test.add(index)
    return list(index_train), list(index_test)


# =======================================================================================================
#   Very simple function that take in input a pandas' dataframe and a list of outliers
#   return the dataframe with rows corresponding to the outliers ID dropped
# =======================================================================================================
def remove_outliers(df, outlier_list):
    tuned_list = []
    for x in outlier_list:
        tuned_list.append(x-1)
    return df.drop(df.index[tuned_list], axis=0)


# =======================================================================================================
#   function compute_skewness take a pandas' dataframe and computes the skewness for each columns,
#   return a pandas' Series containing the skewness indexed with columns names.
# =======================================================================================================
def compute_skewness(dataframe):
    skewness = []
    for column in dataframe:
        skewness.append(skew(dataframe[column].dropna()))
    s = pd.Series(skewness, index=dataframe.columns)
    return s


# =======================================================================================================
#   Utility function to use paired with compute_skewness, select the skewed features according to
#   threshold(skew_thresh) specified
# =======================================================================================================
def select_skewed_features(series, skew_thresh=0.75):
    skewed = []
    for col in series.index:
        if series[col] > skew_thresh:
            skewed.append(col)
    return skewed


# =======================================================================================================
#   This function search the optimum alpha for the lasso regression,
#   at any step try with a different alpha and compute a cross validation on the dataset.
#   Is taken the alpha with max cross validation score associated.
#   The function take a train set X, a test vector y (with the sale prices) and as an optional parameter
#   an array containing alphas to try.
# =======================================================================================================

def opt_alpha_cv(X, y, test_range=np.arange(0.0001, 0.0005, 0.00001)):
    r2_arr = []
    alpha_arr = []
    for alpha in test_range:
        ridge_model = Lasso(alpha).fit(X, y)
        current_r2 = cross_val_score(ridge_model, X, y).mean()
        r2_arr.append(current_r2)
        alpha_arr.append(alpha)
    opt_alpha_idx = np.argmax(r2_arr)
    opt_alpha = alpha_arr[opt_alpha_idx]
    r2_min = r2_arr[opt_alpha_idx]
    return alpha_arr, r2_arr, opt_alpha, r2_min


# =======================================================================================================
#   Like its name suggests this utility function convert categorical feature into a numerical version.
#   Requires a target dataframe, the feature that has to be converted and a list of possible values that
#   the feature can take. Return a new dataframe with requested variable converted.
#
#   IT'S NOT ACTUALLY USED BECAUSE IT SEEMS TO RUIN THE PREDICTION (KAGGLE SCORE DECREASE)
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
#   Function used to plot the R^2 with respect of alpha of the Lasso model
# =======================================================================================================
def pplot(data, dotx, doty):
    plt.rcParams['figure.figsize'] = (12.0, 6.0)
    fig, ax = plt.subplots()
    ax.plot(data)
    ax.plot(dotx, doty, 'ro')
    ax.annotate(r'$max(R^2)='+str(np.round(doty, decimals=4))+'$', xy=(dotx, doty), xytext=(-20, 20),
                textcoords='offset points', ha='left', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.6),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    ax.grid(True)
    plt.title('Cross Validation Lasso Model')
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$R^2$')
    plt.xlim([0.0001, 0.0025])
    plt.ylim([0.910, 0.927])

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


