import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.linear_model import Ridge, RidgeCV
from sklearn import cross_validation



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
def cv(X, y, test_range=100):
    scores = []
    for alpha in np.arange(0, test_range):
        ridge_cv = Ridge(alpha).fit(X, y)
        scores.append(cross_validation.cross_val_score(ridge_cv, X, y, cv=10, scoring='mean_squared_error').mean())
    opt_alpha = np.argmax(scores)+1
    return opt_alpha, scores


# def remove_outliers(df):
#     # computing percentile
#     low = .05
#     high = .95
#     quant_df = df.quantile([low, high])
#     # filtering dataset
#     filt_df = df.apply(lambda x: x[(x > quant_df.loc[low, x.name]) & (x < quant_df.loc[high, x.name])], axis=0)
#     print(filt_df.head())

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
