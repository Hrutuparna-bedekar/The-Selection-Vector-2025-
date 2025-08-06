import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import FunctionTransformer , LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans
from itertools import combinations
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import Pipeline as ImbPipeline   
from imblearn.over_sampling import SMOTE   
from sklearn.linear_model import LogisticRegression   
from splitter import split_columns
import imblearn
import re


#==============Your Name======================
# Your code
#==============Your Name=====================
#Dont remove the following snippet and follow the same

#==============Hrutuparna Bedekar======================


def hb_dropper(X):
    X=X.copy()
    X=X.drop(columns=['length_mm','width_mm','surface_area_mm2','height_mm','carat_weight','price_per_carat'])  
    cols=X.columns
    si_mean=SimpleImputer(strategy='mean')
    X=si_mean.fit_transform(X)
    X=pd.DataFrame(X,columns=cols)
    return X

    
    
hb_drop_tra=FunctionTransformer(hb_dropper)
#==============Hrutuparna Bedekar=====================
